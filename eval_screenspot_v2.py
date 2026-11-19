# job_05_phase7_subpatch_and_cascade.py
# Phase 7: Final Paper-Complete Benchmark on visjev_phase6_best.pt (Zero retraining, ~15 min on L4!)
#   1. Sub-Patch 3x3 Soft-Argmax Continuous Refinement vs Discrete Patch Argmax (0 ms extra latency)
#   2. 3-Tier Confidence-Gated Cascade (System 1 Single-Pass -> System 1.5 InnerZoom -> System 2 Autoregressive Fallback)
#      Target: Match/BEAT Qwen2.5-VL-3B Autoregressive (~87%) at 4x-6x LOWER latency + 0.97+ Abstention AUROC!
#   3. Full 1,272-sample ScreenSpot-v2 Official Table for visjev_phase6_best.pt (Web/Desktop/Mobile x Text/Icon)

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import torch, os, time, math, json, re, random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from datasets import load_dataset
from sklearn.metrics import roc_auc_score
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

import peft.import_utils as _peft_iu
import peft.tuners.lora.torchao as _peft_tao
_peft_iu.is_torchao_available = lambda: False
_peft_tao.is_torchao_available = lambda: False
from peft import LoraConfig, get_peft_model

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

DEVICE = "cuda"
DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
LOGS_DIR = os.environ.get("VISJEV_LOGS_DIR", ".")
CKPT_PATH = os.path.join(LOGS_DIR, "visjev_phase6_best.pt")

gpu_name = torch.cuda.get_device_name(0)
print(f"[Phase 7 Sub-Patch + 3-Tier Cascade + Full 1272 Eval] GPU: {gpu_name}", flush=True)

MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
MIN_PIXELS = 256 * 28 * 28
HIRES_PIXELS = 768 * 28 * 28

processor = AutoProcessor.from_pretrained(MODEL_ID, min_pixels=MIN_PIXELS, max_pixels=HIRES_PIXELS)
backbone = Qwen2_5_VLForConditionalGeneration.from_pretrained(MODEL_ID, dtype=DTYPE, device_map="auto")

text_cfg = getattr(backbone.config, "text_config", backbone.config)
vision_cfg = getattr(backbone.config, "vision_config", None)
HIDDEN_SIZE = getattr(text_cfg, "hidden_size", 2048)
NUM_LAYERS = getattr(text_cfg, "num_hidden_layers", 36)
NUM_HEADS = getattr(text_cfg, "num_attention_heads", 16)
NUM_KV_HEADS = getattr(text_cfg, "num_key_value_heads", 2)
HEAD_DIM = HIDDEN_SIZE // NUM_HEADS
IMAGE_TOKEN_ID = getattr(backbone.config, "image_token_id", 151655)
SPATIAL_MERGE = getattr(vision_cfg, "spatial_merge_size", 2) if vision_cfg is not None else 2
PATCH_SIZE = getattr(vision_cfg, "patch_size", 14) if vision_cfg is not None else 14

def get_decoder_layers(model):
    for _, module in model.named_modules():
        if isinstance(module, nn.ModuleList) and len(module) == NUM_LAYERS:
            if hasattr(module[0], "self_attn") and hasattr(module[0], "input_layernorm"):
                return module
    raise AttributeError("Cannot locate decoder layers!")

TOP_LORA_LAYERS = list(range(NUM_LAYERS - 16, NUM_LAYERS))
lora_cfg = LoraConfig(
    r=24, lora_alpha=48, lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    layers_pattern="layers", layers_to_transform=TOP_LORA_LAYERS, bias="none",
)
backbone = get_peft_model(backbone, lora_cfg)

POINTER_LAYERS = list(range(NUM_LAYERS - 12, NUM_LAYERS))
N_PTR_HEADS = len(POINTER_LAYERS) * NUM_HEADS
KV_REPEAT = NUM_HEADS // NUM_KV_HEADS

def patch_index_to_norm_xy(idx, H_grid, W_grid):
    r, c = idx // W_grid, idx % W_grid
    return (c + 0.5) / W_grid, (r + 0.5) / H_grid

def subpatch_soft_argmax_xy(patch_probs, H_grid, W_grid):
    '''
    Zero-cost continuous coordinate refinement:
    Computes probability-weighted 2D center of mass inside the 3x3 neighborhood around the peak patch.
    '''
    pred_idx = int(patch_probs.argmax().item())
    pr, pc = pred_idx // W_grid, pred_idx % W_grid
    grid_p = patch_probs.view(H_grid, W_grid)
    r0, r1 = max(0, pr - 1), min(H_grid, pr + 2)
    c0, c1 = max(0, pc - 1), min(W_grid, pc + 2)
    sub = grid_p[r0:r1, c0:c1]
    w_sum = float(sub.sum().item())
    if w_sum <= 1e-8:
        return patch_index_to_norm_xy(pred_idx, H_grid, W_grid), w_sum
    rs = torch.arange(r0, r1, device=patch_probs.device, dtype=torch.float32)
    cs = torch.arange(c0, c1, device=patch_probs.device, dtype=torch.float32)
    gy, gx = torch.meshgrid(rs, cs, indexing="ij")
    cx = float(((gx + 0.5) * sub).sum().item() / w_sum) / W_grid
    cy = float(((gy + 0.5) * sub).sum().item() / w_sum) / H_grid
    return (cx, cy), w_sum

def extract_qk_pointer_maps(outputs, input_ids, model):
    layers = get_decoder_layers(model)
    img_mask = (input_ids == IMAGE_TOKEN_ID)
    img_idx = torch.where(img_mask)[0]
    after_img = int(img_idx[-1].item()) + 1
    all_scores = []
    scale = 1.0 / math.sqrt(HEAD_DIM)
    for l_idx in POINTER_LAYERS:
        layer = layers[l_idx]
        h_in = outputs.hidden_states[l_idx][0]
        q_norm = layer.input_layernorm(h_in[-1:])
        k_norm = layer.input_layernorm(h_in[img_mask])
        q = layer.self_attn.q_proj(q_norm).view(NUM_HEADS, HEAD_DIM).float()
        k = layer.self_attn.k_proj(k_norm).view(-1, NUM_KV_HEADS, HEAD_DIM).float()
        k = k.repeat_interleave(KV_REPEAT, dim=1)
        all_scores.append(torch.einsum("hd,nhd->hn", q, k) * scale)
    attn_maps = torch.cat(all_scores, dim=0)
    h_last = outputs.hidden_states[-1][0]
    return attn_maps, h_last[img_mask].float(), (0.5 * h_last[-1] + 0.5 * h_last[after_img:].mean(dim=0)).float()

class VisJevAttnPatchHeadV3(nn.Module):
    def __init__(self, hidden_size, n_ptr_heads, proj_dim=256):
        super().__init__()
        self.head_router = nn.Sequential(
            nn.LayerNorm(hidden_size), nn.Linear(hidden_size, 256), nn.GELU(), nn.Linear(256, n_ptr_heads),
        )
        self.head_prior = nn.Parameter(torch.zeros(n_ptr_heads))
        self.attn_temp = nn.Parameter(torch.tensor(1.5))
        self.q_res = nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, proj_dim))
        self.v_res = nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, proj_dim))
        self.pos_mlp = nn.Sequential(nn.Linear(2, 32), nn.GELU(), nn.Linear(32, proj_dim))
        self.res_scale = nn.Parameter(torch.tensor(0.5))
        self.abstain_head = nn.Sequential(nn.Linear(hidden_size * 2 + 2, 256), nn.GELU(), nn.Linear(256, 1))

    def forward(self, attn_maps, h_vis, h_q, H_grid, W_grid):
        device = attn_maps.device
        m_mean = attn_maps.mean(dim=-1, keepdim=True)
        m_std = attn_maps.std(dim=-1, keepdim=True).clamp(min=1e-4)
        norm_maps = (attn_maps - m_mean) / m_std
        gate_w = F.softmax(self.head_router(h_q) + self.head_prior, dim=0)
        ptr_logits = (gate_w.unsqueeze(1) * norm_maps).sum(dim=0) * self.attn_temp.clamp(min=0.2, max=12.0)

        ys = (torch.arange(H_grid, device=device, dtype=torch.float32) + 0.5) / H_grid
        xs = (torch.arange(W_grid, device=device, dtype=torch.float32) + 0.5) / W_grid
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        coords = torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=-1)
        qr = F.normalize(self.q_res(h_q), dim=-1)
        vr = F.normalize(self.v_res(h_vis) + self.pos_mlp(coords), dim=-1)
        res_logits = (vr @ qr) * (self.res_scale * 10.0)

        patch_logits = ptr_logits + res_logits
        patch_p = F.softmax(patch_logits, dim=-1)
        v_att = (patch_p.unsqueeze(1) * h_vis).sum(dim=0)
        max_logit = patch_logits.max().unsqueeze(0)
        entropy = -(patch_p * (patch_p.clamp(min=1e-8).log())).sum().unsqueeze(0)
        abstain_logit = self.abstain_head(torch.cat([h_q, v_att, max_logit, entropy], dim=0))
        return torch.cat([patch_logits, abstain_logit], dim=0)

head = VisJevAttnPatchHeadV3(HIDDEN_SIZE, N_PTR_HEADS, proj_dim=256).to(DEVICE)
ckpt = torch.load(CKPT_PATH, map_location="cpu")
backbone.load_state_dict(ckpt["lora"], strict=False)
head.load_state_dict(ckpt["head"], strict=True)
backbone.eval(); head.eval()

# Load Platt calibration from Phase 6
PA, PB = 0.352, 1.991
p6_json = os.path.join(LOGS_DIR, "visjev_phase6_results.json")
if os.path.exists(p6_json):
    with open(p6_json) as f:
        p6m = json.load(f)
    if "platt" in p6m:
        PA, PB = p6m["platt"]["a"], p6m["platt"]["b"]

def cal_conf(c3):
    eps = 1e-4
    c = min(1.0 - eps, max(eps, c3))
    return 1.0 / (1.0 + math.exp(-(PA * math.log(c / (1.0 - c)) + PB)))

def parse_qwen25_bbox_center(text, W_proc, H_proc):
    m4 = re.search(r'"bbox_2d"\s*:\s*\[\s*([0-9.]+)\s*,\s*([0-9.]+)\s*,\s*([0-9.]+)\s*,\s*([0-9.]+)\s*\]', text)
    if m4:
        x1, y1, x2, y2 = [float(m4.group(i)) for i in range(1, 5)]
        cx, cy = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
    else:
        nums = re.findall(r'[-+]?\d*\.?\d+', text)
        if len(nums) >= 4:
            x1, y1, x2, y2 = [float(v) for v in nums[:4]]
            cx, cy = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
        elif len(nums) >= 2:
            cx, cy = float(nums[0]), float(nums[1])
        else:
            return None
    if cx <= 1.0 and cy <= 1.0:
        return cx, cy
    return cx / float(W_proc), cy / float(H_proc)

@torch.no_grad()
def infer_single_pass(img, instruction):
    prompt = f"Locate the UI element: {instruction}, report the bbox coordinates in JSON format."
    messages = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[img], return_tensors="pt").to(DEVICE)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    outputs = backbone(**inputs, output_hidden_states=True, use_cache=False)
    attn_maps, h_vis, h_q = extract_qk_pointer_maps(outputs, inputs.input_ids[0], backbone)
    grid_thw = inputs.image_grid_thw[0].cpu().tolist()
    Hg, Wg = grid_thw[1] // SPATIAL_MERGE, grid_thw[2] // SPATIAL_MERGE
    logits = head(attn_maps, h_vis, h_q, Hg, Wg)
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) * 1000.0
    return logits, Hg, Wg, ms

@torch.no_grad()
def infer_system2_ar(img, instruction):
    # Temporarily disable LoRA adapter so System 2 uses pure untouched pretrained Qwen2.5-VL-3B!
    prompt = f"Locate the UI element: {instruction}, report the bbox coordinates in JSON format."
    messages = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[img], return_tensors="pt").to(DEVICE)
    grid_thw = inputs.image_grid_thw[0].cpu().tolist()
    H_proc, W_proc = grid_thw[1] * PATCH_SIZE, grid_thw[2] * PATCH_SIZE
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with backbone.disable_adapter():
        out_ids = backbone.generate(**inputs, max_new_tokens=48, do_sample=False)
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) * 1000.0
    gen = processor.batch_decode(out_ids[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0].strip()
    return parse_qwen25_bbox_center(gen, W_proc, H_proc), ms

# Load ScreenSpot-v2
ss_ds = load_dataset("lmms-lab/ScreenSpot-v2", split="train")
ss_all = []
for row in ss_ds:
    img = row["image"]
    if not isinstance(img, Image.Image):
        continue
    img = img.convert("RGB")
    W, H = img.size
    x, y, w, h = [float(v) for v in row["bbox"]]
    ss_all.append({
        "image": img,
        "instruction": str(row["instruction"]).strip(),
        "bbox_norm": [max(0.0, min(1.0, x/W)), max(0.0, min(1.0, y/H)), max(0.0, min(1.0, (x+w)/W)), max(0.0, min(1.0, (y+h)/H))],
        "data_type": str(row.get("data_type", "unknown")).lower(),
        "data_source": str(row.get("data_source", "unknown")).lower(),
    })

rng42 = random.Random(42)
ss_shuf = list(ss_all)
rng42.shuffle(ss_shuf)
eval_300 = ss_shuf[:300]

# ===== EXP 1: SUB-PATCH SOFT-ARGMAX + 3-TIER CONFIDENCE-GATED CASCADE ON n=300 =====
print("\n[Exp 1] Running Sub-Patch Soft-Argmax + 3-Tier Cascade (System 1 -> 1.5 -> 2) on n=300...", flush=True)
disc_hits, sub_hits = 0, 0
s1_lats, records = [], []

for idx, s in enumerate(eval_300):
    lg, Hg, Wg, ms1 = infer_single_pass(s["image"], s["instruction"])
    probs = F.softmax(lg, dim=-1)
    pp = probs[:-1]
    p_abs = float(probs[-1].item())
    x1, y1, x2, y2 = s["bbox_norm"]

    # Discrete argmax
    d_idx = int(pp.argmax().item())
    dx, dy = patch_index_to_norm_xy(d_idx, Hg, Wg)
    h_disc = int(x1 <= dx <= x2 and y1 <= dy <= y2)

    # Sub-patch 3x3 soft-argmax
    (sx, sy), c3 = subpatch_soft_argmax_xy(pp, Hg, Wg)
    h_sub = int(x1 <= sx <= x2 and y1 <= sy <= y2)
    cc = cal_conf(c3)

    disc_hits += h_disc
    sub_hits += h_sub
    s1_lats.append(ms1)
    records.append({
        "idx": idx, "sx": sx, "sy": sy, "cc": cc, "p_abs": p_abs,
        "h_sub": h_sub, "ms1": ms1, "type": s["data_type"],
    })

print(f"  Discrete Argmax Acc (n=300): {100*disc_hits/300:.2f}% | Sub-Patch Soft-Argmax Acc: {100*sub_hits/300:.2f}% (Delta: {100*(sub_hits-disc_hits)/300:+.2f}%)", flush=True)

# Cache System 1.5 (InnerZoom) & System 2 (Untouched Pretrained AR) for all uncertain samples (cc < 0.88)
zoom_map, ar_map = {}, {}
for r in records:
    if r["cc"] < 0.88:
        s = eval_300[r["idx"]]
        img = s["image"]; W, H = img.size
        cx1, cy1 = max(0.0, min(0.5, r["sx"] - 0.25)), max(0.0, min(0.5, r["sy"] - 0.25))
        crop = img.crop((int(cx1*W), int(cy1*H), int((cx1+0.5)*W), int((cy1+0.5)*H)))
        lg2, Hg2, Wg2, ms2 = infer_single_pass(crop, s["instruction"])
        (lx, ly), c3_2 = subpatch_soft_argmax_xy(F.softmax(lg2, dim=-1)[:-1], Hg2, Wg2)
        gx, gy = cx1 + lx * 0.5, cy1 + ly * 0.5
        x1, y1, x2, y2 = s["bbox_norm"]
        zoom_map[r["idx"]] = (int(x1 <= gx <= x2 and y1 <= gy <= y2), ms2, cal_conf(c3_2))

        xy_ar, ms_ar = infer_system2_ar(img, s["instruction"])
        h_ar = int(xy_ar is not None and x1 <= xy_ar[0] <= x2 and y1 <= xy_ar[1] <= y2)
        ar_map[r["idx"]] = (h_ar, ms_ar)

cascade_table = []
# Sweep operating points: (t_zoom, t_ar)
for (t_zoom, t_ar) in [(0.82, 0.0), (0.85, 0.0), (0.85, 0.78), (0.86, 0.82), (0.88, 0.85)]:
    hits_c, lats_c, n_z, n_ar = 0, [], 0, 0
    for r in records:
        cc = r["cc"]
        if cc >= t_zoom:
            hits_c += r["h_sub"]; lats_c.append(r["ms1"])
        elif cc >= t_ar:
            n_z += 1
            hz, ms2, _ = zoom_map[r["idx"]]
            hits_c += hz; lats_c.append(r["ms1"] + ms2)
        else:
            n_ar += 1
            ha, ms_ar = ar_map[r["idx"]]
            hits_c += ha; lats_c.append(r["ms1"] + ms_ar)
    row_res = {
        "t_zoom": t_zoom, "t_ar": t_ar,
        "acc": round(100.0 * hits_c / 300, 2),
        "s1_pct": round(100.0 * (300 - n_z - n_ar) / 300, 1),
        "zoom_pct": round(100.0 * n_z / 300, 1),
        "ar_pct": round(100.0 * n_ar / 300, 1),
        "median_ms": round(float(np.median(lats_c)), 1),
        "mean_ms": round(float(np.mean(lats_c)), 1),
        "speedup_mean_vs_2151ms": round(2151.2 / float(np.mean(lats_c)), 2),
    }
    cascade_table.append(row_res)
    print(f"  Cascade (z<{t_zoom}, ar<{t_ar}): Acc={row_res['acc']:.2f}% | S1={row_res['s1_pct']}% Z={row_res['zoom_pct']}% AR={row_res['ar_pct']}% | Med={row_res['median_ms']}ms Mean={row_res['mean_ms']}ms ({row_res['speedup_mean_vs_2151ms']}x)", flush=True)

# ===== EXP 2: FULL 1,272-SAMPLE SCREENSPOT-V2 OFFICIAL BENCHMARK FOR PHASE 6 MODEL =====
print(f"\n[Exp 2] Running FULL 1,272-sample ScreenSpot-v2 Eval for Phase 6 (Sub-Patch + InnerZoom 0.85)...", flush=True)
f_disc, f_sub, f_zoom, f_esc = 0, 0, 0, 0
f_lats1, f_latsz = [], []
buckets = {}

for idx, s in enumerate(ss_all):
    lg, Hg, Wg, ms1 = infer_single_pass(s["image"], s["instruction"])
    probs = F.softmax(lg, dim=-1)
    pp = probs[:-1]
    p_abs = float(probs[-1].item())
    x1, y1, x2, y2 = s["bbox_norm"]

    dx, dy = patch_index_to_norm_xy(int(pp.argmax().item()), Hg, Wg)
    hd = int(x1 <= dx <= x2 and y1 <= dy <= y2)
    (sx, sy), c3 = subpatch_soft_argmax_xy(pp, Hg, Wg)
    hs = int(x1 <= sx <= x2 and y1 <= sy <= y2)
    cc = cal_conf(c3)

    hz, msz = hs, ms1
    if cc < 0.85 and p_abs < 0.5:
        f_esc += 1
        img = s["image"]; W, H = img.size
        cx1, cy1 = max(0.0, min(0.5, sx - 0.25)), max(0.0, min(0.5, sy - 0.25))
        crop = img.crop((int(cx1*W), int(cy1*H), int((cx1+0.5)*W), int((cy1+0.5)*H)))
        lg2, Hg2, Wg2, ms2 = infer_single_pass(crop, s["instruction"])
        (lx, ly), _ = subpatch_soft_argmax_xy(F.softmax(lg2, dim=-1)[:-1], Hg2, Wg2)
        gx, gy = cx1 + lx * 0.5, cy1 + ly * 0.5
        hz = int(x1 <= gx <= x2 and y1 <= gy <= y2)
        msz = ms1 + ms2

    f_disc += hd; f_sub += hs; f_zoom += hz
    f_lats1.append(ms1); f_latsz.append(msz)
    k = f"{s['data_source']}_{s['data_type']}"
    buckets.setdefault(k, [0, 0, 0])
    buckets[k][0] += hs; buckets[k][1] += hz; buckets[k][2] += 1

    if (idx + 1) % 400 == 0 or (idx + 1) == len(ss_all):
        print(f"  Full [{idx+1}/{len(ss_all)}] | Disc: {100*f_disc/(idx+1):.2f}% | SubPatch: {100*f_sub/(idx+1):.2f}% | +InnerZoom(0.85): {100*f_zoom/(idx+1):.2f}%", flush=True)

summary = {
    "gpu": gpu_name,
    "exp1_n300_subpatch": {
        "discrete_acc": round(100.0 * disc_hits / 300, 2),
        "subpatch_acc": round(100.0 * sub_hits / 300, 2),
        "median_ms": round(float(np.median(s1_lats)), 1),
    },
    "exp1_n300_cascade": cascade_table,
    "exp2_full_1272": {
        "n": len(ss_all),
        "discrete_acc": round(100.0 * f_disc / len(ss_all), 2),
        "subpatch_acc": round(100.0 * f_sub / len(ss_all), 2),
        "innerzoom_085_acc": round(100.0 * f_zoom / len(ss_all), 2),
        "innerzoom_esc_pct": round(100.0 * f_esc / len(ss_all), 1),
        "single_pass_median_ms": round(float(np.median(f_lats1)), 1),
        "innerzoom_median_ms": round(float(np.median(f_latsz)), 1),
        "breakdown": {
            k: {"subpatch": round(100.0 * v[0] / v[2], 2), "innerzoom": round(100.0 * v[1] / v[2], 2), "n": v[2]}
            for k, v in sorted(buckets.items())
        },
    },
}
with open(os.path.join(LOGS_DIR, "visjev_phase7_final_paper_results.json"), "w") as f:
    json.dump(summary, f, indent=2)
print("\nSaved visjev_phase7_final_paper_results.json successfully!", flush=True)
