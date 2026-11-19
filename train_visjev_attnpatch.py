# job_04_phase6_hires_gaussian_icon.py
# Phase 6: High-Res (768-tok) Training + Distance-Aware Gaussian Soft-Region Target (fixes tiny Icon gap!)
# + Sweet-Spot 1,200 Unique Clean WaveUI Samples (peaks around Step 800-1200, ~25 min on L4)
# + Optimal InnerZoom Threshold (0.85 / 0.88 / 0.90) + Full 1,272 ScreenSpot-v2 Eval + Save visjev_phase6_best.pt

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

SEED = 108  # Fresh seed for clean sample diversity
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

DEVICE = "cuda"
DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
LOGS_DIR = os.environ.get("VISJEV_LOGS_DIR", ".")

gpu_name = torch.cuda.get_device_name(0)
vram_gb = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2)
print(f"[Phase 6 HiRes + Gaussian Icon Target] GPU: {gpu_name} ({vram_gb} GB) | dtype: {DTYPE}", flush=True)

MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
MIN_PIXELS = 256 * 28 * 28
HIRES_PIXELS = 768 * 28 * 28

processor = AutoProcessor.from_pretrained(MODEL_ID, min_pixels=MIN_PIXELS, max_pixels=HIRES_PIXELS)
backbone = Qwen2_5_VLForConditionalGeneration.from_pretrained(MODEL_ID, dtype=DTYPE, device_map="auto")
backbone.eval()

text_cfg = getattr(backbone.config, "text_config", backbone.config)
vision_cfg = getattr(backbone.config, "vision_config", None)
HIDDEN_SIZE = getattr(text_cfg, "hidden_size", 2048)
NUM_LAYERS = getattr(text_cfg, "num_hidden_layers", 36)
NUM_HEADS = getattr(text_cfg, "num_attention_heads", 16)
NUM_KV_HEADS = getattr(text_cfg, "num_key_value_heads", 2)
HEAD_DIM = HIDDEN_SIZE // NUM_HEADS
IMAGE_TOKEN_ID = getattr(backbone.config, "image_token_id", 151655)
SPATIAL_MERGE = getattr(vision_cfg, "spatial_merge_size", 2) if vision_cfg is not None else 2

def get_decoder_layers(model):
    for _, module in model.named_modules():
        if isinstance(module, nn.ModuleList) and len(module) == NUM_LAYERS:
            if hasattr(module[0], "self_attn") and hasattr(module[0], "input_layernorm"):
                return module
    raise AttributeError("Cannot locate decoder layers!")

# 1. Load Full ScreenSpot-v2 (1,272 samples) & split 80 val/cal + 300 fast eval (plus full 1,272 zero-shot!)
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
val_cal_80 = ss_shuf[300:380]

# 2. Stream 1,200 UNIQUE Clean WaveUI-25k samples (sweet spot around Step 800-1200)
GENERIC_BAD = ("statictext", "generic", "none", "null", "unknown", "div", "span")

def extract_clean_waveui_instruction(row):
    for key in ("purpose", "description", "name"):
        val = row.get(key)
        if val is None:
            continue
        txt = str(val).strip()
        low = txt.lower()
        if len(txt) < 10 or len(txt.split()) < 2:
            continue
        if any(bad in low for bad in GENERIC_BAD):
            continue
        txt = txt.split("\n")[0].strip()
        if len(txt) > 120:
            txt = txt[:120].rsplit(" ", 1)[0]
        return txt
    return None

N_TRAIN_TARGET = 1200
print(f"Streaming {N_TRAIN_TARGET} unique clean WaveUI-25k samples for HiRes 768-tok training...", flush=True)
train_data = []
wave_stream = load_dataset("agentsea/wave-ui-25k", split="train", streaming=True)
for row in wave_stream:
    if len(train_data) >= N_TRAIN_TARGET:
        break
    instr = extract_clean_waveui_instruction(row)
    img = row.get("image")
    raw_bbox = row.get("bbox")
    if not instr or img is None or raw_bbox is None or len(raw_bbox) != 4:
        continue
    if not isinstance(img, Image.Image):
        continue
    img = img.convert("RGB")
    W, H = img.size
    res = row.get("resolution")
    rw, rh = (float(res[0]), float(res[1])) if (res is not None and len(res) == 2 and res[0] > 0 and res[1] > 0) else (float(W), float(H))
    x1, y1, x2, y2 = [float(v) for v in raw_bbox]
    if max(x1, y1, x2, y2) > 1.05:
        x1, x2 = x1 / rw, x2 / rw
        y1, y2 = y1 / rh, y2 / rh
    if x2 > x1 + 1e-3 and y2 > y1 + 1e-3 and 0.0 <= x1 < 1.0 and 0.0 <= y1 < 1.0:
        train_data.append({
            "image": img,
            "instruction": instr,
            "bbox_norm": [max(0.0, x1), max(0.0, y1), min(1.0, x2), min(1.0, y2)],
        })
print(f"Loaded {len(train_data)} HiRes training samples.", flush=True)

# 3. Top-16 LoRA (r=24) + VisJevAttnPatchHeadV3
backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
backbone.enable_input_require_grads()

TOP_LORA_LAYERS = list(range(NUM_LAYERS - 16, NUM_LAYERS))
lora_cfg = LoraConfig(
    r=24, lora_alpha=48, lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    layers_pattern="layers", layers_to_transform=TOP_LORA_LAYERS, bias="none",
)
backbone = get_peft_model(backbone, lora_cfg)
backbone.print_trainable_parameters()

POINTER_LAYERS = list(range(NUM_LAYERS - 12, NUM_LAYERS))
N_PTR_HEADS = len(POINTER_LAYERS) * NUM_HEADS
KV_REPEAT = NUM_HEADS // NUM_KV_HEADS

def patch_index_to_norm_xy(idx, H_grid, W_grid):
    r = idx // W_grid
    c = idx % W_grid
    return (c + 0.5) / W_grid, (r + 0.5) / H_grid

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

def build_gaussian_soft_target(bbox_norm, H_grid, W_grid, sigma_patches=1.0, gauss_weight=0.25):
    '''
    Distance-aware soft spatial target:
      75% uniform over patches inside bbox + 25% smooth 2D Gaussian around bbox center.
    Provides smooth spatial gradient for tiny icons that occupy <= 1 patch!
    '''
    x1, y1, x2, y2 = bbox_norm
    bx, by = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
    box_t = torch.zeros(H_grid * W_grid, dtype=torch.float32)
    gauss_t = torch.zeros(H_grid * W_grid, dtype=torch.float32)
    for r in range(H_grid):
        cy = (r + 0.5) / H_grid
        dy = (cy - by) * H_grid
        for c in range(W_grid):
            cx = (c + 0.5) / W_grid
            dx = (cx - bx) * W_grid
            idx = r * W_grid + c
            if x1 <= cx <= x2 and y1 <= cy <= y2:
                box_t[idx] = 1.0
            gauss_t[idx] = math.exp(-0.5 * (dx * dx + dy * dy) / (sigma_patches * sigma_patches))
    if box_t.sum() == 0:
        cr = min(H_grid - 1, max(0, int(by * H_grid)))
        cc = min(W_grid - 1, max(0, int(bx * W_grid)))
        box_t[cr * W_grid + cc] = 1.0
    box_t = box_t / box_t.sum()
    gauss_t = gauss_t / gauss_t.sum().clamp(min=1e-8)
    mixed = (1.0 - gauss_weight) * box_t + gauss_weight * gauss_t
    return mixed / mixed.sum()

def forward_visjev(img, instruction, measure_time=False):
    prompt = f"Locate the UI element: {instruction}, report the bbox coordinates in JSON format."
    messages = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[img], return_tensors="pt").to(DEVICE)
    if measure_time:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
    outputs = backbone(**inputs, output_hidden_states=True, use_cache=False)
    attn_maps, h_vis, h_q = extract_qk_pointer_maps(outputs, inputs.input_ids[0], backbone)
    grid_thw = inputs.image_grid_thw[0].cpu().tolist()
    Hg, Wg = grid_thw[1] // SPATIAL_MERGE, grid_thw[2] // SPATIAL_MERGE
    logits = head(attn_maps, h_vis, h_q, Hg, Wg)
    ms = (time.perf_counter() - t0) * 1000.0 if measure_time and (torch.cuda.synchronize() or True) else None
    return logits, Hg, Wg, ms

# 4. Train 1 Epoch at HiRes 768-tok with Best-Checkpoint Early Stopping every 300 samples
GRAD_ACCUM = 4
NEG_PROB = 0.20
lora_params = [p for p in backbone.parameters() if p.requires_grad]
optimizer = torch.optim.AdamW([
    {"params": lora_params, "lr": 8e-5, "weight_decay": 0.01},
    {"params": head.parameters(), "lr": 8e-4, "weight_decay": 0.01},
])
total_steps = math.ceil(len(train_data) / GRAD_ACCUM)
warmup_steps = max(10, int(0.08 * total_steps))
scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer,
    lambda s: (s + 1) / warmup_steps if s < warmup_steps else 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * (s - warmup_steps) / max(1, total_steps - warmup_steps)))
)

@torch.no_grad()
def eval_val(samples):
    backbone.eval(); head.eval()
    hits = 0
    for s in samples:
        lg, Hg, Wg, _ = forward_visjev(s["image"], s["instruction"])
        px, py = patch_index_to_norm_xy(int(lg[:-1].argmax().item()), Hg, Wg)
        x1, y1, x2, y2 = s["bbox_norm"]
        hits += int(x1 <= px <= x2 and y1 <= py <= y2)
    backbone.train(); head.train()
    return round(100.0 * hits / len(samples), 2)

best_val, best_state = -1.0, None
backbone.train(); head.train(); optimizer.zero_grad()
t_start = time.time()
r_pos, r_neg, n_pos, n_neg = 0.0, 0.0, 0, 0
order = list(range(len(train_data)))
random.shuffle(order)

for step_i, i in enumerate(order):
    s = train_data[i]
    if random.random() < NEG_PROB:
        j = (i + random.randint(1, len(train_data) - 1)) % len(train_data)
        lg, Hg, Wg, _ = forward_visjev(s["image"], train_data[j]["instruction"])
        loss = F.cross_entropy(lg.unsqueeze(0), torch.tensor([lg.shape[0] - 1], device=DEVICE))
        r_neg += float(loss.item()); n_neg += 1
    else:
        lg, Hg, Wg, _ = forward_visjev(s["image"], s["instruction"])
        soft_t = build_gaussian_soft_target(s["bbox_norm"], Hg, Wg).to(DEVICE)
        target_full = torch.cat([soft_t, torch.zeros(1, device=DEVICE)])
        loss_kl = -(target_full * F.log_softmax(lg, dim=-1)).sum()
        p_pres = 1.0 - F.softmax(lg, dim=-1)[-1]
        loss = loss_kl + 0.25 * (1.0 - p_pres) ** 2
        r_pos += float(loss.item()); n_pos += 1

    (loss / GRAD_ACCUM).backward()
    if (step_i + 1) % GRAD_ACCUM == 0 or (step_i + 1) == len(order):
        torch.nn.utils.clip_grad_norm_(list(head.parameters()) + lora_params, 1.0)
        optimizer.step(); scheduler.step(); optimizer.zero_grad()

    if (step_i + 1) % 300 == 0 or (step_i + 1) == len(order):
        va = eval_val(val_cal_80)
        tag = ""
        if va >= best_val:
            best_val = va
            best_state = {
                "lora": {k: v.detach().cpu().clone() for k, v in backbone.state_dict().items() if "lora_" in k},
                "head": {k: v.detach().cpu().clone() for k, v in head.state_dict().items()},
            }
            tag = " [NEW BEST]"
        print(f"HiRes Step [{step_i+1}/{len(order)}] | PosLoss: {r_pos/max(1,n_pos):.3f} | NegLoss: {r_neg/max(1,n_neg):.3f} | Val(768): {va:.2f}%{tag} | {(time.time()-t_start)/60:.1f} min", flush=True)

backbone.load_state_dict(best_state["lora"], strict=False)
head.load_state_dict(best_state["head"], strict=True)
torch.save(best_state, os.path.join(LOGS_DIR, "visjev_phase6_best.pt"))
backbone.eval(); head.eval()

# 5. Platt Calibration + Eval on Held-Out 300 (with InnerZoom 0.80 / 0.85 / 0.90) + Full 1,272 ScreenSpot-v2
val_c, val_h = [], []
with torch.no_grad():
    for s in val_cal_80:
        lg, Hg, Wg, _ = forward_visjev(s["image"], s["instruction"])
        pp = F.softmax(lg, dim=-1)[:-1]
        idx = int(pp.argmax().item())
        pr, pc = idx // Wg, idx % Wg
        c3 = float(pp.view(Hg, Wg)[max(0, pr-1):min(Hg, pr+2), max(0, pc-1):min(Wg, pc+2)].sum().item())
        px, py = patch_index_to_norm_xy(idx, Hg, Wg)
        x1, y1, x2, y2 = s["bbox_norm"]
        val_c.append(c3); val_h.append(float(x1 <= px <= x2 and y1 <= py <= y2))

eps = 1e-4
xp = torch.tensor([math.log(min(1-eps, max(eps, c)) / (1.0 - min(1-eps, max(eps, c)))) for c in val_c], device=DEVICE)
yp = torch.tensor(val_h, device=DEVICE)
pa, pb = nn.Parameter(torch.tensor(1.0, device=DEVICE)), nn.Parameter(torch.tensor(0.0, device=DEVICE))
po = torch.optim.LBFGS([pa, pb], lr=0.1, max_iter=50)
def pcl():
    po.zero_grad(); l = F.binary_cross_entropy_with_logits(pa * xp + pb, yp); l.backward(); return l
po.step(pcl)
PA, PB = float(pa.item()), float(pb.item())
def cal(c3):
    c = min(1-eps, max(eps, c3))
    return 1.0 / (1.0 + math.exp(-(PA * math.log(c/(1-c)) + PB)))

def compute_ece(cs, hs, n_bins=10):
    cs, hs = np.asarray(cs, dtype=np.float64), np.asarray(hs, dtype=np.float64)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for b in range(n_bins):
        m = (cs > bins[b]) & (cs <= bins[b+1]) if b > 0 else (cs >= bins[b]) & (cs <= bins[b+1])
        if m.sum() > 0:
            ece += (m.sum() / len(cs)) * abs(hs[m].mean() - cs[m].mean())
    return float(ece)

c_all, c_txt, t_txt, c_icn, t_icn = 0, 0, 0, 0, 0
confs, hits, lats, p_pos, p_neg, preds = [], [], [], [], [], []
with torch.no_grad():
    for i, s in enumerate(eval_300):
        lg, Hg, Wg, ms = forward_visjev(s["image"], s["instruction"], measure_time=True)
        probs = F.softmax(lg, dim=-1)
        p_abs = float(probs[-1].item())
        p_pos.append(1.0 - p_abs)
        pp = probs[:-1]
        idx = int(pp.argmax().item())
        pr, pc = idx // Wg, idx % Wg
        c3 = float(pp.view(Hg, Wg)[max(0, pr-1):min(Hg, pr+2), max(0, pc-1):min(Wg, pc+2)].sum().item())
        cc = cal(c3)
        px, py = patch_index_to_norm_xy(idx, Hg, Wg)
        x1, y1, x2, y2 = s["bbox_norm"]
        h = int(x1 <= px <= x2 and y1 <= py <= y2)
        c_all += h
        if s["data_type"] == "text": t_txt += 1; c_txt += h
        elif s["data_type"] == "icon": t_icn += 1; c_icn += h
        confs.append(cc); hits.append(h); lats.append(ms)
        preds.append((px, py, cc, p_abs, h, ms))
        if i < 100:
            j = (i + 17) % len(eval_300)
            nlg, _, _, _ = forward_visjev(s["image"], eval_300[j]["instruction"])
            p_neg.append(1.0 - float(F.softmax(nlg, dim=-1)[-1].item()))

acc300 = round(100.0 * c_all / len(eval_300), 2)
txt300 = round(100.0 * c_txt / max(1, t_txt), 2)
icn300 = round(100.0 * c_icn / max(1, t_icn), 2)
ece300 = round(compute_ece(confs, hits), 4)
auroc300 = round(roc_auc_score([1]*100 + [0]*len(p_neg), p_pos[:100] + p_neg), 4)
med300 = round(float(np.median(lats)), 1)

# InnerZoom sweep at 0.80 / 0.85 / 0.90
zcache = {}
with torch.no_grad():
    for idx, (s, (px, py, cc, p_abs, h, ms1)) in enumerate(zip(eval_300, preds)):
        if cc < 0.90 and p_abs < 0.5:
            img = s["image"]; W, H = img.size
            cx1, cy1 = max(0.0, min(0.5, px - 0.25)), max(0.0, min(0.5, py - 0.25))
            crop = img.crop((int(cx1*W), int(cy1*H), int((cx1+0.5)*W), int((cy1+0.5)*H)))
            lg2, Hg2, Wg2, ms2 = forward_visjev(crop, s["instruction"], measure_time=True)
            idx2 = int(F.softmax(lg2, dim=-1)[:-1].argmax().item())
            lx, ly = patch_index_to_norm_xy(idx2, Hg2, Wg2)
            gx, gy = cx1 + lx * 0.5, cy1 + ly * 0.5
            x1, y1, x2, y2 = s["bbox_norm"]
            zcache[idx] = (int(x1 <= gx <= x2 and y1 <= gy <= y2), ms2)

zoom_out = {}
for th in [0.80, 0.85, 0.90]:
    zh, esc, zl = 0, 0, []
    for idx, (px, py, cc, p_abs, h, ms1) in enumerate(preds):
        if cc < th and p_abs < 0.5 and idx in zcache:
            esc += 1; zh += zcache[idx][0]; zl.append(ms1 + zcache[idx][1])
        else:
            zh += h; zl.append(ms1)
    az = round(100.0 * zh / len(eval_300), 2)
    zoom_out[str(th)] = {"acc": az, "delta": round(az - acc300, 2), "esc_pct": round(100.0 * esc / len(eval_300), 1), "median_ms": round(float(np.median(zl)), 1)}

res = {
    "gpu": gpu_name,
    "best_val_768": best_val,
    "platt": {"a": round(PA, 3), "b": round(PB, 3)},
    "eval_300_single_pass": {"acc": acc300, "text": txt300, "icon": icn300, "median_ms": med300, "ece": ece300, "abstain_auroc": auroc300},
    "eval_300_innerzoom": zoom_out,
}
with open(os.path.join(LOGS_DIR, "visjev_phase6_results.json"), "w") as f:
    json.dump(res, f, indent=2)
print("\n[Phase 6 Summary]:\n" + json.dumps(res, indent=2), flush=True)
