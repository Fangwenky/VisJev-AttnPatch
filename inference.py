"""
VisJev-AttnPatch Standalone Inference Module
Single-Pass (0 Decoded Tokens) DOM-Free GUI Grounding + Calibrated Abstention + Confidence-Gated InnerZoom.
"""
import argparse
import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

import peft.import_utils as _peft_iu
import peft.tuners.lora.torchao as _peft_tao
_peft_iu.is_torchao_available = lambda: False
_peft_tao.is_torchao_available = lambda: False
from peft import LoraConfig, get_peft_model


class VisJevAttnPatchHeadV3(nn.Module):
    def __init__(self, hidden_size: int, n_ptr_heads: int, proj_dim: int = 256):
        super().__init__()
        self.head_router = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, 256),
            nn.GELU(),
            nn.Linear(256, n_ptr_heads),
        )
        self.head_prior = nn.Parameter(torch.zeros(n_ptr_heads))
        self.attn_temp = nn.Parameter(torch.tensor(1.5))
        self.q_res = nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, proj_dim))
        self.v_res = nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, proj_dim))
        self.pos_mlp = nn.Sequential(nn.Linear(2, 32), nn.GELU(), nn.Linear(32, proj_dim))
        self.res_scale = nn.Parameter(torch.tensor(0.5))
        self.abstain_head = nn.Sequential(
            nn.Linear(hidden_size * 2 + 2, 256),
            nn.GELU(),
            nn.Linear(256, 1),
        )

    def forward(self, attn_maps, h_vis, h_q, H_grid: int, W_grid: int):
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


class VisJevGrounder:
    def __init__(
        self,
        checkpoint_path: str,
        model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct",
        lora_r: int = 24,
        max_pixels: int = 768 * 28 * 28,
        platt_a: float = 0.352,
        platt_b: float = 1.991,
        device: str = "cuda",
    ):
        self.device = device
        self.platt_a = platt_a
        self.platt_b = platt_b
        dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

        self.processor = AutoProcessor.from_pretrained(model_id, min_pixels=256 * 28 * 28, max_pixels=max_pixels)
        backbone = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, dtype=dtype, device_map="auto")

        text_cfg = getattr(backbone.config, "text_config", backbone.config)
        vision_cfg = getattr(backbone.config, "vision_config", None)
        self.hidden_size = getattr(text_cfg, "hidden_size", 2048)
        self.num_layers = getattr(text_cfg, "num_hidden_layers", 36)
        self.num_heads = getattr(text_cfg, "num_attention_heads", 16)
        self.num_kv_heads = getattr(text_cfg, "num_key_value_heads", 2)
        self.head_dim = self.hidden_size // self.num_heads
        self.image_token_id = getattr(backbone.config, "image_token_id", 151655)
        self.spatial_merge = getattr(vision_cfg, "spatial_merge_size", 2) if vision_cfg is not None else 2

        lora_cfg = LoraConfig(
            r=lora_r,
            lora_alpha=lora_r * 2,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            layers_pattern="layers",
            layers_to_transform=list(range(self.num_layers - 16, self.num_layers)),
            bias="none",
        )
        self.backbone = get_peft_model(backbone, lora_cfg)
        self.pointer_layers = list(range(self.num_layers - 12, self.num_layers))
        self.kv_repeat = self.num_heads // self.num_kv_heads
        self.head = VisJevAttnPatchHeadV3(self.hidden_size, len(self.pointer_layers) * self.num_heads).to(device)

        ckpt = torch.load(checkpoint_path, map_location="cpu")
        self.backbone.load_state_dict(ckpt["lora"], strict=False)
        self.head.load_state_dict(ckpt["head"], strict=True)
        self.backbone.eval()
        self.head.eval()

    def _get_decoder_layers(self):
        for _, module in self.backbone.named_modules():
            if isinstance(module, nn.ModuleList) and len(module) == self.num_layers:
                if hasattr(module[0], "self_attn") and hasattr(module[0], "input_layernorm"):
                    return module
        raise AttributeError("Cannot locate decoder layers in backbone.")

    def _calibrate(self, c3: float) -> float:
        eps = 1e-4
        c = min(1.0 - eps, max(eps, c3))
        return 1.0 / (1.0 + math.exp(-(self.platt_a * math.log(c / (1.0 - c)) + self.platt_b)))

    @staticmethod
    def _subpatch_soft_argmax(patch_probs: torch.Tensor, H_grid: int, W_grid: int):
        pred_idx = int(patch_probs.argmax().item())
        pr, pc = pred_idx // W_grid, pred_idx % W_grid
        grid_p = patch_probs.view(H_grid, W_grid)
        r0, r1 = max(0, pr - 1), min(H_grid, pr + 2)
        c0, c1 = max(0, pc - 1), min(W_grid, pc + 2)
        sub = grid_p[r0:r1, c0:c1]
        w_sum = float(sub.sum().item())
        rs = torch.arange(r0, r1, device=patch_probs.device, dtype=torch.float32)
        cs = torch.arange(c0, c1, device=patch_probs.device, dtype=torch.float32)
        gy, gx = torch.meshgrid(rs, cs, indexing="ij")
        cx = float(((gx + 0.5) * sub).sum().item() / max(w_sum, 1e-8)) / W_grid
        cy = float(((gy + 0.5) * sub).sum().item() / max(w_sum, 1e-8)) / H_grid
        return (cx, cy), w_sum

    @torch.no_grad()
    def _single_pass(self, img: Image.Image, instruction: str):
        prompt = f"Locate the UI element: {instruction}, report the bbox coordinates in JSON format."
        messages = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": prompt}]}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[img], return_tensors="pt").to(self.device)

        outputs = self.backbone(**inputs, output_hidden_states=True, use_cache=False)
        layers = self._get_decoder_layers()
        input_ids = inputs.input_ids[0]
        img_mask = input_ids == self.image_token_id
        after_img = int(torch.where(img_mask)[0][-1].item()) + 1

        scale = 1.0 / math.sqrt(self.head_dim)
        scores = []
        for l_idx in self.pointer_layers:
            layer = layers[l_idx]
            h_in = outputs.hidden_states[l_idx][0]
            q = layer.self_attn.q_proj(layer.input_layernorm(h_in[-1:])).view(self.num_heads, self.head_dim).float()
            k = layer.self_attn.k_proj(layer.input_layernorm(h_in[img_mask])).view(-1, self.num_kv_heads, self.head_dim).float()
            k = k.repeat_interleave(self.kv_repeat, dim=1)
            scores.append(torch.einsum("hd,nhd->hn", q, k) * scale)

        attn_maps = torch.cat(scores, dim=0)
        h_last = outputs.hidden_states[-1][0]
        h_vis = h_last[img_mask].float()
        h_q = (0.5 * h_last[-1] + 0.5 * h_last[after_img:].mean(dim=0)).float()

        grid_thw = inputs.image_grid_thw[0].cpu().tolist()
        Hg, Wg = grid_thw[1] // self.spatial_merge, grid_thw[2] // self.spatial_merge
        logits = self.head(attn_maps, h_vis, h_q, Hg, Wg)
        return logits, Hg, Wg

    @torch.no_grad()
    def predict(self, img: Image.Image, instruction: str, zoom_thresh: float = 0.82, abstain_thresh: float = 0.50):
        img = img.convert("RGB")
        t0 = time.perf_counter()
        logits, Hg, Wg = self._single_pass(img, instruction)
        probs = F.softmax(logits, dim=-1)
        p_abstain = float(probs[-1].item())
        (x, y), c3 = self._subpatch_soft_argmax(probs[:-1], Hg, Wg)
        conf = self._calibrate(c3)
        zoomed = False

        if p_abstain < abstain_thresh and conf < zoom_thresh:
            zoomed = True
            W, H = img.size
            cx1, cy1 = max(0.0, min(0.5, x - 0.25)), max(0.0, min(0.5, y - 0.25))
            crop = img.crop((int(cx1 * W), int(cy1 * H), int((cx1 + 0.5) * W), int((cy1 + 0.5) * H)))
            lg2, Hg2, Wg2 = self._single_pass(crop, instruction)
            (lx, ly), _ = self._subpatch_soft_argmax(F.softmax(lg2, dim=-1)[:-1], Hg2, Wg2)
            x, y = cx1 + lx * 0.5, cy1 + ly * 0.5

        elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 1)
        return {
            "x_norm": round(x, 4),
            "y_norm": round(y, 4),
            "x_px": int(round(x * img.width)),
            "y_px": int(round(y * img.height)),
            "confidence": round(conf, 4),
            "abstain_prob": round(p_abstain, 4),
            "abstain": p_abstain >= abstain_thresh,
            "innerzoom_triggered": zoomed,
            "latency_ms": elapsed_ms,
        }


def main():
    parser = argparse.ArgumentParser(description="VisJev-AttnPatch Single-Pass GUI Grounding Inference")
    parser.add_argument("--checkpoint", required=True, help="Path to visjev_phase6_best.pt")
    parser.add_argument("--image", required=True, help="Path to GUI screenshot")
    parser.add_argument("--instruction", required=True, help="UI element instruction")
    parser.add_argument("--zoom_thresh", type=float, default=0.82, help="Confidence threshold for 2x InnerZoom")
    args = parser.parse_args()

    model = VisJevGrounder(args.checkpoint)
    img = Image.open(args.image)
    result = model.predict(img, args.instruction, zoom_thresh=args.zoom_thresh)
    print(result)


if __name__ == "__main__":
    main()
