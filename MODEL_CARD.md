---
license: apache-2.0
base_model: Qwen/Qwen2.5-VL-3B-Instruct
datasets:
- agentsea/wave-ui-25k
- lmms-lab/ScreenSpot-v2
tags:
- computer-use
- gui-grounding
- vision-language-model
- single-pass
- calibration
---

# VisJev-AttnPatch-3B (v6)

A **30.5 MB** LoRA (`r=24`, top-16 decoder layers) + Pretrained QK-Attention Pointer Router checkpoint for **Qwen/Qwen2.5-VL-3B-Instruct** enabling **single-forward-pass (0 decoded tokens)** DOM-free GUI click localization, calibrated spatial confidence, and `[ABSTAIN]` refusal on impossible instructions.

## Key Metrics (Zero-Shot on `lmms-lab/ScreenSpot-v2`, NVIDIA L4)
- **Full `ScreenSpot-v2` (`n=1,272`)**: `81.92%` single-pass (`331.6 ms`, `6.49×` faster) / `83.02%` with InnerZoom (`356.5 ms`, `6.03×` faster).
- **Controlled `ScreenSpot-v2` (`n=300`)**: `83.33%` single-pass / `85.67%` with pure-vision InnerZoom (`AR=0%`, `5.28×` mean speedup).
- **Target-Presence Abstention AUROC**: `0.9719` (ECE `0.0776`).

## Training Details
- **Base Model**: `Qwen/Qwen2.5-VL-3B-Instruct` (BF16)
- **Trainable Parameters**: 4.92M LoRA (`layers 20..35`, `q/k/v/o_proj`, `r=24`) + 2.69M `VisJevAttnPatchHeadV3`
- **Training Data**: 1,200 unique filtered samples from `agentsea/wave-ui-25k` (`purpose`/`description` fields) + 20% hard-negative mismatched instructions
- **Hardware**: Single NVIDIA L4 (24 GB VRAM), ~29 minutes
- **Author & Contact**: Fang Jinheng (`fangmierui@gmail.com`)
- **Code Repository**: https://github.com/Fangwenky/VisJev-AttnPatch
