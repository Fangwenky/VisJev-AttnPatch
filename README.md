# VisJev-AttnPatch: Single-Pass DOM-Free GUI Grounding & Calibrated Abstention

**VisJev-AttnPatch** replaces slow autoregressive JSON coordinate decoding (`~2,151 ms/click`) in Computer Use Vision-Language Models with a **single-forward-pass (0 decoded tokens)** calibrated visual-patch decision architecture (**30.5 MB** adapter + pointer head checkpoint) built on [Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct).

## Key Results (Zero-Shot on [ScreenSpot-v2](https://huggingface.co/datasets/lmms-lab/ScreenSpot-v2), NVIDIA L4)

- **81.92% Full ScreenSpot-v2 (`n=1,272`) Single-Pass** at **331.6 ms** median latency (**6.49x faster** than official `Qwen2.5-VL-3B-Instruct` autoregressive decoding) via O(N_vis) pretrained QK-pointer routing + 0 ms 3x3 sub-patch soft-argmax.
- **83.02% Full (`n=1,272`) / 85.67% Controlled (`n=300`)** with pure-vision **Confidence-Gated InnerZoom** (**0% autoregressive tokens**, **5.28x-6.03x speedup**).
- **0.9719-0.9778 AUROC** impossible-instruction refusal (`[ABSTAIN]` gate) and **0.0776 ECE** calibrated spatial confidence.
- Trained on only **1,200 filtered web samples** from [WaveUI-25k](https://huggingface.co/datasets/Voxel51/WaveUI-25k) (~29 min on a single L4 GPU).

| Method | Overall Acc | Text / Icon | Med. Latency (L4) | Speedup | ECE | Abstain AUROC |
|---|---:|---:|---:|---:|---:|---:|
| Qwen2.5-VL-3B Autoregressive (`n=100`) | 87.00% | 94.64% / 77.27% | 2,151.2 ms | 1.00x | - | - |
| VisJev-AttnPatch v6 Discrete (`n=300`) | 77.33% | 86.06% / 66.67% | 332.8 ms | 6.46x | 0.0776 | 0.9719 |
| **VisJev-AttnPatch v6 + 0ms Sub-Patch (`n=300`)** | **83.33%** | - | **327.7 ms** | **6.56x** | **0.0776** | **0.9719** |
| **VisJev-AttnPatch v6 + InnerZoom (`z<0.82`, `n=300`)** | **85.67%** | - | **335.9 ms** | **5.28x (mean)** | **0.0776** | **0.9719** |
| **Full ScreenSpot-v2 (`n=1,272`) Single-Pass** | **81.92%** | **89.26% / 70.64%** | **331.6 ms** | **6.49x** | - | - |
| **Full ScreenSpot-v2 (`n=1,272`) + InnerZoom (`0.85`)** | **83.02%** | **90.02% / 72.18%** | **356.5 ms** | **6.03x** | - | - |

## Repository Structure

```text
.
├── checkpoints/
│   └── visjev_phase6_best.pt       # 30.5 MB LoRA (r=24, top-16 layers) + QK-Pointer Router head
├── inference.py                    # Standalone single-pass + Sub-Patch + InnerZoom inference CLI
├── train_visjev_attnpatch.py       # Reproducible single-GPU training script (WaveUI-25k -> ScreenSpot-v2)
├── eval_screenspot_v2.py           # Full 1,272-sample ScreenSpot-v2 & 3-tier cascade evaluation script
├── MODEL_CARD.md                   # Hugging Face model card
├── requirements.txt
└── LICENSE                         # Apache-2.0
```

## Quickstart

```bash
pip install -r requirements.txt
python inference.py \
  --checkpoint checkpoints/visjev_phase6_best.pt \
  --image screenshot.png \
  --instruction "Click the search icon" \
  --zoom_thresh 0.82
```

## License

Apache License 2.0
