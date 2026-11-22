# VisJev-AttnPatch: Single-Pass DOM-Free GUI Grounding & Calibrated Abstention

**VisJev-AttnPatch** replaces slow autoregressive JSON coordinate decoding (`~2,158.5 ms/click` on 3B; `~2,991.0 ms/click` on 7B) in Computer Use Vision-Language Models with a **single-forward-pass (0 decoded tokens)** calibrated visual-patch decision architecture built on both [Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct) (**30.5 MB** checkpoint) and [Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) (**34.5 MB** checkpoint).

## Key Results (Zero-Shot on [ScreenSpot-v2](https://huggingface.co/datasets/lmms-lab/ScreenSpot-v2) & [ScreenSpot-Pro](https://huggingface.co/datasets/HongxinLi/ScreenSpot-Pro), NVIDIA L4)

- **Cross-Scale Single-Pass Grounding (`n=1,272`)**: **81.92%** (`95% CI [79.80, 84.04]`, `333.3 ms` mean, **6.48x mean speedup**) on **3B** and **83.73%** (`90.95%` Text / `74.37%` Icon, `463.2 ms` mean, **6.46x mean speedup**) on **7B** via O(N_vis) pretrained QK-pointer routing + 0 ms 3x3 sub-patch soft-argmax (exact paired McNemar `p = 3.11e-10` vs. 1x1 discrete argmax).
- **Confidence-Verified InnerZoom (CV-Zoom, `n=1,272`, 0% AR tokens)**: **83.25%** (`5.48x` mean speedup, `theta=0.82`), **84.36%** (`95% CI [82.31, 86.32]`, `4.52x` mean speedup, `theta=0.85`, McNemar `p = 3.12e-6` vs. SP), and **84.91%** (`3.68x`, `theta=0.90`) on **3B**; **84.67%** (`91.36%` Text / **75.99%** Icon, **5.17x mean speedup**, `theta=0.85`) on **7B**---completely eliminating mobile text negative transfer (`3B iOS Text: 99.28% -> 100.00%`, `7B Android Icon: 90.11%`).
- **32.0% (3B) / 38.0% (7B) on 4K ScreenSpot-Pro (`n=200`, Zero Retraining)** via **2x2 Tiled QK-Pointer Routing + 4x Local Zoom** (0 decoded tokens), leaping from `5.0% / 2.0%` single-pass and surpassing official `Qwen2.5-VL-3B` AR (`25.9%`) and `GUI-Actor-7B-LiteTrain + Verifier` (`35.8%`).
- **0.9535 (3B) / 0.9576 (7B) Hard Same-Platform & Same-Type Abstention AUROC** (**0.9720 / 0.9754** Cross-Platform AUROC) and **0.0776 (3B) / 0.0313 (7B) ECE** calibrated spatial confidence.
- Trained on only **1,200 filtered web samples** from [WaveUI-25k](https://huggingface.co/datasets/Voxel51/WaveUI-25k) (`29 min` for 3B, `36.6 min` for 7B on a single L4 GPU).

| Method / Scale | Overall Acc (95% CI) | Text / Icon (Weighted) | Med. / Mean Lat. (L4) | Mean Speedup | ECE / Hard AUROC |
|---|---:|---:|---:|---:|---:|
| Official Qwen2.5-VL-3B AR (Matched `n=300`) | 82.33% | 89.70% / 73.33% | 2,139.1 / 2,158.5 ms | 1.00x (3B) | - |
| Official Qwen2.5-VL-7B AR (Matched `n=300`) | 79.33% | 85.45% / 71.85% | 2,988.3 / 2,991.0 ms | 1.00x (7B) | - |
| **VisJev-3B + 0ms Sub-Patch (`n=300`)** | **83.33%** | - | **327.7 ms (med)** | **6.59x** | **0.0776 / 0.9535** |
| **VisJev-3B + InnerZoom (`theta=0.82`, `n=300`)** | **85.67%** | - | **335.9 / 407.4 ms** | **5.30x** | **0.0776 / 0.9535** |
| **VisJev-7B + 0ms Sub-Patch (`n=300`)** | **86.00%** | - | **457.8 / 463.2 ms** | **6.46x** | **0.0313 / 0.9576** |
| **Full ScreenSpot-v2 (`n=1,272`) VisJev-3B SP** | **81.92% [79.8, 84.0]** | **90.53% / 70.76%** | **331.4 / 333.3 ms** | **6.48x** | - |
| **Full ScreenSpot-v2 (`n=1,272`) VisJev-3B CV-Zoom (`0.82`)** | **83.25% [81.2, 85.2]** | **92.06% / 71.84%** | **335.1 / 393.9 ms** | **5.48x** | - |
| **Full ScreenSpot-v2 (`n=1,272`) VisJev-3B CV-Zoom (`0.85`)** | **84.36% [82.3, 86.3]** | **92.76% / 73.47%** | **356.7 / 477.4 ms** | **4.52x** | - |
| **Full ScreenSpot-v2 (`n=1,272`) VisJev-3B CV-Zoom (`0.90`)** | **84.91% [82.9, 86.9]** | - | **648.3 / 587.3 ms** | **3.68x** | - |
| **Full ScreenSpot-v2 (`n=1,272`) VisJev-7B SP** | **83.73%** | **90.95% / 74.37%** | **457.8 / 463.2 ms** | **6.46x** | - |
| **Full ScreenSpot-v2 (`n=1,272`) VisJev-7B CV-Zoom (`0.82`)** | **84.12%** | **91.23% / 74.91%** | **460.3 / 515.3 ms** | **5.80x** | - |
| **Full ScreenSpot-v2 (`n=1,272`) VisJev-7B CV-Zoom (`0.85`)** | **84.67%** | **91.36% / 75.99%** | **484.2 / 578.1 ms** | **5.17x** | - |
| **4K ScreenSpot-Pro (`n=200`) VisJev-3B 2x2 Tiled + 4x Zoom** | **32.00%** | - | **1,560.8 ms (mean)** | **1.38x** | - |
| **4K ScreenSpot-Pro (`n=200`) VisJev-7B 2x2 Tiled + 4x Zoom** | **38.00%** | - | **2,142.8 ms (mean)** | **1.40x** | - |

## Paper

- **PDF**: [VisJev_AttnPatch_Paper.pdf](paper/VisJev_AttnPatch_Paper.pdf) (Camera-ready two-column conference paper with vector TikZ figures, aligned equations, SOTA comparison, CV-Zoom, 4K Tiled Routing, and full ScreenSpot-v2 / ScreenSpot-Pro results)
- **LaTeX Source**: [VisJev_AttnPatch_Paper.tex](paper/VisJev_AttnPatch_Paper.tex)

## Repository Structure

```text
.
├── paper/
│   ├── VisJev_AttnPatch_Paper.pdf  # Camera-ready conference paper (vector figures & booktabs tables)
│   └── VisJev_AttnPatch_Paper.tex  # Reproducible LaTeX + TikZ source
├── checkpoints/
│   └── visjev_phase6_best.pt       # 30.5 MB LoRA (r=24, top-16 layers) + QK-Pointer Router head
├── inference.py                    # Standalone CLI: Single-Pass + Sub-Patch + CV-Zoom + 2x2 Tiled 4K Routing
├── train_visjev_attnpatch.py       # Reproducible single-GPU training script (WaveUI-25k -> ScreenSpot-v2)
├── eval_screenspot_v2.py           # Full 1,272-sample ScreenSpot-v2 & 3-tier cascade evaluation script
├── MODEL_CARD.md                   # Hugging Face model card
├── requirements.txt
└── LICENSE                         # Apache-2.0
```

## Quickstart

```bash
pip install -r requirements.txt

# Standard screenshot (auto Confidence-Verified InnerZoom)
python inference.py \
  --checkpoint checkpoints/visjev_phase6_best.pt \
  --image screenshot.png \
  --instruction "Click the search icon" \
  --zoom_thresh 0.85

# 4K / Ultra-high-resolution screenshot (2x2 Overlapping Tiled Routing + 4x Local Zoom)
python inference.py \
  --checkpoint checkpoints/visjev_phase6_best.pt \
  --image screenshot_4k.png \
  --instruction "Toggle grid snap" \
  --tiled_4k
```

## Citation

```bibtex
@misc{fang2026visjevattnpatch,
  title={VisJev-AttnPatch: Single-Pass DOM-Free GUI Grounding via Pretrained QK-Attention Pointer Routing, Calibrated Abstention, and Sub-Patch Refinement},
  author={Fang Jinheng},
  year={2026},
  howpublished={\url{https://github.com/Fangwenky/VisJev-AttnPatch}}
}
```

## Contact & License

- **Author**: Fang Jinheng (`fangmierui@gmail.com`)
- **Repository**: https://github.com/Fangwenky/VisJev-AttnPatch
- **License**: Apache License 2.0
