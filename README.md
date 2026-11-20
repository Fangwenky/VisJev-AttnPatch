# VisJev-AttnPatch: Single-Pass DOM-Free GUI Grounding & Calibrated Abstention

**VisJev-AttnPatch** replaces slow autoregressive JSON coordinate decoding (`~2,158.5 ms/click` mean) in Computer Use Vision-Language Models with a **single-forward-pass (0 decoded tokens)** calibrated visual-patch decision architecture (**30.5 MB** adapter + pointer head checkpoint) built on [Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct).

## Key Results (Zero-Shot on [ScreenSpot-v2](https://huggingface.co/datasets/lmms-lab/ScreenSpot-v2) & [ScreenSpot-Pro](https://huggingface.co/datasets/HongxinLi/ScreenSpot-Pro), NVIDIA L4)

- **81.92% Full ScreenSpot-v2 (`n=1,272`) Single-Pass** at **333.3 ms mean / 331.4 ms median** latency (**6.48x mean speedup** over matched `Qwen2.5-VL-3B-Instruct` AR decoding at `2,158.5 ms`) via O(N_vis) pretrained QK-pointer routing + 0 ms 3x3 sub-patch soft-argmax.
- **83.25% (`5.48x` mean speedup, `theta=0.82`) / 84.36% (`4.52x` mean speedup, `theta=0.85`) Full ScreenSpot-v2 (`n=1,272`)** with **Confidence-Verified InnerZoom (CV-Zoom)**---completely eliminating mobile text negative transfer (`iOS Text: 99.28% -> 100.00%`, `Android Text: 95.83% -> 97.50%`) and outperforming matched `Qwen2.5-VL-3B` AR (`82.33%`) with **0% autoregressive decoding**.
- **31.5% on 4K ScreenSpot-Pro (`n=200`, Zero Retraining)** via **2x2 Overlapping Tiled QK-Pointer Routing + 4x Local Zoom** (`1,560.8 ms` mean, 0 decoded tokens), leaping from `5.0%` single-pass and surpassing official `Qwen2.5-VL-3B` AR (`25.9%`).
- **0.9535 Hard Same-Platform / 0.9720 Cross-Platform AUROC** impossible-instruction refusal (`[ABSTAIN]` gate) and **0.0776 ECE** calibrated spatial confidence.
- Trained on only **1,200 filtered web samples** from [WaveUI-25k](https://huggingface.co/datasets/Voxel51/WaveUI-25k) (~29 min on a single L4 GPU).

| Method | Overall Acc | Text / Icon (Weighted) | Med. / Mean Lat. (L4) | Mean Speedup | Abstain AUROC |
|---|---:|---:|---:|---:|---:|
| Official Qwen2.5-VL-3B AR (Matched `n=300`) | 82.33% | 89.70% / 73.33% | 2,139.1 / 2,158.5 ms | 1.00x | - |
| VisJev-AttnPatch v6 Discrete (`n=300`) | 77.33% | 86.06% / 66.67% | 332.8 ms (med) | 6.49x | 0.9719 |
| **VisJev-v6 + 0ms Sub-Patch (`n=300`)** | **83.33%** | - | **327.7 ms (med)** | **6.59x** | **0.9719** |
| **VisJev-v6 + InnerZoom (`theta=0.82`, `n=300`)** | **85.67%** | - | **335.9 / 407.4 ms** | **5.30x** | **0.9720 / 0.9535 (Hard)** |
| **Full ScreenSpot-v2 (`n=1,272`) Single-Pass (SP)** | **81.92%** | **90.53% / 70.76%** | **331.4 / 333.3 ms** | **6.48x** | - |
| Full ScreenSpot-v2 (`n=1,272`) Naive Zoom (`0.85`) | 83.02% | 90.81% / 72.92% | 356.7 / 477.4 ms | 4.52x | - |
| **Full ScreenSpot-v2 (`n=1,272`) CV-Zoom (`0.82`)** | **83.25%** | **92.06% / 71.84%** | **335.1 / 393.9 ms** | **5.48x** | - |
| **Full ScreenSpot-v2 (`n=1,272`) CV-Zoom (`0.85`)** | **84.36%** | **92.76% / 73.47%** | **356.7 / 477.4 ms** | **4.52x** | - |
| **4K ScreenSpot-Pro (`n=200`) 2x2 Tiled + 4x Zoom** | **31.50%** | - | **1,560.8 ms (mean)** | **1.38x** | - |

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
