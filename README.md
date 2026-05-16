# Homework 2 — Image Captioning

**Course:** Deep Network Development 25/26/2 · Eötvös Loránd University  
**Dataset:** MS COCO Captions 2017  

---

## Overview

This repository contains a complete implementation of an **Encoder–Attention–Transformer Decoder** image captioning system trained on MS COCO 2017, plus a comparison against the pre-trained **BLIP** model.

### Architecture

```
Image
  │
  ▼
CNNEncoder (ResNet-50, fine-tuned from layer3)
  │  outputs (B, 196, 2048) spatial patch features
  ▼
TransformerDecoder
  ├── Word Embedding + Positional Encoding
  ├── Control Embedding  (length bucket: short / medium / long / very long)
  └── N × TransformerDecoderLayer
        ├── Masked Multi-Head Self-Attention
        ├── Cross-Attention  →  attention weights saved for heatmap viz
        └── Feed-Forward Network
  │
  ▼
Vocabulary logits  →  greedy / beam-search caption
```

### Key features

| Feature | Details |
|---|---|
| Encoder | ResNet-50 (ImageNet pre-trained, fine-tuned from `layer3`) |
| Decoder | 3-layer Transformer, `d_model=512`, 8 heads |
| Controllable generation | Length-bucket embedding (4 buckets) injected at every decoding step |
| Cross-attention viz | Per-word heatmaps overlaid on the original image |
| Decoding | Greedy (default) + beam search |
| Overfitting prevention | Early stopping, `ReduceLROnPlateau`, label smoothing, gradient clipping |
| Comparison model | BLIP (`Salesforce/blip-image-captioning-base`) |

---

## Repository structure

```
.
├── Homework_2_Image_Captioning_solved.ipynb   # Main notebook (all code)
├── Dockerfile                                  # Reproducible GPU environment
├── requirements.txt                            # Python dependencies
└── README.md
```

---

## Quick start

### Option A — Docker (recommended, GPU)

```bash
# 1. Build the image
docker build -t image-captioning .

# 2. Run Jupyter Lab (mount a local data folder for COCO)
docker run --gpus all -p 8888:8888 \
    -v $(pwd)/coco:/workspace/coco \
    image-captioning
```

Then open `http://localhost:8888` and run `Homework_2_Image_Captioning_solved.ipynb`.

> **CPU-only?** Remove `--gpus all` from the run command. Training will be much slower — consider reducing `TRAIN_SUBSET` to ≤5 000.

### Option B — Local virtual environment

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
jupyter lab Homework_2_Image_Captioning_solved.ipynb
```

---

## Dataset setup

Download and extract MS COCO 2017 so the directory looks like this:

```
coco/
├── images/
│   ├── train2017/      # ~118 K images
│   └── val2017/        #   ~5 K images
└── annotations/
    ├── captions_train2017.json
    └── captions_val2017.json
```

Download links:

```bash
wget http://images.cocodataset.org/zips/train2017.zip            # ~18 GB
wget http://images.cocodataset.org/zips/val2017.zip              # ~1 GB
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip

unzip train2017.zip -d coco/images/
unzip val2017.zip   -d coco/images/
unzip annotations_trainval2017.zip -d coco/
```

> ⚠️ **Hardware note:** The notebook uses `TRAIN_SUBSET = 20000` by default. Set it to `None` to train on the full dataset. A single T4 GPU trains 10 epochs on 20 k images in roughly 3–4 hours.

---

## Notebook walkthrough

| Section | What happens |
|---|---|
| 0 | Imports, device setup, reproducibility seed |
| 1 | Data paths, optional download commands |
| 2 | Train / val image transforms (resize, crop, flip, colour jitter, normalise) |
| 3 | `Vocabulary` class (freq ≥ 5), `COCOCaptionsDataset`, padding `collate_fn` |
| 4 | Visualise 8 training samples with original + tokenised captions |
| 5 | Full model definition: `CNNEncoder`, `TransformerDecoderLayer`, `TransformerDecoder`, `ImageCaptioningModel` |
| 6 | `CrossEntropyLoss` (label smoothing), `AdamW` (two LR groups), `ReduceLROnPlateau` |
| 7 | Training loop — teacher forcing, gradient clipping, early stopping, checkpoint saving |
| 8.1 | Load best checkpoint, plot train/val loss curves |
| 8.2 | Cross-attention heatmaps for 5 validation images |
| 8.3 | BLEU-1/2/3/4 on 1 000 val samples; 10 sample captions; controllable generation demo |
| 9 | Load BLIP (`Salesforce/blip-image-captioning-base`) |
| 10 | Evaluate BLIP with identical BLEU pipeline |
| 11 | Comparison table + bar chart + side-by-side captions + written analysis |

---

## Results (example — 20 k subset, 10 epochs)

| Metric | Custom model | BLIP |
|---|---|---|
| BLEU-1 | ~0.55 | ~0.72 |
| BLEU-2 | ~0.35 | ~0.53 |
| BLEU-3 | ~0.22 | ~0.38 |
| BLEU-4 | ~0.13 | ~0.27 |

*Exact numbers depend on hardware, subset size, and random seed.*

---

## Requirements

- Python 3.10+
- PyTorch 2.2+ (CUDA 11.8 recommended for GPU training)
- See `requirements.txt` for the full list
