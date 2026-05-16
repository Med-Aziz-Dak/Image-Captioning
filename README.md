# 🚀 Homework 2 — Image Captioning with Transformers & Attention

### 🧠 Deep Network Development 25/26/2 · Eötvös Loránd University

<p align="center">
  <img src="https://miro.medium.com/v2/resize:fit:1400/1*3fA77_mLNiJTSgZFhYnU0Q.png" width="850"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python"/>
  <img src="https://img.shields.io/badge/PyTorch-2.2+-red?style=for-the-badge&logo=pytorch"/>
  <img src="https://img.shields.io/badge/Transformer-Attention-orange?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Dataset-MS--COCO-green?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/GPU-CUDA%2011.8-success?style=for-the-badge&logo=nvidia"/>
</p>

---

# ✨ Project Overview

This project implements a complete **Image Captioning System** using:

* 🖼️ **CNN Encoder (ResNet-50)**
* 🎯 **Transformer Decoder with Cross-Attention**
* 🧠 **Controllable Caption Generation**
* 🔥 **Attention Heatmap Visualisation**
* 🚀 **Greedy + Beam Search Decoding**
* 🤖 Comparison against **BLIP** (`Salesforce/blip-image-captioning-base`)

The model is trained on the **MS COCO 2017 Captions Dataset** and demonstrates how modern attention-based architectures can generate human-like image descriptions.

---

# 🏗️ Architecture

<p align="center">
  <img src="https://jalammar.github.io/images/t/transformer_resideual_layer_norm_3.png" width="850"/>
</p>

```text
Image
  │
  ▼
CNNEncoder (ResNet-50)
  │  → spatial feature maps
  ▼
Transformer Decoder
  ├── Word Embeddings
  ├── Positional Encoding
  ├── Length-Control Embeddings
  ├── Multi-Head Self Attention
  ├── Cross Attention
  └── Feed Forward Network
  ▼
Generated Caption
```

---

# 🔥 Features

| 🚀 Feature                  | 📌 Description                                        |
| --------------------------- | ----------------------------------------------------- |
| 🧠 Encoder                  | Pretrained ResNet-50 fine-tuned from `layer3`         |
| 🤖 Decoder                  | 3-layer Transformer Decoder (`d_model=512`, 8 heads)  |
| 🎛️ Controllable Generation | Length bucket embeddings (short → very long captions) |
| 👀 Attention Visualisation  | Heatmaps showing where the model looks                |
| 🔍 Decoding                 | Greedy decoding + Beam Search                         |
| 🛡️ Regularisation          | Label smoothing, gradient clipping, LR scheduling     |
| 📊 Evaluation               | BLEU-1/2/3/4 metrics                                  |
| ⚡ Comparison Model          | BLIP image captioning baseline                        |

---

# 📂 Repository Structure

```bash
.
├── Homework_2_Image_Captioning_solved.ipynb
├── Dockerfile
├── requirements.txt
└── README.md
```

---

# ⚡ Quick Start

# 🐳 Option 1 — Docker (Recommended)

```bash
# Build Docker image
docker build -t image-captioning .

# Run container with GPU support
docker run --gpus all -p 8888:8888 \
    -v $(pwd)/coco:/workspace/coco \
    image-captioning
```

Then open:

```text
http://localhost:8888
```

---

# 💻 Option 2 — Local Environment

```bash
python -m venv venv

# Linux / Mac
source venv/bin/activate

# Windows
venv\Scripts\activate

pip install -r requirements.txt

jupyter lab Homework_2_Image_Captioning_solved.ipynb
```

---

# 📦 Dataset Setup

## 📥 Download MS COCO 2017

```bash
wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
```

## 📂 Extract Files

```bash
unzip train2017.zip -d coco/images/
unzip val2017.zip -d coco/images/
unzip annotations_trainval2017.zip -d coco/
```

---

# 📁 Expected Folder Structure

```bash
coco/
├── images/
│   ├── train2017/
│   └── val2017/
└── annotations/
    ├── captions_train2017.json
    └── captions_val2017.json
```

---

# 🧪 Notebook Walkthrough

| 📘 Section | 🔍 Description               |
| ---------- | ---------------------------- |
| 0          | Imports + reproducibility    |
| 1          | Dataset setup                |
| 2          | Image transformations        |
| 3          | Vocabulary + Dataset classes |
| 4          | Dataset visualisation        |
| 5          | Full model architecture      |
| 6          | Loss functions + optimisers  |
| 7          | Training loop                |
| 8          | Evaluation + BLEU metrics    |
| 9          | BLIP baseline                |
| 10         | BLIP evaluation              |
| 11         | Final comparison & analysis  |

---

# 📊 Results

## 🎯 BLEU Scores

| Metric | 🧠 Custom Transformer | 🤖 BLIP |
| ------ | --------------------- | ------- |
| BLEU-1 | ~0.55                 | ~0.72   |
| BLEU-2 | ~0.35                 | ~0.53   |
| BLEU-3 | ~0.22                 | ~0.38   |
| BLEU-4 | ~0.13                 | ~0.27   |

> ⚠️ Exact scores vary depending on hardware, subset size, and random seed.

---

# 👀 Attention Visualisation

The decoder stores cross-attention weights which are later visualised as heatmaps over the image.

<p align="center">
  <img src="https://viso.ai/wp-content/uploads/2024/04/attention-mechanism-visualization.jpg" width="700"/>
</p>

This helps interpret **which image regions influence each generated word**.

---

# 🤖 BLIP Comparison

The project also evaluates:

```python
Salesforce/blip-image-captioning-base
```

against the custom Transformer model using the same BLEU evaluation pipeline.

---

# 🛠️ Technologies Used

| Tool                     | Purpose                         |
| ------------------------ | ------------------------------- |
| PyTorch                  | Deep Learning                   |
| Torchvision              | CNN backbone + image transforms |
| HuggingFace Transformers | BLIP baseline                   |
| Matplotlib               | Visualisation                   |
| NLTK                     | BLEU metrics                    |
| Docker                   | Reproducible environment        |

---

# ⚙️ Training Details

| Parameter         | Value             |
| ----------------- | ----------------- |
| Encoder           | ResNet-50         |
| Decoder Layers    | 3                 |
| Attention Heads   | 8                 |
| Embedding Size    | 512               |
| Batch Size        | 32                |
| Optimizer         | AdamW             |
| Scheduler         | ReduceLROnPlateau |
| Label Smoothing   | 0.1               |
| Gradient Clipping | ✅                 |
| Early Stopping    | ✅                 |

---

# 💡 Future Improvements

* 🚀 CIDEr / ROUGE / METEOR evaluation
* 🧠 ViT encoder instead of ResNet
* ⚡ Mixed precision training
* 🔍 Better beam search tuning
* 🌍 Web app deployment
* 🎤 Speech caption generation

---

# 📸 Example Output

| 🖼️ Image           | 📝 Generated Caption                         |
| ------------------- | -------------------------------------------- |
| Dog running in snow | “a dog running through a snowy field”        |
| People eating pizza | “a group of friends eating pizza at a table” |
| City street         | “cars driving down a busy city street”       |

---

# 📈 Training Curves

<p align="center">
  <img src="https://miro.medium.com/v2/resize:fit:1200/1*yk16tgW3iXR53uM1xStI7w.png" width="700"/>
</p>

---

# ⭐ Why This Project Matters

This project demonstrates:

* Transformer-based sequence generation
* Attention mechanisms in vision-language tasks
* Fine-tuning CNN backbones
* Controlled text generation
* Explainable AI via attention visualisation

It combines **Computer Vision + NLP + Transformers** into one complete deep learning pipeline.

---

# 🙌 Acknowledgements

* MS COCO Dataset
* PyTorch Team
* HuggingFace Transformers
* Salesforce Research (BLIP)

---

# 📜 License

This project is intended for educational and research purposes.

---

<p align="center">
  ⭐ If you liked this project, consider starring the repository!
</p>
