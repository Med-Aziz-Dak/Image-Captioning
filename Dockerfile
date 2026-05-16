# ── Base image ─────────────────────────────────────────────────────────
# Use NVIDIA PyTorch image so CUDA is available out of the box.
# CPU-only users: replace with python:3.11-slim and remove cuda references.
FROM pytorch/pytorch:2.2.0-cuda11.8-cudnn8-runtime

# Avoid interactive prompts during apt installs
ENV DEBIAN_FRONTEND=noninteractive

# ── System dependencies ─────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        wget \
        unzip \
        libgl1-mesa-glx \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ───────────────────────────────────────────────────
WORKDIR /workspace

# ── Python dependencies ─────────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Download NLTK data at build time ────────────────────────────────────
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"

# ── Copy notebook ────────────────────────────────────────────────────────
COPY Homework_2_Image_Captioning_solved.ipynb .

# ── Expose Jupyter port ──────────────────────────────────────────────────
EXPOSE 8888

# ── Default command: start Jupyter Lab ──────────────────────────────────
CMD ["jupyter", "lab", \
     "--ip=0.0.0.0", \
     "--port=8888", \
     "--no-browser", \
     "--allow-root", \
     "--NotebookApp.token=''", \
     "--NotebookApp.password=''"]
