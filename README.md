

<div align="center">

<img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white"/>
<img src="https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white"/>
<img src="https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white"/>
<img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
<img src="https://img.shields.io/badge/Groq-F55036?style=for-the-badge&logo=groq&logoColor=white"/>

# ⬡ Geometry-Aware Variational Autoencoder

**A research-grade deep learning system comparing Euclidean vs. Spherical latent spaces with a futuristic real-time web dashboard**

[Overview](#-overview) · [Features](#-features) · [Architecture](#-architecture) · [Setup](#-quickstart-with-docker) · [Usage](#-usage) · [Results](#-results) · [Project Structure](#-project-structure) · [Tech Stack](#-tech-stack)

</div>

---

## 📌 Overview

Standard Variational Autoencoders (VAEs) assume a flat **Euclidean latent space** — forcing real-world data that lives on curved manifolds into a geometry it does not naturally fit. This project proves that a **Spherical latent space** better preserves the intrinsic structure of image data, producing superior clustering, smoother interpolation, and higher quality reconstructions.

This system implements and rigorously compares two complete VAE pipelines:

- **Gaussian VAE** — standard reparameterization with `z ~ N(0, I)`, Euclidean latent space
- **Spherical VAE** — latent vectors normalized to the unit hypersphere `S^(d-1)` using von Mises-Fisher inspired reparameterization

Both models are trained on **MNIST**, **Fashion-MNIST**, and **CIFAR-10**, with every experiment tracked, visualized, and analyzed through a fully interactive research dashboard powered by **Llama 3.3 70B via Groq**.

---

## ✨ Features

### 🧠 Model & Research
- Complete **Gaussian VAE** implementation with closed-form KL divergence
- Complete **Spherical VAE** with power-spherical reparameterization on `S^(d-1)`
- Von Mises-Fisher KL divergence approximation for spherical prior
- Numerical stability via logvar clamping and gradient clipping
- Cosine annealing learning rate scheduler
- Fast CPU training with configurable data subsets

### 📊 Evaluation & Metrics
- **Silhouette Score** — measures latent space clustering quality
- **SSIM** — structural similarity of reconstructions
- **Reconstruction Loss** (BCE) tracked per epoch
- **KL Divergence** tracked separately from reconstruction loss
- Side-by-side **Gaussian vs Spherical** comparison table
- PCA projection of high-dimensional latent vectors

### 🎨 Visualizations
- Real-time **loss curves** (train + val) generated after training
- **t-SNE** plots of latent space colored by class label
- **2D PCA latent scatter** plots with interactive Plotly rendering
- **Latent space interpolation** — smooth walk between two points
- **Reconstruction grid** — original vs reconstructed side by side
- **Random sample generation** from prior distribution

### 🤖 AI-Powered Research Assistant
- **AI Analysis** — Llama 3.3 70B analyzes your experiment results and gives research-grade insights
- **Interactive AI Chat** — ask anything about VAEs, geometry, results, math — answered by Llama 3.3 70B
- Automatic metric interpretation and improvement suggestions

### 🖥️ Dashboard
- Futuristic dark-theme web UI built with Flask + Plotly
- Real-time training progress with **actual epoch counter** (not simulated)
- Live loss and val metrics displayed during training
- Train multiple models sequentially with queue display
- All tabs: Dashboard, Train, Latent Space, Generate, Compare, AI Analysis

---

## 🏗️ Architecture

### Encoder (both models)
```
Input Image (28×28×1 or 32×32×3)
    → Flatten
    → Linear(784, 512) + BatchNorm + ReLU
    → Linear(512, 256) + BatchNorm + ReLU
    → Linear(256, 128) + ReLU
    → fc_mu(128, latent_dim)
    → fc_logvar(128, latent_dim)   [Gaussian]
    → fc_kappa(128, 1)             [Spherical]
```

### Geometry Layer — The Key Difference

| | Gaussian VAE | Spherical VAE |
|---|---|---|
| **Prior** | `z ~ N(0, I)` | Uniform on `S^(d-1)` |
| **Latent Space** | Unbounded Euclidean `R^d` | Unit hypersphere `S^(d-1)` |
| **Reparameterization** | `z = μ + σ · ε` | `z = w·μ + √(1-w²)·e` |
| **KL Divergence** | Closed-form Gaussian KL | vMF approximation |
| **Sampling** | `z ~ N(0,I)` | `z ~ Uniform(S^(d-1))` |

### Decoder (both models)
```
z (latent_dim)
    → Linear(latent_dim, 128) + ReLU
    → Linear(128, 256) + BatchNorm + ReLU
    → Linear(256, 512) + BatchNorm + ReLU
    → Linear(512, 784) + Sigmoid
    → Reshape to (B, C, H, W)
```

### Loss Function
```
Total Loss = BCE(x_recon, x) + β × KL(q(z|x) || p(z))

Gaussian:  KL = -0.5 × Σ(1 + log σ² - μ² - σ²)
Spherical: KL ≈ κ - (d/2 - 1) × log(κ)
```

---

## 🚀 Quickstart with Docker

### Prerequisites
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running
- [Groq API Key](https://console.groq.com/keys) (free)

### 1. Clone the repository
```bash
git clone https://github.com/alijaipuri/Geometry-Aware-VAE.git
cd Geometry-Aware-VAE
```

### 2. Build the Docker image
```bash
docker build -t geometry-vae .
```

### 3. Run the container
```bash
docker run -p 5050:5050 \
  -e GROQ_API_KEY="your_groq_api_key_here" \
  -v $(pwd)/results:/app/results \
  -v $(pwd)/data:/app/data \
  --name geo-vae \
  geometry-vae
```

### 4. Open the dashboard
```
http://localhost:5050
```

> **Datasets are downloaded automatically** — MNIST, Fashion-MNIST, and CIFAR-10 are fetched by torchvision on first training run. No manual download needed.

---

## 💻 Usage

### Training a Model
1. Open the dashboard at `http://localhost:5050`
2. Go to **Train** tab
3. Select model: `Gaussian VAE` or `Spherical VAE`
4. Select dataset: `MNIST`, `Fashion-MNIST`, or `CIFAR-10`
5. Click **Start Training**
6. Watch real-time epoch progress: `Epoch 7/15 | loss: 0.2341 | val: 0.2289 (46%)`

### Train All Models at Once
- Click **Train All Models** to queue all 4 combinations sequentially
- Each job shows live progress and marks ✓ Done when complete

### Visualize Latent Space
- Go to **Latent Space** tab → click **Visualize**
- Interactive Plotly scatter plot colored by class
- Silhouette score displayed below

### Generate Samples
- Go to **Generate** tab
- Set count (1–64) → click **Generate Samples**
- Samples drawn from the prior distribution

### Compare Models
- Go to **Compare** tab → click **Compare**
- Side-by-side metrics table
- Both latent space plots shown together

### AI Research Analysis
- Go to **AI Analysis** tab → click **Analyze Results**
- Llama 3.3 70B reads your actual metrics and gives research insights
- Use the **chat** to ask specific questions about your results

---

## 📈 Results

### What to Expect After Training

| Metric | Gaussian VAE | Spherical VAE |
|---|---|---|
| **Latent Structure** | Scattered, overlapping clusters | Tighter, more separated clusters |
| **Silhouette Score** | Lower (worse clustering) | Higher (better clustering) |
| **Interpolation** | Passes through empty regions | Smooth, stays on manifold |
| **KL Collapse** | Possible with high β | More stable due to bounded space |
| **Reconstruction** | Sharp after 10+ epochs | Sharp after 10+ epochs |

### Why Spherical is Better
- The hypersphere has **no boundary** — no empty corners like Euclidean space
- Distance on the sphere is **geodesic** — more faithful to data manifold
- Uniform prior on sphere = **no mode collapse** toward origin
- Concentration parameter κ gives **adaptive uncertainty** per sample

---

## 📁 Project Structure

```
Geometry-Aware-VAE/
│
├── 📄 Dockerfile                   # Container definition
├── 📄 requirements.txt             # Python dependencies
├── 📄 main.py                      # Entry point (web/train/evaluate/compare)
│
├── 📁 configs/
│   └── config.yaml                 # All hyperparameters
│
├── 📁 models/
│   ├── __init__.py
│   ├── vae_gaussian.py             # Standard Gaussian VAE
│   └── vae_spherical.py            # Spherical VAE (vMF latent space)
│
├── 📁 utils/
│   ├── __init__.py
│   ├── losses.py                   # BCE + KL loss functions
│   ├── helpers.py                  # DataLoaders, checkpoints, seeding
│   ├── metrics.py                  # Silhouette, SSIM, FID
│   └── visualization.py            # All plot generation functions
│
├── 📁 experiments/
│   ├── train.py                    # CLI training script
│   ├── evaluate.py                 # CLI evaluation script
│   └── compare_models.py           # Full comparison pipeline
│
├── 📁 web/
│   ├── app.py                      # Flask backend + all API routes
│   ├── templates/
│   │   └── index.html              # Single-page dashboard
│   └── static/
│       ├── css/style.css           # Dark futuristic theme
│       └── js/app.js               # All frontend logic + Plotly
│
└── 📁 results/                     # Generated during training
    ├── models/                     # Saved .pth checkpoints
    ├── plots/                      # Loss curves, t-SNE, latent plots
    └── logs/                       # JSON metrics per model/dataset
```

---

## ⚙️ Configuration

Edit `configs/config.yaml` to tune hyperparameters:

```yaml
gaussian:
  latent_dim: 20        # Dimensions of latent space
  batch_size: 256       # Training batch size
  epochs: 15            # Number of training epochs
  lr: 0.001             # Adam learning rate
  beta: 0.5             # KL weight (lower = sharper reconstructions)
  seed: 42              # Reproducibility seed

spherical:
  latent_dim: 20
  batch_size: 256
  epochs: 15
  lr: 0.001
  beta: 0.5
  seed: 42
```

**Beta tuning guide:**
- `β = 0.1` → Very sharp reconstructions, poor disentanglement
- `β = 0.5` → Good balance (recommended)
- `β = 1.0` → Strong regularization, blurrier reconstructions
- `β > 1.0` → β-VAE regime, maximum disentanglement

---

## 🛠️ Tech Stack

| Category | Technology |
|---|---|
| **Deep Learning** | PyTorch 2.2+ |
| **Datasets** | torchvision (MNIST, Fashion-MNIST, CIFAR-10) |
| **Web Framework** | Flask |
| **Visualization** | Matplotlib, Plotly.js, Seaborn |
| **ML Utilities** | scikit-learn (t-SNE, Silhouette, PCA) |
| **AI Assistant** | Groq API — Llama 3.3 70B Versatile |
| **Containerization** | Docker |
| **Config** | PyYAML |
| **Image Processing** | Pillow, NumPy |

---

## 🔬 Mathematical Background

### von Mises-Fisher Distribution
The spherical VAE uses the **vMF distribution** on `S^(d-1)`:

```
p(z | μ, κ) ∝ exp(κ · μᵀz)
```

Where:
- `μ ∈ S^(d-1)` is the mean direction (unit vector)
- `κ ≥ 0` is the concentration parameter
- Higher `κ` → distribution more concentrated around `μ`
- `κ = 0` → uniform distribution on the sphere (the prior)

### KL Divergence (Spherical)
```
KL(vMF(μ,κ) || Uniform(S^(d-1))) ≈ κ - (d/2 - 1)·log(κ) + constant
```

### Reparameterization on the Sphere
```
ε ~ N(0, I)
tangent = normalize(ε - (εᵀμ)μ)       # project to tangent plane
w = sigmoid(κ / √d)                    # interpolation weight
z = normalize(w·μ + (1-w)·tangent)     # combine and project back
```

---

## 🗺️ Future Work

- [ ] **Hyperbolic VAE** — Poincaré ball latent space for hierarchical data
- [ ] **Product space VAE** — combine spherical + Euclidean + hyperbolic
- [ ] **Conditional VAE** — class-conditioned generation
- [ ] **Disentanglement metrics** — MIG, SAP, DCI scores
- [ ] **CIFAR-10 full training** — CNN encoder/decoder for color images
- [ ] **FID score** — Fréchet Inception Distance for generation quality
- [ ] **Latent arithmetic** — semantic vector operations on the sphere

---

## 👤 Author

**Ali Jaipuri**
- GitHub: [@alijaipuri](https://github.com/alijaipuri)

---

## 📄 License

This project is licensed under the **MIT License** — free to use for academic and commercial purposes.

---

## 🙏 Acknowledgements

- [Mathieu et al. (2019)](https://arxiv.org/abs/1904.02113) — *Continuous Hierarchical Representations with Poincaré Variational Auto-Encoders*
- [Davidson et al. (2018)](https://arxiv.org/abs/1804.00891) — *Hyperspherical Variational Auto-Encoders*
- [Kingma & Welling (2013)](https://arxiv.org/abs/1312.6114) — *Auto-Encoding Variational Bayes*
- [Groq](https://groq.com) — for blazing fast Llama 3.3 70B inference

---

<div align="center">

**⭐ Star this repo if you found it useful for your research!**

</div>
