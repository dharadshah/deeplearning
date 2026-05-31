# Deep Learning  — Seven Notebooks from Perceptrons to Transfer Learning

A structured, hands-on deep learning curriculum built in Python and PyTorch,
covering seven core topics with full explanatory markdowns, working code, and
visualisations. Built as part of an AI/ML learning plan to demonstrate
production-grade understanding of deep learning fundamentals.

---

## What This Project Is

Seven Jupyter notebooks, each covering one deep learning topic end to end.
Every notebook follows the same structure: conceptual explanation in markdown,
manual implementation to expose the underlying mechanics, PyTorch implementation,
training, evaluation, and a key takeaways section written as a learning journal.

The goal is not just working code — it is demonstrable understanding of what
each component computes and why it is designed the way it is.

---

## Notebooks

| # | Topic | Dataset | Key Concepts |
|---|---|---|---|
| 01 | Neural Networks | Iris (sklearn) | Perceptron mechanics, activation functions, loss functions, training loop |
| 02 | PyTorch and Keras | MNIST | Framework comparison, nn.Module, model.fit, callbacks, EarlyStopping |
| 03 | CNNs | CIFAR-10 | Convolution, pooling, BatchNorm, feature maps, filter visualisation, MLP vs CNN |
| 04 | RNNs and LSTMs | Air Quality UCI | Hidden state, vanishing gradients, LSTM gates, sliding window, sequence length experiment |
| 05 | GANs | MNIST | Adversarial training, Generator, Discriminator, latent space interpolation, mode collapse |
| 06 | Transformers | IMDb (HuggingFace) | Scaled dot-product attention, multi-head attention, positional encoding, encoder, CLS token |
| 07 | Transfer Learning | Oxford Flowers 102 + IMDb | Feature extraction, fine-tuning, differential learning rates, Grad-CAM, DistilBERT |

---

## Results Summary

### Vision

| Method | Dataset | Test Accuracy |
|---|---|---|
| Baseline CNN (from scratch) | CIFAR-10 | ~70% |
| MLP (from scratch) | CIFAR-10 | ~55% |
| Feature extraction (ResNet18) | Flowers 102 | significantly above baseline |
| Fine-tuning (ResNet18) | Flowers 102 | best vision result |

### NLP

| Method | Dataset | Test Accuracy |
|---|---|---|
| From-scratch Transformer | IMDb (5k samples) | varies by run |
| DistilBERT frozen backbone | IMDb (500 samples) | competitive with fine-tuning |
| DistilBERT full fine-tuning | IMDb (500 samples) | varies by data scale |

### Time-Series

| Method | Dataset | Metric |
|---|---|---|
| Plain RNN | Air Quality UCI | MAE in mg/m3 |
| LSTM (seq_len=24) | Air Quality UCI | lower MAE than RNN |

---

## Tech Stack

| Tool | Version | Purpose |
|---|---|---|
| Python | 3.13 | Language |
| PyTorch | 2.11 | Primary deep learning framework |
| TensorFlow / Keras | 2.x | Framework comparison in Notebook 02 |
| torchvision | latest | Datasets and pre-trained models |
| HuggingFace datasets | latest | IMDb dataset |
| HuggingFace transformers | latest | DistilBERT |
| NumPy | latest | Numerical operations |
| Pandas | latest | Data loading and preprocessing |
| Matplotlib | latest | All visualisations |
| scikit-learn | latest | Metrics and preprocessing |

---

## Project Structure

```
DeepLearningPhase5/
├── notebooks/
│   ├── 01_neural_networks.ipynb
│   ├── 02_pytorch_keras.ipynb
│   ├── 03_cnns.ipynb
│   ├── 04_rnns_lstms.ipynb
│   ├── 05_gans.ipynb
│   ├── 06_transformers.ipynb
│   └── 07_transfer_learning.ipynb
├── data/                        # datasets downloaded here (gitignored)
├── assets/                      # saved plots if needed
├── requirements.txt
├── pyproject.toml
└── README.md
```

---

## Setup

**Requirements:** Python 3.10+, Poetry

```powershell
# clone the repo
git clone https://github.com/YOUR_USERNAME/DeepLearningPhase5.git
cd DeepLearningPhase5

# install dependencies
poetry install

# register the kernel
poetry run python -m ipykernel install --user --name=dl-phase5 --display-name "Deep Learning Phase 5"

# launch JupyterLab
poetry run jupyter lab
```

Open any notebook from the `notebooks/` folder and select the
**Deep Learning Phase 5** kernel.

**Note on datasets:**
Most datasets download automatically on first run via torchvision or HuggingFace.
The Air Quality UCI dataset (Notebook 04) requires a manual download:
1. Go to https://archive.ics.uci.edu/dataset/360/air+quality
2. Download and extract `AirQualityUCI.csv`
3. Place it in the `data/` folder

---

## What Each Notebook Demonstrates

**01 — Neural Networks**
Builds a perceptron manually in NumPy before touching PyTorch. Implements and
plots all four activation functions from scratch. Builds a 3-layer MLP using
nn.Module and trains it on Iris classification. The focus is understanding every
line of the training loop before using any abstractions.

**02 — PyTorch and Keras**
Builds the same MNIST classifier in both frameworks side by side. Covers
nn.Module patterns, model.train() vs model.eval(), torch.no_grad(), and learning
rate schedulers in PyTorch. Covers Sequential vs Functional API, model.compile,
model.fit, and callbacks in Keras. Ends with a direct accuracy and code
complexity comparison.

**03 — CNNs**
Starts with manual 2D convolution using hand-coded Sobel, sharpen, and blur
filters to show what convolution computes before any learning. Builds a 3-block
CNN on CIFAR-10. Visualises learned filters, feature maps at each depth using
forward hooks, and most vs least activated filters. Ends with a direct MLP vs
CNN accuracy comparison on identical data.

**04 — RNNs and LSTMs**
Implements a plain RNN forward pass manually in NumPy and demonstrates the
vanishing gradient problem with a gradient magnitude plot. Implements the full
LSTM forward pass with all four gates manually before using nn.LSTM. Trains both
on real air quality time-series data. Includes a sequence length experiment
across four window sizes to find the optimal temporal context empirically.

**05 — GANs**
Builds a DCGAN-style MLP GAN from scratch on MNIST. Explains the minimax
objective with loss function plots. Uses a fixed noise vector to track the
Generator's visual progress at checkpoints. Implements latent space interpolation
to reveal what the Generator learned. Covers mode collapse, discriminator
dominance, and training instabilities with simulated examples and a health
diagnostic.

**06 — Transformers**
Implements scaled dot-product attention manually in NumPy and verifies against
PyTorch's built-in. Builds MultiHeadAttention, sinusoidal PositionalEncoding,
and EncoderLayer as nn.Module from scratch. Adds a CLS token classification head.
Trains on IMDb sentiment. Visualises attention weights on real reviews. Ends with
a direct comparison against a bidirectional LSTM on identical data.

**07 — Transfer Learning**
Compares three strategies on Oxford Flowers 102: from-scratch baseline CNN,
feature extraction with frozen ResNet18, and fine-tuning with differential
learning rates. Visualises first-layer filters before and after training, feature
maps at each residual block, and Grad-CAM heatmaps. Fine-tunes DistilBERT on
IMDb and compares frozen vs partial vs full fine-tuning. Ends with a decision
framework for choosing the right transfer learning strategy on new tasks.

---

## Key Learning Outcomes

- Understand what each deep learning component computes at the mathematical level
- Implement attention, LSTM gates, convolution, and GAN training from scratch
- Debug training issues from loss curves and gradient flow analysis
- Choose the right architecture for spatial (CNN), sequential (LSTM), and
  long-range (Transformer) data
- Apply transfer learning correctly with appropriate learning rate strategies
- Interpret model behaviour through filter visualisation, attention heatmaps,
  and Grad-CAM

---

## Author

Dhara Shah
AI/ML Engineer — ePGD AI & DS, IIT Bombay

---

## License

MIT License — free to use, adapt, and share with attribution.
