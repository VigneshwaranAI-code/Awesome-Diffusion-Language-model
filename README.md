<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:0f0c29,50:302b63,100:24243e&height=200&section=header&text=Awesome%20Diffusion%20Language%20Models&fontSize=36&fontColor=ffffff&fontAlignY=38&desc=From%20noise%20to%20meaning%20—%20visualized%2C%20demystified%2C%20understood&descAlignY=58&descSize=16&animation=fadeIn" width="100%"/>

<br/>

[![Awesome](https://awesome.re/badge-flat2.svg)](https://awesome.re)
![Stars](https://img.shields.io/github/stars/yourusername/Awesome-Diffusion-Language-model?style=flat-square&color=f5a623&logo=github)
![Forks](https://img.shields.io/github/forks/yourusername/Awesome-Diffusion-Language-model?style=flat-square&color=6c63ff)
![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)
![License](https://img.shields.io/badge/license-MIT-blue.svg?style=flat-square)
![Last Commit](https://img.shields.io/github/last-commit/yourusername/Awesome-Diffusion-Language-model?style=flat-square&color=ff6b6b)
![Visitors](https://visitor-badge.laobi.icu/badge?page_id=yourusername.Awesome-Diffusion-Language-model&style=flat-square)

<br/>

> **The most visual, intuitive, and beginner-friendly guide to understanding how Diffusion Language Models generate text — step by step, concept by concept.**

<br/>

[🚀 Start Here](#-what-is-a-diffusion-language-model) · [📊 Visualizations](#-interactive-visualizations) · [📚 Papers](#-curated-paper-list) · [💻 Code](#-implementation-guides) · [🤝 Contribute](#-contributing)

<br/>
<br/>

</div>

---

## 📌 Table of Contents

- [✨ Why This Repo Exists](#-why-this-repo-exists)
- [🧠 What Is a Diffusion Language Model?](#-what-is-a-diffusion-language-model)
- [🔬 Core Concepts — Visually Explained](#-core-concepts--visually-explained)
  - [The Forward Process (Adding Noise)](#the-forward-process-adding-noise)
  - [The Reverse Process (Denoising Text)](#the-reverse-process-denoising-text)
  - [Masked Diffusion vs. Gaussian Diffusion](#masked-diffusion-vs-gaussian-diffusion)
  - [Score Function & Denoising Score Matching](#score-function--denoising-score-matching)
- [🗺️ The Architecture Landscape](#️-the-architecture-landscape)
- [📊 Interactive Visualizations](#-interactive-visualizations)
- [⚔️ DLM vs. Autoregressive LLMs](#️-dlm-vs-autoregressive-llms)
- [📚 Curated Paper List](#-curated-paper-list)
- [💻 Implementation Guides](#-implementation-guides)
- [🏋️ Training From Scratch](#️-training-from-scratch)
- [🔨 Build a DLM from Scratch](#-build-a-dlm-from-scratch)
  - [Step 0 — Project Blueprint](#step-0--project-blueprint)
  - [Step 1 — Tokenizer & Vocabulary](#step-1--tokenizer--vocabulary)
  - [Step 2 — Noise Schedule](#step-2--noise-schedule)
  - [Step 3 — Forward Process](#step-3--forward-process)
  - [Step 4 — The Denoiser (Transformer)](#step-4--the-denoiser-transformer)
  - [Step 5 — Loss Function](#step-5--loss-function)
  - [Step 6 — Full Training Loop](#step-6--full-training-loop)
  - [Step 7 — Sampling & Generation](#step-7--sampling--generation)
  - [Step 8 — Putting It All Together](#step-8--putting-it-all-together)
- [📈 Benchmarks & Evaluation](#-benchmarks--evaluation)
- [🌍 Real-World Applications](#-real-world-applications)
- [🛠️ Tools & Libraries](#️-tools--libraries)
- [🤝 Contributing](#-contributing)
- [🙏 Acknowledgements](#-acknowledgements)

---

## ✨ Why This Repo Exists

> **Problem:** Diffusion Language Models are one of the most promising frontiers in AI research — yet most resources assume you already have a PhD in probability theory.
>
> **Solution:** This repo meets you where you are. Every concept is explained with visual diagrams, step-by-step walkthroughs, and working code — whether you're a curious beginner or a seasoned researcher.

This is **not** just a list of links. This is a **structured learning journey** from zero to research-level understanding.

```
You start here:          "What even IS a Diffusion Language Model?"
You end here:            Implementing your own, reading frontier papers, contributing research.
```

---

## 🧠 What Is a Diffusion Language Model?

Diffusion models were initially a triumph of **image generation** (Stable Diffusion, DALL·E). But what happens when we apply the same iterative refinement idea to **language** — sequences of tokens, sentences, paragraphs?

That is the question Diffusion Language Models (DLMs) are answering.

### The Core Idea in 3 Sentences

1. **Forward:** Gradually corrupt a real sentence by masking or noising tokens until nothing meaningful remains.
2. **Learn:** Train a neural network to predict and undo small corruption steps.
3. **Generate:** Start from pure noise and iteratively denoise until a fluent sentence emerges.

### 🖼️ Big Picture — The Paradigm Shift

```
AUTOREGRESSIVE MODELS (GPT-style)          DIFFUSION LANGUAGE MODELS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━          ━━━━━━━━━━━━━━━━━━━━━━━━━
                                          
  "The"→"cat"→"sat"→"on"→"mat"            [MASK][MASK][MASK][MASK][MASK]
                                                       ↓   (step T)
  • Left-to-right only                     [MASK][ on ][MASK][MASK][MASK]
  • Sequential token generation                        ↓   (step T/2)
  • Fixed generation order                 [The ][cat ][ sat][ on ][MASK]
  • Excellent fluency                                  ↓   (step 1)
                                           [The ][cat ][ sat][ on ][ mat]
                                          
                                          • Parallel token refinement
                                          • Any-order generation
                                          • Iterative global coherence
                                          • Strong controllability
```

---

## 🔬 Core Concepts — Visually Explained

### The Forward Process (Adding Noise)

The forward process defines **how we corrupt data**. In continuous diffusion (for images), we add Gaussian noise. For discrete token sequences, we use different strategies:

#### 🎭 Absorbing/Mask Diffusion

Each token independently transitions to a `[MASK]` state at each timestep `t`:

```
t=0   |  The  |  quick  |  brown  |  fox   |  jumps  |
       ──────────────────────────────────────────────
t=1   |  The  |  quick  | [MASK]  |  fox   |  jumps  |   ← 1 token masked
       ──────────────────────────────────────────────
t=2   |  The  | [MASK]  | [MASK]  | [MASK] |  jumps  |   ← 3 tokens masked
       ──────────────────────────────────────────────
t=3   | [MASK]| [MASK]  | [MASK]  | [MASK] | [MASK]  |   ← fully masked
       ──────────────────────────────────────────────
       
  Masking probability at time t:   q(xₜ | x₀) = Mask(x₀, β_t)
  where β_t increases monotonically from 0 → 1
```

#### 🌡️ Uniform Token Diffusion

Tokens diffuse uniformly across the vocabulary — any token can become any other:

```
t=0  →  "fox"
t=1  →  "dog"    ← replaced with random vocab token
t=2  →  "the"    ← replaced again
t=3  →  "zxq"    ← near-uniform distribution
         ↑
   High entropy: model has lost all signal
```

#### 📐 The Formal Mathematics

For absorbing diffusion, the forward posterior is:

```
q(xₜ | x₀) = Cat(xₜ ; x₀ · (1 − βₜ) + βₜ / K)

  where:
    xₜ         = token at time t
    x₀         = original token
    βₜ ∈ [0,1] = noise schedule at step t
    K          = vocabulary size
    Cat(·)     = categorical distribution
```

---

### The Reverse Process (Denoising Text)

The **reverse process** is where the magic happens. A neural network `p_θ` learns to undo the corruption:

```
REVERSE DIFFUSION — GENERATION FLOW

  z_T ~ Uniform noise
    │
    ▼
  ┌─────────────────────────────────┐
  │        Denoising Network        │
  │   (Transformer + Time Embed)    │
  │                                 │
  │  Input:  [MASK][MASK][MASK]...  │
  │  Output: P(x₀ | zₜ, t)         │
  └──────────────┬──────────────────┘
                 │   sample from predicted distribution
                 ▼
  z_{T-1}: [MASK][ a ][MASK][ the ]...
                 │
                 ▼ (repeat T times)
                 │
  z_0:    [ The ][ cat ][ sat ][ on ][ mat ]
          ═══ FINAL GENERATED TEXT ═══
```

#### The Training Objective

The model is trained by **denoising score matching**:

```
L = E_{t, x₀, xₜ} [ -log p_θ(x₀ | xₜ, t) ]

  Intuitively: "Given a noisy sentence at time t,
                how well can you predict the original?"
```

---

### Masked Diffusion vs. Gaussian Diffusion

| Property | Masked Diffusion | Gaussian Diffusion on Embeddings |
|---|---|---|
| **Noise Space** | Discrete tokens → `[MASK]` | Continuous embedding vectors |
| **Corruption** | Token → masked w/ prob β_t | x → x + ε · σ_t |
| **Denoiser** | Predicts original token ID | Predicts score ∇ₓ log p(x) |
| **Sampling** | Categorical sampling | Langevin dynamics / DDPM/DDIM |
| **Coherence** | Strong (discrete stays in vocab) | Risk of leaving embedding manifold |
| **Examples** | MDLM, SEDD, D3PM | CDCD, Diffusion-LM |

---

### Score Function & Denoising Score Matching

The **score function** is the gradient of the log-probability of the data:

```
Score Function:   s(x) = ∇ₓ log p(x)
                          ↑
                 "Which direction makes x more likely?"

Denoising Score Matching Loss:
  L_DSM = E[||s_θ(xₜ, t) − ∇_{xₜ} log q(xₜ|x₀)||²]

  In practice: train network to predict the noise that was added
  ≡ train network to predict the clean data from the noisy data
```

**Visualization of the score field:**

```
Token Probability Landscape at time t

  High                                    
   │    ╭──╮       ╭──╮       ╭──╮
   │   ╱    ╲     ╱    ╲     ╱    ╲     ← peaks = likely tokens
   │  ╱      ╲   ╱      ╲   ╱      ╲
   │ ╱        ╲_╱        ╲_╱        ╲
  Low ─────────────────────────────── tokens
        "cat"  "dog"  "fox"  "the"

  Score arrows point UPHILL → toward high-probability tokens
  Denoising = following the score arrows from noise to data
```

---

## 🗺️ The Architecture Landscape

```
                    DIFFUSION LANGUAGE MODELS
                           │
           ┌───────────────┼───────────────┐
           │               │               │
    ┌──────▼──────┐ ┌──────▼──────┐ ┌─────▼───────┐
    │  Discrete   │ │  Continuous │ │   Hybrid    │
    │  Diffusion  │ │  Diffusion  │ │  Approaches │
    └──────┬──────┘ └──────┬──────┘ └─────┬───────┘
           │               │               │
    ┌──────▼──────┐ ┌──────▼──────┐ ┌─────▼───────┐
    │  D3PM       │ │ Diffusion-LM│ │  MDLM       │
    │  SEDD       │ │ CDCD        │ │  Plaid      │
    │  MDLM       │ │ SED         │ │  DiffuSeq   │
    │  MaskGIT    │ │ LD4LG       │ └─────────────┘
    └─────────────┘ └─────────────┘
    
    ┌─────────────────────────────────────────────┐
    │            BACKBONE ARCHITECTURES           │
    │                                             │
    │  Transformer (BERT-style, bidirectional)    │
    │  ↳ Full self-attention over all positions   │
    │                                             │
    │  UNet (adapted from image diffusion)        │
    │  ↳ Hierarchical feature processing          │
    │                                             │
    │  Mamba / SSM (state space models)           │
    │  ↳ Efficient long-sequence handling         │
    └─────────────────────────────────────────────┘
```

---

## 📊 Interactive Visualizations

All notebooks below are **self-contained** and run in Google Colab with a free GPU runtime.

### 🎬 Visualization Gallery

| # | Visualization | What You'll See | Open |
|---|---|---|---|
| 01 | **Forward Diffusion on Text** | Watch a sentence get destroyed step-by-step | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](#) |
| 02 | **Reverse Denoising Animation** | Token probabilities evolving during generation | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](#) |
| 03 | **Noise Schedule Comparison** | Linear vs. cosine vs. sqrt schedules | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](#) |
| 04 | **Score Field Visualization** | Gradient arrows on embedding space | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](#) |
| 05 | **Attention Pattern During Denoising** | How attention changes across timesteps | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](#) |
| 06 | **Parallel vs. Sequential Generation** | DLM vs. GPT side-by-side token reveal | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](#) |
| 07 | **Classifier-Free Guidance** | How conditioning shapes the denoising | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](#) |
| 08 | **Vocabulary Diffusion Matrix** | The Q(t|0) transition matrix visualized | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](#) |

---

## ⚔️ DLM vs. Autoregressive LLMs

Understanding where Diffusion Language Models fit versus GPT-style models:

```
┌────────────────────────────────────────────────────────────────────┐
│                    HEAD-TO-HEAD COMPARISON                         │
├───────────────────────┬────────────────────┬───────────────────────┤
│ Property              │ Autoregressive     │ Diffusion LM          │
├───────────────────────┼────────────────────┼───────────────────────┤
│ Generation Direction  │ Left → Right only  │ Any order, parallel   │
│ Speed (raw)           │ ✅ Fast (1 pass)   │ ⚠️ Slower (T passes)  │
│ Speed (with caching)  │ ✅ KV Cache        │ 🔬 Active research    │
│ Bidirectional Context │ ❌ Left-only       │ ✅ Full context        │
│ Controllability       │ ⚠️ Via prompting   │ ✅ Native constraints  │
│ Infilling / Editing   │ ⚠️ Awkward         │ ✅ Natural             │
│ Training Objective    │ Next-token predict │ Denoising             │
│ Theoretical Grounding │ Maximum Likelihood │ Score Matching / ELBO │
│ Long-form Coherence   │ ✅ Strong          │ 🔬 Improving           │
│ Likelihood Evaluation │ ✅ Exact           │ ⚠️ ELBO bound          │
│ Sampling Diversity    │ ⚠️ Repetitive      │ ✅ More diverse        │
└───────────────────────┴────────────────────┴───────────────────────┘

WHEN TO USE WHAT:
  ✅ Choose Autoregressive for:  chat, reasoning, code completion
  ✅ Choose Diffusion LM for:    constrained generation, infilling,
                                 controlled style/content, editing
```

---

## 📚 Curated Paper List

Papers organized by topic with difficulty ratings and visual summaries.

### 🏛️ Foundational Papers

| Paper | Venue | Key Contribution | Difficulty |
|---|---|---|---|
| [**D3PM**: Structured Denoising Diffusion Models in Discrete State Spaces](https://arxiv.org/abs/2107.03006) | NeurIPS 2021 | First principled discrete diffusion framework | ⭐⭐⭐ |
| [**Diffusion-LM**: Improving Controllable Text Generation](https://arxiv.org/abs/2205.14217) | NeurIPS 2022 | Continuous diffusion on embeddings; control via plug-in classifiers | ⭐⭐⭐ |
| [**MDLM**: Masked Diffusion Language Models](https://arxiv.org/abs/2406.07524) | 2024 | Simplified, scalable masked diffusion; SOTA on language benchmarks | ⭐⭐ |
| [**SEDD**: Discrete Diffusion Modeling by Estimating the Ratios of the Data Distribution](https://arxiv.org/abs/2310.16834) | ICML 2024 | Score entropy for discrete spaces; first DLM to match GPT-2 perplexity | ⭐⭐⭐⭐ |

### 🔬 Advanced & Specialized

| Paper | Focus | Key Idea |
|---|---|---|
| [**DiffuSeq**](https://arxiv.org/abs/2210.08933) | Seq2Seq | Conditional text generation via diffusion |
| [**GENIE**](https://arxiv.org/abs/2212.11685) | Long-form | Latent diffusion for long document generation |
| [**CDCD**](https://arxiv.org/abs/2211.15029) | Continuous | Continuous diffusion with classifier-free guidance |
| [**Plaid**](https://arxiv.org/abs/2306.05445) | Efficient | Latent space compression for faster DLMs |
| [**LD4LG**](https://arxiv.org/abs/2212.09462) | Latent | Latent diffusion adapted from image to language |
| [**DiffAR**](https://arxiv.org/abs/2305.04114) | Hybrid | Combining AR and diffusion for best of both |

### 📖 Survey Papers (Start Here)

| Paper | Scope |
|---|---|
| [A Survey of Diffusion Models in Natural Language Processing](https://arxiv.org/abs/2305.14671) | Comprehensive NLP coverage |
| [Diffusion Models for Non-autoregressive Text Generation: A Survey](https://arxiv.org/abs/2305.12972) | NAR-focused comparison |

---

## 💻 Implementation Guides

### 🟢 Beginner: Minimal Masked Diffusion in 50 Lines

```python
import torch
import torch.nn.functional as F
from transformers import BertTokenizer, BertForMaskedLM

# ── Config ─────────────────────────────────────────────────────────
T        = 100           # diffusion timesteps
MASK_ID  = 103           # [MASK] token ID in BERT vocab
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model     = BertForMaskedLM.from_pretrained("bert-base-uncased").to(DEVICE)

# ── Forward Process: corrupt x₀ → xₜ ──────────────────────────────
def q_sample(x0: torch.Tensor, t: int) -> torch.Tensor:
    """Mask each token independently with probability t/T."""
    mask_prob = t / T
    mask      = torch.bernoulli(torch.full_like(x0, mask_prob, dtype=torch.float))
    return torch.where(mask.bool(), torch.full_like(x0, MASK_ID), x0)

# ── Reverse Step: denoise xₜ → x₀ prediction ──────────────────────
@torch.no_grad()
def p_sample(xt: torch.Tensor, t: int) -> torch.Tensor:
    """Single reverse step: predict x₀, then re-noise to t-1."""
    logits   = model(xt).logits          # (B, L, V)
    x0_pred  = logits.argmax(dim=-1)     # greedy decode
    xt_prev  = q_sample(x0_pred, t - 1)  # re-noise to t-1
    return xt_prev

# ── Generation Loop ────────────────────────────────────────────────
def generate(prompt_length: int = 20) -> str:
    # Start from fully masked sequence
    xt = torch.full((1, prompt_length), MASK_ID, dtype=torch.long).to(DEVICE)
    
    for t in range(T, 0, -1):
        xt = p_sample(xt, t)
        if t % 20 == 0:
            print(f"  t={t:3d}: {tokenizer.decode(xt[0])}")
    
    return tokenizer.decode(xt[0], skip_special_tokens=True)

print("Generated:", generate())
```

### 🟡 Intermediate: Noise Schedule Visualization

```python
import numpy as np
import matplotlib.pyplot as plt

def linear_schedule(T, beta_start=0.0001, beta_end=0.02):
    return np.linspace(beta_start, beta_end, T)

def cosine_schedule(T, s=0.008):
    steps = np.arange(T + 1)
    alphas_cumprod = np.cos(((steps / T) + s) / (1 + s) * np.pi / 2) ** 2
    alphas_cumprod /= alphas_cumprod[0]
    return np.clip(1 - (alphas_cumprod[1:] / alphas_cumprod[:-1]), 0.0001, 0.9999)

def sqrt_schedule(T):
    return np.sqrt(np.linspace(0, 1, T))

T = 1000
schedules = {
    "Linear":  np.cumsum(linear_schedule(T)),
    "Cosine":  np.cumsum(cosine_schedule(T)),
    "Sqrt":    np.cumsum(sqrt_schedule(T)),
}

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for name, beta_cumsum in schedules.items():
    axes[0].plot(beta_cumsum / beta_cumsum.max(), label=name)
    signal_ratio = 1 - beta_cumsum / beta_cumsum.max()
    axes[1].plot(signal_ratio, label=name)

axes[0].set_title("Cumulative Noise (βₜ_cumsum)"); axes[0].legend()
axes[1].set_title("Signal Remaining"); axes[1].legend()
plt.tight_layout(); plt.savefig("noise_schedules.png", dpi=150)
```

### 🔴 Advanced: SEDD Score Entropy Training

```python
# Score Entropy for Discrete Diffusion (SEDD - Lou et al., 2024)
# Full training loop with absorbing diffusion and score entropy loss

import torch
import torch.nn as nn

class ScoreEntropyLoss(nn.Module):
    """
    Score Entropy Loss for discrete diffusion.
    Estimates the ratio s_θ(xₜ, t)[x'] = p(x' | xₜ, t) / p(xₜ | xₜ, t)
    """
    def __init__(self, vocab_size: int):
        super().__init__()
        self.V = vocab_size

    def forward(
        self,
        score_logits: torch.Tensor,  # (B, L, V): log ratio predictions
        x0: torch.Tensor,             # (B, L): original tokens
        xt: torch.Tensor,             # (B, L): noised tokens
        t: torch.Tensor,              # (B,):   timesteps
    ) -> torch.Tensor:
        # Compute per-position cross-entropy over ratio predictions
        log_ratios = F.log_softmax(score_logits, dim=-1)           # (B, L, V)
        target     = x0                                              # (B, L)
        loss       = F.nll_loss(
            log_ratios.view(-1, self.V),
            target.view(-1),
            reduction="mean"
        )
        return loss
```

---

## 🏋️ Training From Scratch

### Full Training Pipeline

```
DATA PIPELINE
─────────────
Raw Text Corpus
      ↓
  Tokenize (BPE / WordPiece)
      ↓
  Pack into fixed-length sequences
      ↓
  DataLoader (with shuffle + prefetch)
      
TRAINING LOOP
─────────────
  for each batch x₀:
    1. Sample timestep t ~ Uniform[1, T]
    2. Corrupt: xₜ = q_sample(x₀, t)         ← forward process
    3. Predict: x̂₀ = model(xₜ, t)            ← neural network
    4. Loss:    L  = -log p_θ(x₀ | xₜ, t)    ← denoising objective
    5. Backprop + gradient clip + optimizer step
    6. Log: loss, perplexity, grad norm

EVALUATION
──────────
  • Bits-per-character (BPC) / Bits-per-token (BPT)
  • ELBO (Evidence Lower Bound)
  • Generation quality: MAUVE, BLEU, BERTScore
  • Generalization gap (train vs. val loss)
```

### Recommended Hyperparameters (Starting Point)

| Hyperparameter | Small (debug) | Medium | Large |
|---|---|---|---|
| Parameters | ~10M | ~110M | ~800M |
| Timesteps T | 100 | 1000 | 1000 |
| Batch Size | 64 | 256 | 1024 |
| LR | 3e-4 | 1e-4 | 3e-5 |
| Warmup Steps | 500 | 2000 | 10000 |
| Noise Schedule | linear | cosine | cosine |
| Optimizer | AdamW | AdamW | AdamW |
| Grad Clip | 1.0 | 1.0 | 1.0 |

---

## 📈 Benchmarks & Evaluation

### How to Evaluate a Diffusion Language Model

```
┌─────────────────────────────────────────────────────────────────┐
│  EVALUATION METRICS — WHAT THEY MEASURE AND WHY                │
├──────────────────┬──────────────────────────────────────────────┤
│ Metric           │ What It Tells You                           │
├──────────────────┼──────────────────────────────────────────────┤
│ Perplexity       │ How surprised the model is by real text     │
│ BPT (bits/token) │ Compression efficiency; lower = better      │
│ ELBO             │ Lower bound on log-likelihood               │
│ MAUVE            │ How close generated text is to human text   │
│ Diversity        │ Distinct n-grams across generations         │
│ Coherence        │ Do sentences connect logically?             │
│ Controllability  │ % time model honors hard constraints        │
└──────────────────┴──────────────────────────────────────────────┘
```

### Published Benchmarks (text8, OpenWebText)

| Model | BPT (text8) ↓ | Perplexity ↓ | Year |
|---|---|---|---|
| Transformer LM (AR) | 1.13 | — | 2019 |
| D3PM (absorbing) | 1.45 | — | 2021 |
| MDLM | 1.31 | — | 2024 |
| SEDD | **1.16** | **\~GPT-2** | 2024 |
| Best AR (baseline) | 1.07 | — | 2023 |

> 📌 DLMs are rapidly closing the gap with autoregressive models, while gaining unique capabilities around bidirectionality and controllability.

---

## 🌍 Real-World Applications

Where Diffusion Language Models shine in practice:

```
┌─────────────────────────────────────────────────────────────────┐
│ 🖊️  TEXT INFILLING & EDITING                                    │
│     Fill in the [BLANK] given surrounding context              │
│     e.g. Document completion, code repair, story expansion     │
├─────────────────────────────────────────────────────────────────┤
│ 🎛️  CONSTRAINED GENERATION                                     │
│     Generate text satisfying hard constraints                  │
│     e.g. "Write a sentence with exactly these keywords"        │
├─────────────────────────────────────────────────────────────────┤
│ 💊  MOLECULE & PROTEIN SEQUENCE DESIGN                         │
│     Discrete sequences with structural constraints             │
│     e.g. Drug discovery, enzyme engineering                    │
├─────────────────────────────────────────────────────────────────┤
│ 🎵  MUSIC & SYMBOLIC GENERATION                                │
│     Token sequences with strong global structure               │
│     e.g. MIDI generation, chord progression design             │
├─────────────────────────────────────────────────────────────────┤
│ 🔄  STYLE TRANSFER                                              │
│     Interpolate between two text distributions                 │
│     e.g. Formal ↔ Casual, Positive ↔ Negative sentiment       │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🛠️ Tools & Libraries

| Library | Purpose | Link |
|---|---|---|
| 🤗 **Hugging Face Diffusers** | Unified diffusion API (image + text) | [github](https://github.com/huggingface/diffusers) |
| **mdlm** (official) | Reference MDLM implementation | [github](https://github.com/kuleshov-group/mdlm) |
| **SEDD** (official) | Score Entropy Discrete Diffusion | [github](https://github.com/louaaron/Score-Entropy-Discrete-Diffusion) |
| **DiffuSeq** | Seq2Seq diffusion | [github](https://github.com/Shark-NLP/DiffuSeq) |
| **tqdm + wandb** | Training monitoring | Standard |
| **einops** | Tensor manipulation for transformers | [github](https://github.com/arogozhnikov/einops) |

---

## 🤝 Contributing

This repo lives by community contribution! Here's how to add value:

```bash
# 1. Fork the repo
git clone https://github.com/yourusername/Awesome-Diffusion-Language-model
cd Awesome-Diffusion-Language-model

# 2. Create a feature branch
git checkout -b feat/add-new-visualization

# 3. Make your changes (notebook, diagram, paper, or code)

# 4. Open a Pull Request with a clear title and description
```

### Contribution Types We Love ❤️

- 📊 **New visualizations** — animations, diagrams, interactive notebooks
- 📄 **Paper summaries** — 1-paragraph plain-English explanations of new papers
- 💻 **Code examples** — minimal, well-commented implementations
- 🐛 **Bug fixes** — typos, broken links, incorrect math
- 🌐 **Translations** — README in other languages
- ❓ **Questions** — open an Issue; great questions become new sections!

### Code of Conduct

- Be kind and constructive. Beginners are welcome.
- Cite all sources properly.
- Prefer clarity over cleverness in code examples.

---

## 📂 Repository Structure

```
Awesome-Diffusion-Language-model/
│
├── README.md                        ← You are here
├── CONTRIBUTING.md
├── LICENSE
│
├── 📁 notebooks/
│   ├── 01_forward_process.ipynb
│   ├── 02_reverse_denoising.ipynb
│   ├── 03_noise_schedules.ipynb
│   ├── 04_score_field_viz.ipynb
│   ├── 05_attention_patterns.ipynb
│   ├── 06_generation_comparison.ipynb
│   ├── 07_cfg_guidance.ipynb
│   └── 08_transition_matrix_viz.ipynb
│
├── 📁 implementations/
│   ├── minimal_masked_diffusion.py
│   ├── d3pm_absorbing.py
│   ├── sedd_score_entropy.py
│   └── diffusion_lm_continuous.py
│
├── 📁 diagrams/
│   ├── architecture_overview.svg
│   ├── forward_reverse_process.svg
│   └── comparison_ar_vs_dlm.svg
│
└── 📁 assets/
    └── animations/
```

---

## 🙏 Acknowledgements

This repository is built on the brilliant work of the research community. Special recognition to:

- **Austin et al.** for D3PM, laying the discrete diffusion foundation
- **Li et al.** for Diffusion-LM, bringing controllability to text
- **Shi et al.** for MDLM, making masked diffusion practical at scale
- **Lou et al.** for SEDD, achieving autoregressive-level perplexity
- **Ho et al.** for DDPM, the image diffusion foundation that inspired all of this

---

<div align="center">

### 🌟 If this repo helped you understand diffusion language models, please star it!

*Stars help others discover this resource and motivate more content.*

<br/>

**Made with 🔥 for the open-source AI community**

*Open an [Issue](../../issues) · Start a [Discussion](../../discussions) · Submit a [PR](../../pulls)*

<br/>

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:24243e,50:302b63,100:0f0c29&height=100&section=footer" width="100%"/>

</div>
