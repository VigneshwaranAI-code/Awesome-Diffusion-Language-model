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

[🚀 Start Here](#-what-is-a-diffusion-language-model) · [📊 Visualizations](#-interactive-visualizations) · [📚 Papers](#-curated-paper-list) · [💻 Code](#-implementation-guides) · [🔨 Build from Scratch](#-build-a-dlm-from-scratch) · [🤝 Contribute](#-contributing)

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

## 🔨 Build a DLM from Scratch

> **Goal:** Build a working Masked Diffusion Language Model from absolute zero — no pretrained weights, no black-box libraries. Just PyTorch, math, and intuition.
>
> Every step below is a standalone, runnable file. Follow them in order and you'll have a trained DLM generating text by the end.

---

### Step 0 — Project Blueprint

Before writing a single line of code, understand what we're building and why each piece exists:

```
YOUR DLM — FULL SYSTEM MAP
══════════════════════════════════════════════════════════════════════

  RAW TEXT  →  TOKENIZER  →  DATASET  →  DATALOADER
                                               │
                                    ┌──────────▼──────────┐
                                    │   TRAINING LOOP      │
                                    │                      │
                                    │  x₀ (clean tokens)   │
                                    │       │               │
                                    │  NOISE SCHEDULE      │
                                    │  (β₁ ... β_T)        │
                                    │       │               │
                                    │  FORWARD PROCESS     │
                                    │  q(xₜ | x₀)          │
                                    │       │               │
                                    │  xₜ (noisy tokens)   │
                                    │       │               │
                                    │  DENOISER (θ)        │
                                    │  [Transformer]       │
                                    │       │               │
                                    │  x̂₀ prediction      │
                                    │       │               │
                                    │  LOSS + BACKPROP     │
                                    └──────────────────────┘
                                               │
                                    ┌──────────▼──────────┐
                                    │  SAMPLING LOOP       │
                                    │  z_T → z_{T-1} →    │
                                    │  ... → z_0 = TEXT    │
                                    └──────────────────────┘

  FILES YOU WILL BUILD:
  ├── 01_tokenizer.py         (vocab + encode/decode)
  ├── 02_noise_schedule.py    (β schedule + q_sample)
  ├── 03_forward_process.py   (forward diffusion)
  ├── 04_denoiser.py          (transformer architecture)
  ├── 05_loss.py              (denoising objective)
  ├── 06_train.py             (full training loop)
  ├── 07_sample.py            (generation / inference)
  └── 08_demo.py              (end-to-end demo)
```

---

### Step 1 — Tokenizer & Vocabulary

The tokenizer converts raw text into integer token IDs and back. We build a simple character-level tokenizer first, then upgrade to BPE.

```python
# 01_tokenizer.py
# ──────────────────────────────────────────────────────────────────
# CHARACTER-LEVEL TOKENIZER
# Simple, transparent, no external dependencies.
# Perfect for understanding the full pipeline before scaling up.
# ──────────────────────────────────────────────────────────────────

class CharTokenizer:
    """
    Maps every unique character to an integer ID.

    Special tokens:
      [PAD]  = 0   padding to fixed sequence length
      [MASK] = 1   the absorbing state in masked diffusion
      [UNK]  = 2   unknown / out-of-vocabulary character
    """

    PAD_ID  = 0
    MASK_ID = 1
    UNK_ID  = 2
    SPECIAL = ["[PAD]", "[MASK]", "[UNK]"]

    def __init__(self):
        self.char2id: dict[str, int] = {}
        self.id2char: dict[int, str] = {}

    def build_vocab(self, text: str) -> None:
        """Build vocabulary from a corpus string."""
        # Reserve IDs 0-2 for special tokens
        vocab = self.SPECIAL + sorted(set(text))
        self.char2id = {ch: i for i, ch in enumerate(vocab)}
        self.id2char = {i: ch for ch, i in self.char2id.items()}
        print(f"Vocabulary size: {len(self.char2id)}")
        print(f"  Special tokens : {self.SPECIAL}")
        print(f"  First 10 chars : {list(self.char2id.keys())[3:13]}")

    @property
    def vocab_size(self) -> int:
        return len(self.char2id)

    def encode(self, text: str, max_len: int) -> list[int]:
        """
        text  → list of token IDs, padded/truncated to max_len
        e.g.  "cat" → [7, 3, 24, 0, 0, 0, ...]
        """
        ids = [self.char2id.get(ch, self.UNK_ID) for ch in text]
        ids = ids[:max_len]                                  # truncate
        ids += [self.PAD_ID] * (max_len - len(ids))         # pad
        return ids

    def decode(self, ids: list[int]) -> str:
        """
        list of token IDs → string (skips PAD and MASK tokens)
        e.g.  [7, 3, 24, 1, 0] → "cat"
        """
        return "".join(
            self.id2char.get(i, "?")
            for i in ids
            if i not in (self.PAD_ID, self.MASK_ID)
        )


# ── Quick sanity check ────────────────────────────────────────────
if __name__ == "__main__":
    corpus = "the quick brown fox jumps over the lazy dog"
    tok    = CharTokenizer()
    tok.build_vocab(corpus)

    sample = "fox jumps"
    ids    = tok.encode(sample, max_len=16)
    back   = tok.decode(ids)

    print(f"\nOriginal : '{sample}'")
    print(f"Encoded  : {ids}")
    print(f"Decoded  : '{back}'")
    assert back == sample, "Round-trip failed!"
    print("✅ Tokenizer round-trip passed")
```

**What the vocabulary looks like:**

```
ID   Token    Role
──   ─────    ────────────────────────
0    [PAD]    padding / ignored
1    [MASK]   absorbing state (noise)
2    [UNK]    unknown character
3    ' '      space
4    'a'      regular character
5    'b'      regular character
...  ...      ...
N-1  'z'      regular character

  Total vocab size ≈ 3 + 26 + punctuation ≈ ~70 for English char-level
  For BPE (e.g. tiktoken / sentencepiece): vocab size ≈ 32,000 – 100,000
```

---

### Step 2 — Noise Schedule

The noise schedule controls **how fast information is destroyed** across timesteps `t = 1 … T`. It is the single most impactful hyperparameter in a diffusion model.

```python
# 02_noise_schedule.py
# ──────────────────────────────────────────────────────────────────
# NOISE SCHEDULE: compute β_t, ᾱ_t for every timestep
# ──────────────────────────────────────────────────────────────────

import torch
import matplotlib.pyplot as plt

def linear_schedule(T: int, β_start=1e-4, β_end=0.02) -> torch.Tensor:
    """β increases linearly: easy to understand, less optimal in practice."""
    return torch.linspace(β_start, β_end, T)

def cosine_schedule(T: int, s: float = 0.008) -> torch.Tensor:
    """
    Cosine schedule (Nichol & Dhariwal, 2021).
    Keeps signal longer early on → better gradient flow.
    """
    steps = torch.arange(T + 1, dtype=torch.float64)
    f     = torch.cos(((steps / T) + s) / (1 + s) * torch.pi / 2) ** 2
    ᾱ     = f / f[0]
    β     = torch.clamp(1 - ᾱ[1:] / ᾱ[:-1], min=1e-5, max=0.999)
    return β.float()

def sqrt_schedule(T: int) -> torch.Tensor:
    """sqrt schedule — aggressive early corruption; good for text."""
    t  = torch.linspace(0, 1, T)
    β  = torch.sqrt(t)
    β  = torch.clamp(β / β.sum() * T * 0.02, min=1e-5, max=0.999)
    return β

class NoiseSchedule:
    """
    Precomputes and caches all diffusion coefficients for fast sampling.

    Key quantities:
      β_t   : noise added at step t
      α_t   : signal retained at step t      (α_t = 1 - β_t)
      ᾱ_t   : cumulative signal from 0→t    (ᾱ_t = ∏ α_i)
      σ_t   : std dev of noise at step t     (σ_t = √(1 - ᾱ_t))
    """
    def __init__(self, T: int = 1000, schedule: str = "cosine"):
        self.T = T
        β = {"linear": linear_schedule,
             "cosine": cosine_schedule,
             "sqrt":   sqrt_schedule}[schedule](T)

        self.β  = β                                 # (T,)
        self.α  = 1.0 - β                           # (T,)
        self.ᾱ  = torch.cumprod(self.α, dim=0)      # (T,)  cumulative product
        self.σ  = torch.sqrt(1.0 - self.ᾱ)          # (T,)  noise std dev

    def mask_prob(self, t: torch.Tensor) -> torch.Tensor:
        """
        Probability of a token being masked at timestep t.
        Uses ᾱ_t: the fraction of original signal remaining.

          mask_prob(t) = 1 - ᾱ_t
          →  t=0  : mask_prob ≈ 0    (no masking, clean data)
          →  t=T  : mask_prob ≈ 1    (fully masked)
        """
        return 1.0 - self.ᾱ[t - 1]                 # t is 1-indexed

    def plot(self, save_path: str = "noise_schedule.png") -> None:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        t = range(1, self.T + 1)

        axes[0].plot(t, self.β,  color="#6c63ff"); axes[0].set_title("β_t  (noise per step)")
        axes[1].plot(t, self.ᾱ,  color="#f5a623"); axes[1].set_title("ᾱ_t  (signal remaining)")
        axes[2].plot(t, self.σ,  color="#ff6b6b"); axes[2].set_title("σ_t  (noise std dev)")

        for ax in axes:
            ax.set_xlabel("timestep t"); ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        print(f"Saved schedule plot → {save_path}")


# ── Quick check ───────────────────────────────────────────────────
if __name__ == "__main__":
    sched = NoiseSchedule(T=1000, schedule="cosine")
    for t in [1, 100, 500, 900, 1000]:
        print(f"  t={t:4d}  mask_prob={sched.mask_prob(torch.tensor(t)):.4f}")
    sched.plot()
```

**Visualizing the three schedules:**

```
Signal Remaining (ᾱ_t) across 1000 timesteps
──────────────────────────────────────────────
1.0 ┤
    │╲  Linear  ← drops fast early
0.8 ┤ ╲
    │  ╲_____
0.6 ┤        ╲____  Cosine ← slow start, smooth drop
    │              ╲____
0.4 ┤                   ╲___
    │  √t (sqrt)              ╲__  ← aggressive early
0.2 ┤╲___                         ╲___
    │     ╲___                         ╲____
0.0 ┼──────────────────────────────────────── t
    0     200    400    600    800    1000

👉 COSINE is recommended for language (signal preserved longer → better gradients)
```

---

### Step 3 — Forward Process

The forward process **applies noise** to a clean token sequence `x₀` to produce a noisy version `xₜ` at any timestep `t` in a single shot (no iterative simulation needed, thanks to the closed form).

```python
# 03_forward_process.py
# ──────────────────────────────────────────────────────────────────
# FORWARD PROCESS: q(xₜ | x₀)
# Given a clean sequence x₀, corrupt it to xₜ in one step.
# ──────────────────────────────────────────────────────────────────

import torch
from tokenizer       import CharTokenizer
from noise_schedule  import NoiseSchedule

MASK_ID = CharTokenizer.MASK_ID   # = 1


def q_sample(
    x0:    torch.Tensor,   # (B, L)  clean token IDs
    t:     torch.Tensor,   # (B,)    timestep per sample
    sched: NoiseSchedule,
) -> torch.Tensor:
    """
    Sample xₜ ~ q(xₜ | x₀) for absorbing (mask) diffusion.

    Each token independently masked with probability mask_prob(t).

    Visual:
      t=0   [ The ][ cat ][ sat ][ on ][ mat ]   ← clean
      t=200 [ The ][ cat ][MASK ][ on ][ mat ]   ← 1 token masked
      t=500 [ The ][MASK ][MASK ][ on ][MASK ]   ← ~half masked
      t=1000 [MASK][MASK ][MASK ][MASK][MASK ]   ← all masked
    """
    B, L  = x0.shape
    p_mask = sched.mask_prob(t)               # (B,)  one value per sample

    # Expand to (B, L) so each token has its own Bernoulli draw
    p_mask = p_mask.unsqueeze(1).expand(B, L)  # (B, L)

    # Draw a binary mask: 1 = mask this token, 0 = keep it
    noise_mask = torch.bernoulli(p_mask).bool()  # (B, L)

    # Apply: replace masked positions with MASK_ID
    xt = x0.clone()
    xt[noise_mask] = MASK_ID

    return xt


def visualize_forward(sentence: str, T: int = 10):
    """
    Show how a sentence degrades step-by-step.
    Great for intuition building.
    """
    tok   = CharTokenizer()
    tok.build_vocab(sentence)
    sched = NoiseSchedule(T=T, schedule="cosine")

    x0  = torch.tensor(tok.encode(sentence, max_len=len(sentence))).unsqueeze(0)  # (1, L)
    
    print(f"\nForward Process Visualization  (T={T})")
    print(f"{'t':>5}  {'Sequence':<40}  {'Masked%':>7}")
    print("─" * 58)

    for t_val in [0, *range(1, T + 1, max(1, T // 8)), T]:
        if t_val == 0:
            xt = x0.clone()
        else:
            t_tensor = torch.tensor([t_val])
            xt = q_sample(x0, t_tensor, sched)

        decoded   = tok.decode(xt[0].tolist()).ljust(40)
        n_masked  = (xt == MASK_ID).sum().item()
        pct       = 100 * n_masked / x0.shape[1]
        masked_vis = "".join(
            "▓" if tok_id == MASK_ID else "░"
            for tok_id in xt[0].tolist()
        )
        print(f"{t_val:>5}  {masked_vis:<40}  {pct:>6.1f}%")

    print("\n  ░ = original token  ▓ = masked token")


# ── Demo ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    visualize_forward("the quick brown fox jumps", T=100)
```

**Sample output:**

```
Forward Process Visualization  (T=100)
Sentence: "the quick brown fox jumps"
──────────────────────────────────────────────────────────
  t=0    ░░░░░░░░░░░░░░░░░░░░░░░░░   0.0%   the quick brown fox
  t=10   ░░░░░░░░▓░░░░░░░░▓░░░░░░░   8.0%   the qui k brow  fox
  t=25   ░░░▓░░░▓▓░░░░▓░░▓▓░░░░░▓░  28.0%   the uic  bro n f x j
  t=50   ░▓░▓░▓░▓▓░▓░░▓░░▓▓▓░░▓░▓░  52.0%   t e  i   ro    f    j
  t=75   ▓▓░▓▓▓░▓▓▓▓▓▓▓░░▓▓▓▓▓▓▓▓░  76.0%     e          o       s
  t=100  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ 100.0%   [fully masked]
```

---

### Step 4 — The Denoiser (Transformer)

The heart of the DLM. A **bidirectional Transformer** that takes a noisy token sequence `xₜ` and timestep `t`, and predicts the original clean token at every position.

```python
# 04_denoiser.py
# ──────────────────────────────────────────────────────────────────
# DENOISER ARCHITECTURE
# Bidirectional Transformer with sinusoidal time conditioning.
# Designed for clarity — every sub-module is labelled and explained.
# ──────────────────────────────────────────────────────────────────

import torch
import torch.nn as nn
import math


# ── 1. Sinusoidal Timestep Embedding ─────────────────────────────
class TimestepEmbedding(nn.Module):
    """
    Converts a scalar timestep t into a d_model-dimensional vector.
    Uses the same sin/cos encoding as positional embeddings.

    Why? The model needs to behave DIFFERENTLY at t=10 vs t=900.
    Without a time signal, it cannot distinguish heavy noise from light.

    t=10  → [0.84, 0.54, 0.91, ...]   "lightly noised"
    t=900 → [0.01, 0.99, 0.13, ...]   "heavily noised"
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.proj = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.SiLU(),
            nn.Linear(d_model * 4, d_model),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """t: (B,) integer timesteps → (B, d_model) embeddings"""
        half = self.d_model // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device) / (half - 1)
        )
        args  = t[:, None].float() * freqs[None, :]   # (B, half)
        emb   = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # (B, d_model)
        return self.proj(emb)                           # (B, d_model)


# ── 2. Positional Encoding ────────────────────────────────────────
class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding (Vaswani et al., 2017)."""
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))     # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, d_model)"""
        return self.dropout(x + self.pe[:, :x.size(1)])


# ── 3. Single Transformer Block ───────────────────────────────────
class DiffusionBlock(nn.Module):
    """
    One Transformer encoder block adapted for diffusion:
      • Bidirectional self-attention  (BERT-style, NOT causal)
      • Time conditioning via AdaLN   (Adaptive Layer Norm)
      • Standard FFN with GELU

    AdaLN injects the timestep embedding by learning to shift and scale
    the layer norm output — giving the model fine-grained time awareness.
    """
    def __init__(self, d_model: int, n_heads: int, ffn_mult: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attn    = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1   = nn.LayerNorm(d_model)
        self.norm2   = nn.LayerNorm(d_model)
        self.ffn     = nn.Sequential(
            nn.Linear(d_model, d_model * ffn_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * ffn_mult, d_model),
            nn.Dropout(dropout),
        )
        # AdaLN: learn scale (γ) and shift (β) from time embedding
        self.adaln   = nn.Linear(d_model, 2 * d_model)

    def forward(
        self,
        x:        torch.Tensor,   # (B, L, d_model)
        t_emb:    torch.Tensor,   # (B, d_model)
        key_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # ── AdaLN time conditioning ───────────────────────────────
        γ, β  = self.adaln(t_emb).chunk(2, dim=-1)      # each (B, d_model)
        γ     = γ.unsqueeze(1)                           # (B, 1, d_model)
        β     = β.unsqueeze(1)                           # (B, 1, d_model)

        # ── Self-attention block ──────────────────────────────────
        h, _ = self.attn(x, x, x, key_padding_mask=key_mask)
        x    = self.norm1(x + h) * (1 + γ) + β          # AdaLN after attention

        # ── Feed-forward block ────────────────────────────────────
        x    = self.norm2(x + self.ffn(x))

        return x


# ── 4. Full Denoiser Model ────────────────────────────────────────
class DiffusionLM(nn.Module):
    """
    Full Masked Diffusion Language Model.

    Architecture:
      Token Embedding  →  Positional Encoding
            ↓
      N × DiffusionBlock (bidirectional attn + AdaLN)
            ↓
      Linear head  →  logits over vocabulary
            ↓
      Softmax  →  P(x₀ | xₜ, t)   for each position

    Input:  noisy token IDs  (B, L)
            timesteps        (B,)
    Output: logits           (B, L, vocab_size)
    """
    def __init__(
        self,
        vocab_size: int,
        d_model:    int   = 256,
        n_layers:   int   = 6,
        n_heads:    int   = 8,
        max_len:    int   = 128,
        dropout:    float = 0.1,
    ):
        super().__init__()
        self.token_emb  = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_enc    = PositionalEncoding(d_model, max_len, dropout)
        self.time_emb   = TimestepEmbedding(d_model)
        self.blocks     = nn.ModuleList([
            DiffusionBlock(d_model, n_heads, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.norm_out   = nn.LayerNorm(d_model)
        self.head       = nn.Linear(d_model, vocab_size)

        # Weight tying (optional but common — ties input embedding to output head)
        self.head.weight = self.token_emb.weight

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        xt: torch.Tensor,          # (B, L)  noisy token IDs
        t:  torch.Tensor,          # (B,)    integer timesteps
        pad_mask: torch.Tensor | None = None,   # (B, L) True = padding
    ) -> torch.Tensor:             # (B, L, vocab_size)
        t_emb  = self.time_emb(t)                    # (B, d_model)
        x      = self.pos_enc(self.token_emb(xt))    # (B, L, d_model)

        for block in self.blocks:
            x  = block(x, t_emb, key_mask=pad_mask)

        x      = self.norm_out(x)
        logits = self.head(x)                        # (B, L, vocab_size)
        return logits


# ── Architecture Summary ──────────────────────────────────────────
if __name__ == "__main__":
    model = DiffusionLM(vocab_size=70, d_model=256, n_layers=6, n_heads=8, max_len=64)
    total = sum(p.numel() for p in model.parameters())
    print(f"\nDiffusionLM architecture:")
    print(f"  Vocab size   : 70")
    print(f"  d_model      : 256")
    print(f"  Layers       : 6")
    print(f"  Heads        : 8")
    print(f"  Parameters   : {total:,}   (~{total/1e6:.1f}M)")

    # Forward pass test
    B, L = 4, 64
    xt  = torch.randint(0, 70, (B, L))
    t   = torch.randint(1, 1001, (B,))
    out = model(xt, t)
    print(f"\n  Input  shape : {xt.shape}")
    print(f"  Output shape : {out.shape}   ← (B, L, vocab_size)")
    print("✅ Forward pass OK")
```

**Architecture at a glance:**

```
DiffusionLM — Data Flow
═══════════════════════════════════════════════════════

  xₜ: [MASK][fox ][MASK][ on ][MASK]      ← noisy tokens (B, L)
             │
  ┌──────────▼──────────┐
  │   Token Embedding   │  lookup table: token_id → vector
  │   (V  → d_model)    │
  └──────────┬──────────┘
  ┌──────────▼──────────┐
  │  Positional Encoding│  add position information
  └──────────┬──────────┘
             │
  t: [350]   │              ← timestep (B,)
  ┌──────────▼──────────┐
  │  Timestep Embedding │  t → d_model vector
  └──────────┬──────────┘
             │ t_emb injected via AdaLN in each block
  ┌──────────▼──────────┐
  │  DiffusionBlock × N │  bidirectional self-attention
  │  ┌──────────────┐   │  + time-conditioned LayerNorm
  │  │ MHA (bidir.) │   │
  │  │ AdaLN + FFN  │   │
  │  └──────────────┘   │
  └──────────┬──────────┘
  ┌──────────▼──────────┐
  │   Linear Head       │  d_model → vocab_size
  └──────────┬──────────┘
             │
  logits: P( x₀ | xₜ, t )  for every position    (B, L, V)
```

---

### Step 5 — Loss Function

The training objective is **cross-entropy denoising loss**: given a noisy sequence `xₜ`, predict the original tokens `x₀` at every position.

```python
# 05_loss.py
# ──────────────────────────────────────────────────────────────────
# LOSS FUNCTION: Denoising Cross-Entropy
# ──────────────────────────────────────────────────────────────────

import torch
import torch.nn as nn
import torch.nn.functional as F

MASK_ID = 1
PAD_ID  = 0


def denoising_loss(
    logits:    torch.Tensor,   # (B, L, V)  model output
    x0:        torch.Tensor,   # (B, L)     original clean tokens
    xt:        torch.Tensor,   # (B, L)     noisy input tokens
    t:         torch.Tensor,   # (B,)       timesteps
    mask_only: bool = True,    # only compute loss at masked positions
) -> tuple[torch.Tensor, dict]:
    """
    Denoising cross-entropy loss.

    Two modes:
      mask_only=True  → loss only at [MASK] positions (standard for absorbing diffusion)
      mask_only=False → loss at ALL positions (auxiliary signal; sometimes helps)

    Returns:
      loss   : scalar loss for backprop
      metrics: dict of logging values
    """
    B, L, V = logits.shape

    # ── Determine which positions to compute loss on ───────────────
    if mask_only:
        # Only penalize predictions at actually-masked positions
        loss_mask = (xt == MASK_ID)                       # (B, L)  True = was masked
    else:
        # Penalize everywhere except padding
        loss_mask = (x0 != PAD_ID)                        # (B, L)  True = real token

    if loss_mask.sum() == 0:
        # Edge case: no masked tokens (t ≈ 0, all clean)
        return logits.sum() * 0.0, {"loss": 0.0, "n_tokens": 0}

    # ── Cross-entropy at selected positions ────────────────────────
    #  Flatten → (B*L, V) and (B*L,) then mask
    logits_flat = logits.view(B * L, V)
    x0_flat     = x0.view(B * L)
    mask_flat   = loss_mask.view(B * L)

    # Select only the relevant positions
    logits_sel  = logits_flat[mask_flat]       # (N, V)
    targets_sel = x0_flat[mask_flat]           # (N,)

    loss        = F.cross_entropy(logits_sel, targets_sel, reduction="mean")

    # ── Compute token accuracy (for monitoring training progress) ──
    with torch.no_grad():
        preds   = logits_sel.argmax(dim=-1)
        acc     = (preds == targets_sel).float().mean().item()

    metrics = {
        "loss":        loss.item(),
        "token_acc":   acc,
        "n_masked":    mask_flat.sum().item(),
        "mean_t":      t.float().mean().item(),
    }
    return loss, metrics


# ── Intuition check ───────────────────────────────────────────────
if __name__ == "__main__":
    B, L, V = 4, 32, 70
    logits = torch.randn(B, L, V)
    x0     = torch.randint(3, V, (B, L))      # random clean tokens (no specials)
    xt     = x0.clone()
    xt[:, ::3] = MASK_ID                       # mask every 3rd position

    loss, metrics = denoising_loss(logits, x0, xt, t=torch.tensor([50, 200, 400, 700]))
    print(f"\nDenoising Loss Test")
    print(f"  Loss       : {metrics['loss']:.4f}")
    print(f"  Token Acc  : {metrics['token_acc']:.4f}   (random baseline ≈ {1/V:.4f})")
    print(f"  N Masked   : {int(metrics['n_masked'])}")
    print("✅ Loss function OK")
```

---

### Step 6 — Full Training Loop

Everything assembled into a clean, production-style training loop with logging, checkpointing, and gradient clipping.

```python
# 06_train.py
# ──────────────────────────────────────────────────────────────────
# FULL TRAINING LOOP — end to end
# ──────────────────────────────────────────────────────────────────

import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

from tokenizer        import CharTokenizer
from noise_schedule   import NoiseSchedule
from forward_process  import q_sample
from denoiser         import DiffusionLM
from loss             import denoising_loss


# ── Dataset ───────────────────────────────────────────────────────
class TextDataset(Dataset):
    """
    Splits a raw text corpus into fixed-length windows.
    Each item is a (L,) tensor of token IDs.
    """
    def __init__(self, text: str, tokenizer: CharTokenizer, seq_len: int):
        self.seq_len = seq_len
        self.ids     = tokenizer.encode(text, max_len=len(text))

    def __len__(self):
        return max(0, len(self.ids) - self.seq_len)

    def __getitem__(self, idx: int) -> torch.Tensor:
        chunk = self.ids[idx : idx + self.seq_len]
        return torch.tensor(chunk, dtype=torch.long)


# ── Training Config ───────────────────────────────────────────────
class Config:
    # Data
    seq_len    = 64
    # Model
    d_model    = 256
    n_layers   = 6
    n_heads    = 8
    # Diffusion
    T          = 1000
    schedule   = "cosine"
    # Training
    batch_size = 64
    lr         = 3e-4
    max_epochs = 50
    grad_clip  = 1.0
    log_every  = 100
    save_every = 5
    device     = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_dir   = Path("checkpoints")


def train(corpus: str, cfg: Config = Config()) -> None:
    cfg.ckpt_dir.mkdir(exist_ok=True)

    # ── Setup ──────────────────────────────────────────────────────
    print(f"Device: {cfg.device}")

    tok   = CharTokenizer()
    tok.build_vocab(corpus)

    sched = NoiseSchedule(T=cfg.T, schedule=cfg.schedule)

    model = DiffusionLM(
        vocab_size=tok.vocab_size,
        d_model=cfg.d_model,
        n_layers=cfg.n_layers,
        n_heads=cfg.n_heads,
        max_len=cfg.seq_len,
    ).to(cfg.device)

    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.max_epochs)

    dataset = TextDataset(corpus, tok, cfg.seq_len)
    loader  = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, drop_last=True)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters : {total_params:,}  (~{total_params/1e6:.1f}M)")
    print(f"Dataset size     : {len(dataset):,} sequences")
    print(f"Batches per epoch: {len(loader)}")
    print(f"\nStarting training...\n{'═'*60}")

    step = 0
    for epoch in range(1, cfg.max_epochs + 1):
        model.train()
        epoch_loss = 0.0

        for batch in loader:
            x0 = batch.to(cfg.device)                       # (B, L)

            # ── 1. Sample random timesteps ─────────────────────────
            t  = torch.randint(1, cfg.T + 1, (x0.size(0),), device=cfg.device)

            # ── 2. Forward process: corrupt x0 → xt ───────────────
            xt = q_sample(x0, t, sched)

            # ── 3. Predict x0 from xt ─────────────────────────────
            logits = model(xt.to(cfg.device), t)

            # ── 4. Compute loss ────────────────────────────────────
            loss, metrics = denoising_loss(logits, x0, xt, t)

            # ── 5. Backprop ────────────────────────────────────────
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()

            epoch_loss += metrics["loss"]
            step       += 1

            if step % cfg.log_every == 0:
                print(
                    f"  epoch {epoch:3d} | step {step:6d} | "
                    f"loss {metrics['loss']:.4f} | "
                    f"acc {metrics['token_acc']:.3f} | "
                    f"t̄ {metrics['mean_t']:.0f}"
                )

        scheduler.step()
        avg_loss = epoch_loss / len(loader)
        print(f"\nEpoch {epoch:3d} complete — avg loss: {avg_loss:.4f}  lr: {scheduler.get_last_lr()[0]:.2e}\n")

        if epoch % cfg.save_every == 0:
            ckpt = cfg.ckpt_dir / f"dlm_epoch{epoch:03d}.pt"
            torch.save({"epoch": epoch, "model": model.state_dict(),
                        "optim": optimizer.state_dict()}, ckpt)
            print(f"  ✅ Checkpoint saved → {ckpt}\n")


# ── Entry point ───────────────────────────────────────────────────
if __name__ == "__main__":
    with open("corpus.txt") as f:
        text = f.read()
    train(text)
```

---

### Step 7 — Sampling & Generation

The sampling loop runs the reverse process: start from a fully-masked sequence and iteratively denoise it over `T` steps.

```python
# 07_sample.py
# ──────────────────────────────────────────────────────────────────
# SAMPLING — reverse process: noise → text
# ──────────────────────────────────────────────────────────────────

import torch
import torch.nn.functional as F

from tokenizer       import CharTokenizer
from noise_schedule  import NoiseSchedule
from denoiser        import DiffusionLM

MASK_ID = CharTokenizer.MASK_ID


@torch.no_grad()
def sample(
    model:       DiffusionLM,
    sched:       NoiseSchedule,
    tokenizer:   CharTokenizer,
    seq_len:     int   = 64,
    n_samples:   int   = 4,
    temperature: float = 1.0,      # > 1 = more diverse, < 1 = more confident
    top_k:       int   = 0,        # 0 = disabled
    show_steps:  bool  = True,
    device:      str   = "cpu",
) -> list[str]:
    """
    Full reverse diffusion sampling loop.

    Algorithm:
      1. Initialize z_T = [MASK, MASK, ..., MASK]
      2. For t = T, T-1, ..., 1:
           a. Predict x̂₀ = model(zₜ, t)     (logits over vocabulary)
           b. Sample from predicted distribution
           c. Re-mask at rate corresponding to t-1
              → this gives z_{t-1}
      3. Return z_0 = final generated text
    """
    model.eval()
    B, L = n_samples, seq_len

    # ── Step 1: Start from fully masked sequence ───────────────────
    zt = torch.full((B, L), MASK_ID, dtype=torch.long, device=device)

    if show_steps:
        print(f"\nSampling {B} sequences of length {L}  (T={sched.T} steps)")
        print("═" * 60)

    # ── Step 2: Reverse diffusion loop ────────────────────────────
    for t_val in range(sched.T, 0, -1):
        t_tensor = torch.full((B,), t_val, dtype=torch.long, device=device)

        # ── 2a. Predict x̂₀ ────────────────────────────────────────
        logits   = model(zt, t_tensor)              # (B, L, V)
        logits   = logits / temperature

        # Optional top-k filtering for more coherent text
        if top_k > 0:
            topk_vals = logits.topk(top_k, dim=-1).values[..., -1, None]
            logits    = logits.masked_fill(logits < topk_vals, float("-inf"))

        probs    = F.softmax(logits, dim=-1)        # (B, L, V)
        x0_pred  = torch.multinomial(
            probs.view(B * L, -1), num_samples=1
        ).view(B, L)                                # (B, L)

        # ── 2b. Re-mask to level t-1 ───────────────────────────────
        if t_val > 1:
            p_mask_prev = sched.mask_prob(torch.tensor(t_val - 1))
            remask      = torch.bernoulli(
                torch.full((B, L), p_mask_prev.item(), device=device)
            ).bool()
            zt           = x0_pred.clone()
            zt[remask]   = MASK_ID
        else:
            # Final step: commit to prediction, no re-masking
            zt = x0_pred

        if show_steps and t_val % (sched.T // 8) == 0:
            n_masked = (zt == MASK_ID).float().mean().item()
            sample_0 = tokenizer.decode(zt[0].cpu().tolist())
            print(f"  t={t_val:4d}  masked={n_masked:.2f}  sample[0]: '{sample_0[:40]}'")

    # ── Step 3: Decode final sequences ────────────────────────────
    results = [tokenizer.decode(zt[i].cpu().tolist()) for i in range(B)]

    if show_steps:
        print("\n── Final Generations ──────────────────────────────────")
        for i, text in enumerate(results):
            print(f"  [{i+1}] {text}")

    return results


# ── Entry point ───────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt",        required=True,  help="Path to checkpoint .pt")
    parser.add_argument("--seq-len",     type=int, default=64)
    parser.add_argument("--n-samples",   type=int, default=4)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--steps",       type=int, default=None, help="Override T")
    args = parser.parse_args()

    # Load checkpoint
    ckpt = torch.load(args.ckpt, map_location="cpu")

    # Rebuild tokenizer (you'd normally save this alongside the model)
    tok = CharTokenizer()
    tok.build_vocab("the quick brown fox jumps over the lazy dog")

    sched = NoiseSchedule(T=args.steps or 1000, schedule="cosine")
    model = DiffusionLM(vocab_size=tok.vocab_size)
    model.load_state_dict(ckpt["model"])

    texts = sample(model, sched, tok,
                   seq_len=args.seq_len,
                   n_samples=args.n_samples,
                   temperature=args.temperature)
```

---

### Step 8 — Putting It All Together

End-to-end demo: train on a small corpus, generate text, and see the model improve.

```python
# 08_demo.py
# ──────────────────────────────────────────────────────────────────
# END-TO-END DEMO — runs in < 5 minutes on a laptop CPU
# ──────────────────────────────────────────────────────────────────

from tokenizer       import CharTokenizer
from noise_schedule  import NoiseSchedule
from denoiser        import DiffusionLM
from train           import train, Config
from sample          import sample

# ── Tiny corpus for fast demo ─────────────────────────────────────
CORPUS = """
the cat sat on the mat
the dog ran in the fog
the fox jumped over the log
the bird sang a morning song
the fish swam beneath the waves
the sun rose over the hills
the moon shone through the trees
the wind blew across the plain
""" * 200    # repeat to give the model enough data

# ── Train ─────────────────────────────────────────────────────────
cfg            = Config()
cfg.d_model    = 128       # smaller model for quick demo
cfg.n_layers   = 4
cfg.max_epochs = 30
cfg.batch_size = 32
cfg.log_every  = 50

tok = CharTokenizer()
tok.build_vocab(CORPUS)

train(CORPUS, cfg)

# ── Sample ────────────────────────────────────────────────────────
import torch
ckpt  = torch.load("checkpoints/dlm_epoch030.pt", map_location="cpu")
sched = NoiseSchedule(T=1000, schedule="cosine")
model = DiffusionLM(vocab_size=tok.vocab_size, d_model=128, n_layers=4)
model.load_state_dict(ckpt["model"])

print("\n" + "═"*60)
print("GENERATED TEXT SAMPLES")
print("═"*60)
sample(model, sched, tok, seq_len=32, n_samples=6, temperature=0.9)
```

**What to expect as training progresses:**

```
Epoch   1 — [MASK][MASK][MASK][MASK][MASK][MASK]   (random noise)
Epoch   5 — t e   at   e   at    e     (learns common chars)
Epoch  10 — the cat  at the   at       (learns frequent bigrams)
Epoch  20 — the cat sat on the mat     (learns common phrases)
Epoch  30 — the fox jumped over the log  ✅ (fluent generation)
```

---

### 📁 Complete File Structure

```
scratch_dlm/
├── 01_tokenizer.py          ← Char-level tokenizer with special tokens
├── 02_noise_schedule.py     ← β schedule, ᾱ, σ + visualization
├── 03_forward_process.py    ← q_sample + forward process vizualizer  
├── 04_denoiser.py           ← Transformer (TimestepEmb + AdaLN + MHA)
├── 05_loss.py               ← Denoising cross-entropy + token accuracy
├── 06_train.py              ← Full training loop with checkpointing
├── 07_sample.py             ← Reverse diffusion sampling
├── 08_demo.py               ← End-to-end demo script
│
├── checkpoints/             ← Saved model weights
├── corpus.txt               ← Your training text
└── noise_schedule.png       ← Auto-generated schedule plot
```

### ⚡ Quick Start

```bash
# Clone and install
git clone https://github.com/yourusername/Awesome-Diffusion-Language-model
cd Awesome-Diffusion-Language-model/scratch_dlm
pip install torch einops matplotlib

# Run the end-to-end demo
python 08_demo.py

# Or train on your own corpus
echo "Your corpus text here..." > corpus.txt
python 06_train.py

# Generate text from a checkpoint
python 07_sample.py --ckpt checkpoints/dlm_epoch030.pt --n-samples 8 --temperature 0.9
```

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
├── 📁 scratch_dlm/                      ← ⭐ Build from scratch (Step-by-step)
│   ├── 01_tokenizer.py
│   ├── 02_noise_schedule.py
│   ├── 03_forward_process.py
│   ├── 04_denoiser.py
│   ├── 05_loss.py
│   ├── 06_train.py
│   ├── 07_sample.py
│   └── 08_demo.py
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
