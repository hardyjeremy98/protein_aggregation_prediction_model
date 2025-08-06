# 🧬 Protein Aggregation Prediction Model

A deep learning model that predicts protein aggregation behavior by combining protein sequence information with environmental conditions using transformer embeddings and multimodal fusion.

## 🎯 Overview

This project implements a novel approach to predict protein aggregation by integrating:
- **Protein sequence embeddings** from ESM2 transformer with learnable attention pooling
- **Environmental conditions** (temperature, pH, protein concentration) through neural networks
- **Multimodal fusion** to create unified representations for prediction

## 🏗️ Architecture

```
Protein Sequence → ESM2 → Attention Pooling → [1280 dims]
                                                    ↓
Environmental Data → MLP + BatchNorm → [16 dims]   ↓
                                                    ↓
                          Fusion (Concatenation) → [1296 dims] → 4-Block MLP → Binary Classification
```

### Key Components:

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           PROTEIN SEQUENCE INPUT                                │
│                        "DAEFRHDSGYEVHHQKLVFF..."                                │
└─────────────────────────┬───────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         ESM2 TRANSFORMER                                        │
│                    facebook/esm2_t33_650M_UR50D                                 │
│                     [batch_size, seq_len, 1280]                                 │
└─────────────────────────┬───────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                      ATTENTION POOLING                                          │
│          Learnable attention weights across sequence positions                  │
│                    [batch_size, seq_len] → [batch_size, 1280]                   │
└─────────────────────────┬───────────────────────────────────────────────────────┘
                          │
                          │         ┌─────────────────────────────────────────────┐
                          │         │      ENVIRONMENTAL CONDITIONS               │
                          │         │   Temperature (°C), pH, Concentration       │
                          │         │            [batch_size, 3]                  │
                          │         └─────────────┬───────────────────────────────┘
                          │                       │
                          │                       ▼
                          │         ┌─────────────────────────────────────────────┐
                          │         │       ENVIRONMENTAL MLP                     │
                          │         │   Linear → BatchNorm → ReLU                 │
                          │         │        [batch_size, 3] → [batch_size, 16]   │
                          │         └─────────────┬───────────────────────────────┘
                          │                       │
                          ▼                       ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                            FUSION MODULE                                        │
│                      Concatenation Strategy                                     │
│          [batch_size, 1280] + [batch_size, 16] → [batch_size, 1296]             │
└─────────────────────────┬───────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        PREDICTION HEAD                                          │
│                         4-Block MLP                                             │
│                                                                                 │
│   Block 1: Linear(1296→256) → ReLU → Dropout(0.3)                               │
│   Block 2: Linear(256→128)  → ReLU → Dropout(0.3)                               │
│   Block 3: Linear(128→64)   → ReLU → Dropout(0.3)                               │
│   Block 4: Linear(64→2)     → Logits                                            │
│                                                                                 │
│                    [batch_size, 1296] → [batch_size, 2]                         │
└─────────────────────────┬───────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          OUTPUT                                                 │
│                    Softmax → Probabilities                                      │
│              [No Aggregation, Aggregation] + Confidence                         │
└─────────────────────────────────────────────────────────────────────────────────┘
```

**Component Details:**
- **ESM2 Embeddings**: Pre-trained protein language model for sequence understanding
- **Attention Pooling**: Learnable attention mechanism to focus on important sequence positions
- **Environmental Processing**: Neural network layer for temperature, pH, and concentration features
- **Fusion Module**: Concatenation-based multimodal integration
- **Prediction Head**: 4-layer MLP with dropout for aggregation classification