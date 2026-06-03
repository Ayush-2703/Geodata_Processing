"""
dl_models.py
=================
Deep Learning Model Definitions
---------------------------------------------------------------
Models implemented:
  1. SpectralCNN     — U-Net inspired CNN for spatial feature extraction
                       (paper ref: "U-Net and ResNet architectures" [3,5])
  2. TemporalLSTM    — Stacked LSTM/GRU for temporal dynamics
                       (paper ref: "LSTM and GRU variants" for time-series)
  3. SpectralTransformer — Self-attention for spectral-spatial context
                       (paper ref: "Transformer Models, self-attention")
  4. EnsembleFusion  — Multi-modal fusion of all three models
                       (paper ref: "multi-modal fusion framework")

All models use PyTorch 2.x and run on CPU (no GPU required).
"""

from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# ══════════════════════════════════════════════════════════════════════════════
# MODEL 1 — SpectralCNN  (U-Net style encoder-decoder with skip connections)
# Paper: "U-Net and ResNet architectures to extract spatial features
#         from images at various scales"
# ══════════════════════════════════════════════════════════════════════════════

class ConvBlock(nn.Module):
    """Two consecutive Conv→BN→ReLU layers (U-Net basic block)."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class SpectralCNN(nn.Module):
    """
    U-Net style encoder-decoder CNN for pixel-wise classification of
    multi-spectral satellite imagery.

    Architecture:
      Encoder: 3 levels of (ConvBlock + MaxPool)
      Bottleneck: ConvBlock at deepest level
      Decoder: 3 levels of (Upsample + skip concat + ConvBlock)
      Output: 1×1 Conv to num_classes

    Input shape:  (B, in_channels, H, W)   — satellite image patch
    Output shape: (B, num_classes, H, W)   — per-pixel class logits
    """
    def __init__(self, in_channels: int = 4, num_classes: int = 2,
                 base_filters: int = 32):
        super().__init__()

        f = base_filters   # 32

        # ── Encoder ────────────────────────────────────────────────────────
        self.enc1 = ConvBlock(in_channels, f)       # 32
        self.enc2 = ConvBlock(f,           f * 2)   # 64
        self.enc3 = ConvBlock(f * 2,       f * 4)   # 128
        self.pool = nn.MaxPool2d(2)

        # ── Bottleneck ─────────────────────────────────────────────────────
        self.bottleneck = ConvBlock(f * 4, f * 8)   # 256

        # ── Decoder ────────────────────────────────────────────────────────
        self.up3    = nn.ConvTranspose2d(f * 8, f * 4, kernel_size=2, stride=2)
        self.dec3   = ConvBlock(f * 8, f * 4)   # skip from enc3 → concat

        self.up2    = nn.ConvTranspose2d(f * 4, f * 2, kernel_size=2, stride=2)
        self.dec2   = ConvBlock(f * 4, f * 2)

        self.up1    = nn.ConvTranspose2d(f * 2, f,     kernel_size=2, stride=2)
        self.dec1   = ConvBlock(f * 2, f)

        # ── Output ─────────────────────────────────────────────────────────
        self.out_conv = nn.Conv2d(f, num_classes, kernel_size=1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))

        # Bottleneck
        b  = self.bottleneck(self.pool(e3))

        # Decoder with skip connections
        d3 = self.dec3(torch.cat([self.up3(b),  e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return self.out_conv(d1)   # (B, num_classes, H, W)

    def predict_patch(self, patch: torch.Tensor) -> torch.Tensor:
        """Convenience: softmax over class dimension → probabilities."""
        with torch.no_grad():
            logits = self.forward(patch)
        return F.softmax(logits, dim=1)


# ══════════════════════════════════════════════════════════════════════════════
# MODEL 2 — TemporalLSTM  (stacked LSTM + GRU for time-series)
# Paper: "LSTM and GRU variants to capture temporal dynamics in
#         time-series data"
# ══════════════════════════════════════════════════════════════════════════════

class TemporalLSTM(nn.Module):
    """
    Stacked bidirectional LSTM for multi-temporal spectral feature sequences.

    Input shape:  (B, T, input_size)   — T = number of time steps
    Output shape: (B, num_classes)     — final class prediction

    When only a single time step is available (T=1), this degrades
    gracefully to a deep MLP-like classifier.
    """
    def __init__(self, input_size: int = 9,   # bands + indices
                 hidden_size: int = 128,
                 num_layers: int = 2,
                 num_classes: int = 2,
                 dropout: float = 0.3,
                 bidirectional: bool = True):
        super().__init__()

        self.hidden_size   = hidden_size
        self.num_layers    = num_layers
        self.bidirectional = bidirectional
        D = 2 if bidirectional else 1

        # ── LSTM layers ───────────────────────────────────────────────────
        self.lstm = nn.LSTM(
            input_size  = input_size,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            batch_first = True,
            dropout     = dropout if num_layers > 1 else 0.0,
            bidirectional = bidirectional,
        )

        # ── GRU refinement layer ──────────────────────────────────────────
        self.gru = nn.GRU(
            input_size  = hidden_size * D,
            hidden_size = hidden_size,
            num_layers  = 1,
            batch_first = True,
        )

        # ── Classifier head ───────────────────────────────────────────────
        self.dropout  = nn.Dropout(dropout)
        self.norm     = nn.LayerNorm(hidden_size)
        self.fc       = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor,
                h0: Optional[torch.Tensor] = None,
                c0: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: (B, T, input_size)
        Returns: (B, num_classes) — logits
        """
        lstm_out, _ = self.lstm(x)           # (B, T, H*D)
        gru_out, _  = self.gru(lstm_out)     # (B, T, H)

        # Use final time step
        final  = gru_out[:, -1, :]           # (B, H)
        normed = self.norm(self.dropout(final))
        return self.fc(normed)               # (B, num_classes)

    def predict_sequence(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            logits = self.forward(x)
        return F.softmax(logits, dim=-1)


# ══════════════════════════════════════════════════════════════════════════════
# MODEL 3 — SpectralTransformer  (self-attention across spectral bands)
# Paper: "Transformer Models: Self-attention mechanisms facilitate
#         integration of contextual information across spatial and
#         temporal dimensions"
# ══════════════════════════════════════════════════════════════════════════════

class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding (Vaswani et al. 2017)."""
    def __init__(self, d_model: int, max_len: int = 64, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float()
                        * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        pe = pe.unsqueeze(0)               # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class SpectralTransformer(nn.Module):
    """
    Transformer encoder that treats each spectral band (+ derived index)
    as a token in a sequence. Self-attention learns inter-band relationships
    for classification.

    Architecture:
      — Linear projection: input_dim → d_model
      — Positional encoding
      — N × TransformerEncoderLayer (multi-head attention + FFN)
      — Global average pooling across sequence
      — Classification head

    Input shape:  (B, seq_len, input_dim)
                  seq_len = number of spectral bands/indices
                  input_dim = feature dimension per band (e.g. 1 or patch stats)
    Output shape: (B, num_classes)
    """
    def __init__(self, input_dim: int = 1,
                 seq_len: int = 9,           # 4 bands + 5 indices
                 d_model: int = 64,
                 nhead: int = 4,
                 num_encoder_layers: int = 2,
                 dim_feedforward: int = 256,
                 num_classes: int = 2,
                 dropout: float = 0.1):
        super().__init__()

        assert d_model % nhead == 0, "d_model must be divisible by nhead"

        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_enc    = PositionalEncoding(d_model, max_len=seq_len + 4,
                                             dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model         = d_model,
            nhead           = nhead,
            dim_feedforward = dim_feedforward,
            dropout         = dropout,
            activation      = "gelu",
            batch_first     = True,
            norm_first      = True,         # Pre-LN Transformer (more stable)
        )
        self.transformer = nn.TransformerEncoder(encoder_layer,
                                                  num_layers=num_encoder_layers)

        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, seq_len, input_dim)
        Returns logits: (B, num_classes)
        """
        x = self.input_proj(x)       # (B, S, d_model)
        x = self.pos_enc(x)
        x = self.transformer(x)      # (B, S, d_model)
        x = self.norm(x)
        x = x.mean(dim=1)            # global average pool over sequence
        return self.classifier(x)    # (B, num_classes)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return F.softmax(self.forward(x), dim=-1)


# ══════════════════════════════════════════════════════════════════════════════
# MODEL 4 — EnsembleFusion  (multi-modal fusion framework)
# Paper: "multi-modal fusion framework that can incorporate various
#         data sources achieves both computational efficiency and
#         analytical strength"
# ══════════════════════════════════════════════════════════════════════════════

class EnsembleFusion(nn.Module):
    """
    Learnable weighted fusion of CNN + LSTM + Transformer predictions.

    Takes softmax probability vectors from each model and combines them
    via a learnable weight vector (soft ensemble). This implements the
    paper's multi-modal fusion framework.

    Input:  three tensors of shape (B, num_classes) — model probabilities
    Output: (B, num_classes) — fused class probabilities
    """
    def __init__(self, num_models: int = 3, num_classes: int = 2):
        super().__init__()
        # Learnable per-model weights (initialised equally)
        self.weights = nn.Parameter(torch.ones(num_models) / num_models)

    def forward(self, *probs: torch.Tensor) -> torch.Tensor:
        """
        probs: list of (B, num_classes) tensors from each model
        Returns fused (B, num_classes) probabilities
        """
        w = F.softmax(self.weights, dim=0)  # ensure weights sum to 1
        fused = sum(w[i] * p for i, p in enumerate(probs))
        return fused   # already a probability distribution


# ══════════════════════════════════════════════════════════════════════════════
# EXPLAINABLE AI  (XAI) — Gradient-based saliency
# Paper: "incorporation of explainable AI components mitigates the
#         black box issues commonly linked with deep learning methods"
# ══════════════════════════════════════════════════════════════════════════════

class GradientSaliency:
    """
    Compute gradient-based saliency maps (vanilla backprop) to explain
    which input features/pixels drove the model's prediction.
    This implements the paper's Explainable AI (XAI) requirement.
    """
    def __init__(self, model: nn.Module):
        self.model = model

    def compute(self, x: torch.Tensor, target_class: int) -> torch.Tensor:
        """
        Returns saliency map of same shape as x.
        High values = input dimensions most responsible for the prediction.
        """
        x = x.clone().detach().requires_grad_(True)
        self.model.eval()
        output = self.model(x)

        if output.dim() > 2:
            # For CNN: global average pool over spatial dims
            score = output[:, target_class].mean()
        else:
            score = output[:, target_class].mean()

        self.model.zero_grad()
        score.backward()
        saliency = x.grad.data.abs()
        return saliency


# ══════════════════════════════════════════════════════════════════════════════
# QUICK SANITY CHECK
# ══════════════════════════════════════════════════════════════════════════════

def _smoke_test():
    """Verify all model shapes are correct — run on CPU."""
    B, C, H, W = 2, 4, 64, 64
    T, F_t     = 5, 9

    print("SpectralCNN ...")
    cnn    = SpectralCNN(in_channels=C, num_classes=2)
    patch  = torch.randn(B, C, H, W)
    out    = cnn(patch)
    assert out.shape == (B, 2, H, W), f"CNN shape error: {out.shape}"
    print(f"  OK → {out.shape}")

    print("TemporalLSTM ...")
    lstm   = TemporalLSTM(input_size=F_t, hidden_size=64,
                          num_layers=2, num_classes=2)
    seq    = torch.randn(B, T, F_t)
    out    = lstm(seq)
    assert out.shape == (B, 2), f"LSTM shape error: {out.shape}"
    print(f"  OK → {out.shape}")

    print("SpectralTransformer ...")
    trf    = SpectralTransformer(input_dim=1, seq_len=F_t, d_model=64,
                                  nhead=4, num_classes=2)
    tokens = torch.randn(B, F_t, 1)
    out    = trf(tokens)
    assert out.shape == (B, 2), f"Transformer shape error: {out.shape}"
    print(f"  OK → {out.shape}")

    print("EnsembleFusion ...")
    fusion = EnsembleFusion(num_models=3, num_classes=2)
    p1 = torch.softmax(torch.randn(B, 2), dim=-1)
    p2 = torch.softmax(torch.randn(B, 2), dim=-1)
    p3 = torch.softmax(torch.randn(B, 2), dim=-1)
    out = fusion(p1, p2, p3)
    assert out.shape == (B, 2)
    assert torch.allclose(out.sum(dim=-1), torch.ones(B), atol=1e-5)
    print(f"  OK → {out.shape}")

    print("GradientSaliency ...")
    xai  = GradientSaliency(cnn)
    sal  = xai.compute(patch, target_class=0)
    assert sal.shape == patch.shape
    print(f"  OK → {sal.shape}")

    print("\nAll model smoke tests passed ✓")


if __name__ == "__main__":
    _smoke_test()
