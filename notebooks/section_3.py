# %% [markdown]
# # ═══════════════════════════════════════════════════════════════════════════
# # Section 3 — HM-STT: Hierarchical Multi-Modal Spatio-Temporal Transformer
# # ═══════════════════════════════════════════════════════════════════════════
#
# ## Architecture Overview
#
# ```
# ┌─────────────────────────────────────────────────────────────────────────┐
# │                     RAKSHAK HM-STT Architecture                        │
# ├─────────────────────────────────────────────────────────────────────────┤
# │                                                                         │
# │  ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────┐ ┌──────────┐ │
# │  │ Vibration  │ │Temperature │ │   Gauge    │ │Weather │ │Mainten.  │ │
# │  │ [720, 3]   │ │ [720, 1]   │ │ [720, 1]   │ │[72, 6] │ │[16, 64]  │ │
# │  └─────┬──────┘ └─────┬──────┘ └─────┬──────┘ └───┬────┘ └────┬─────┘ │
# │        │              │              │             │           │        │
# │  ┌─────▼──────┐ ┌─────▼──────┐ ┌─────▼──────┐     │           │        │
# │  │  TCN Enc.  │ │  TCN Enc.  │ │  TCN Enc.  │     │           │        │
# │  │ 5 dilated  │ │ 5 dilated  │ │ 5 dilated  │     │           │        │
# │  │ conv layers│ │ conv layers│ │ conv layers│     │           │        │
# │  └─────┬──────┘ └─────┬──────┘ └─────┬──────┘     │           │        │
# │        │              │              │             │           │        │
# │        ▼              ▼              ▼             ▼           ▼        │
# │  ┌───────────────────────────────────────────────────────────────────┐  │
# │  │           STAGE 1: Per-Modality Encoders → [B, T', 128]         │  │
# │  └────────────────────────────┬──────────────────────────────────────┘  │
# │                               │                                         │
# │  ┌────────────────────────────▼──────────────────────────────────────┐  │
# │  │  STAGE 2: Cross-Modal Fusion Transformer (6 layers, 8 heads)    │  │
# │  │    ┌─────────────────────────────────────────────────────────┐   │  │
# │  │    │ Temporal Self-Attn → Cross-Modal Attn → FFN  (×6)      │   │  │
# │  │    └─────────────────────────────────────────────────────────┘   │  │
# │  │    + Learned Positional Encoding                                │  │
# │  └────────────────────────────┬──────────────────────────────────────┘  │
# │                               │                                         │
# │  ┌────────────────────────────▼──────────────────────────────────────┐  │
# │  │  STAGE 3: Spatial GAT (3 layers, 4 heads)                       │  │
# │  │    GATConv → GATConv → GATConv over track topology graph        │  │
# │  └────────────────────────────┬──────────────────────────────────────┘  │
# │                               │                                         │
# │  ┌────────────────────────────▼──────────────────────────────────────┐  │
# │  │  STAGE 4: Bidirectional LSTM (2 layers, hidden=256)             │  │
# │  │    Forward + Backward → Concat → Linear(512→256)                │  │
# │  └────────────────────────────┬──────────────────────────────────────┘  │
# │                               │                                         │
# │  ┌────────────────────────────▼──────────────────────────────────────┐  │
# │  │  STAGE 5: Multi-Task Prediction Heads (24h, 48h, 72h)          │  │
# │  │    ┌──────────────┬──────────────┬────────────────┐             │  │
# │  │    │ P(failure)   │ P(category)  │ Time-to-Failure│             │  │
# │  │    │ sigmoid [B,1]│ softmax [B,8]│ ReLU    [B,1]  │             │  │
# │  │    └──────────────┴──────────────┴────────────────┘             │  │
# │  └────────────────────────────┬──────────────────────────────────────┘  │
# │                               │                                         │
# │  ┌────────────────────────────▼──────────────────────────────────────┐  │
# │  │  STAGE 6: Uncertainty Quantification                            │  │
# │  │    MC Dropout (T=50 passes) + Deep Ensemble (N=5 models)        │  │
# │  └──────────────────────────────────────────────────────────────────┘  │
# │                                                                         │
# │  Loss: Multi-Task with Kendall & Gal uncertainty weighting             │
# │        FocalLoss(γ=2) + CrossEntropy + HuberLoss                       │
# └─────────────────────────────────────────────────────────────────────────┘
# ```

# %%
# ═══════════════════════════════════════════════════════════════════════════
# Cell 3.2 — Stage 1: Temporal Convolutional Network
# ═══════════════════════════════════════════════════════════════════════════
# TCN encoders process each sensor modality independently using dilated
# causal convolutions with exponentially increasing receptive fields.

import math
import copy
import warnings
from typing import Dict, List, Tuple, Optional, Any, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
from torch.optim import AdamW
from torch.amp import autocast, GradScaler

from tqdm.auto import tqdm

# Try importing torch_geometric; provide a graceful fallback stub
try:
    from torch_geometric.nn import GATConv
    from torch_geometric.data import Data as PyGData
    HAS_PYG = True
except ImportError:
    HAS_PYG = False
    warnings.warn(
        "torch_geometric not found. SpatialGAT will use a fallback "
        "multi-head attention layer instead of GATConv."
    )

from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.calibration import calibration_curve


class TemporalConvBlock(nn.Module):
    """Single temporal convolution block with dilated causal convolution.

    Implements:
        Conv1D → WeightNorm → ReLU → Dropout →
        Conv1D → WeightNorm → ReLU → Dropout
        + Residual skip connection (with 1x1 conv if dims differ).

    Args:
        in_channels: Input feature dimension.
        out_channels: Output feature dimension.
        kernel_size: Convolution kernel size (default: 3).
        dilation: Dilation factor for causal convolution.
        dropout: Dropout probability (default: 0.1).

    Returns:
        Tensor of shape [B, out_channels, T] after forward pass.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        # Causal padding ensures output length == input length
        self.causal_padding = (kernel_size - 1) * dilation

        # First dilated causal convolution with weight normalisation
        self.conv1 = weight_norm(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size,
                padding=self.causal_padding,
                dilation=dilation,
            )
        )
        self.relu1 = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout(dropout)

        # Second dilated causal convolution with weight normalisation
        self.conv2 = weight_norm(
            nn.Conv1d(
                out_channels,
                out_channels,
                kernel_size,
                padding=self.causal_padding,
                dilation=dilation,
            )
        )
        self.relu2 = nn.ReLU(inplace=True)
        self.dropout2 = nn.Dropout(dropout)

        # 1×1 convolution for residual when dimensions differ
        self.residual_conv = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

        self.init_weights()

    def init_weights(self) -> None:
        """Initialise convolution weights with Kaiming normal."""
        nn.init.kaiming_normal_(self.conv1.weight, nonlinearity="relu")
        nn.init.kaiming_normal_(self.conv2.weight, nonlinearity="relu")
        nn.init.zeros_(self.conv1.bias)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through temporal conv block.

        Args:
            x: Input tensor of shape [B, C_in, T].

        Returns:
            Output tensor of shape [B, C_out, T].
        """
        # First convolution path
        out = self.conv1(x)  # [B, C_out, T + padding]
        # Trim future timesteps to enforce causality
        out = out[:, :, : x.size(2)]  # [B, C_out, T]
        out = self.relu1(out)
        out = self.dropout1(out)

        # Second convolution path
        out = self.conv2(out)  # [B, C_out, T + padding]
        out = out[:, :, : x.size(2)]  # [B, C_out, T]
        out = self.relu2(out)
        out = self.dropout2(out)

        # Residual connection
        residual = self.residual_conv(x)  # [B, C_out, T]
        return out + residual  # [B, C_out, T]


class ModalityTCNEncoder(nn.Module):
    """Per-modality TCN encoder with 5 dilated convolution layers.

    Processes a single sensor modality (vibration, temperature, or gauge)
    through a stack of dilated causal convolutions with exponentially
    increasing receptive field.

    Dilation factors: [1, 2, 4, 8, 16]
    Each layer has residual connections and weight normalisation.
    Final output is mean-pooled across time and projected.

    Args:
        input_channels: Number of input channels (e.g., 3 for vibration,
            1 for temperature).
        d_enc: Encoding dimension (default: 128).
        kernel_size: Conv kernel size (default: 3).
        dropout: Dropout rate (default: 0.1).
        dilation_factors: List of dilation factors for each layer
            (default: [1, 2, 4, 8, 16]).

    Returns:
        forward() → Tensor of shape [B, T, d_enc] (temporal encoded features)
            and a pooled summary [B, d_enc].
    """

    def __init__(
        self,
        input_channels: int,
        d_enc: int = 128,
        kernel_size: int = 3,
        dropout: float = 0.1,
        dilation_factors: Optional[List[int]] = None,
    ) -> None:
        super().__init__()
        if dilation_factors is None:
            dilation_factors = [1, 2, 4, 8, 16]

        layers: List[TemporalConvBlock] = []
        for i, dilation in enumerate(dilation_factors):
            in_ch = input_channels if i == 0 else d_enc
            layers.append(
                TemporalConvBlock(
                    in_channels=in_ch,
                    out_channels=d_enc,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                )
            )
        self.tcn_layers = nn.ModuleList(layers)

        # Layer normalisation on the output
        self.layer_norm = nn.LayerNorm(d_enc)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the TCN encoder.

        Args:
            x: Input tensor of shape [B, T, C_in] (batch, time, channels).

        Returns:
            temporal_features: Encoded tensor of shape [B, T, d_enc].
            pooled: Mean-pooled summary of shape [B, d_enc].
        """
        # Transpose to [B, C_in, T] for Conv1d
        out = x.transpose(1, 2)  # [B, C_in, T]

        for layer in self.tcn_layers:
            out = layer(out)  # [B, d_enc, T]

        # Transpose back to [B, T, d_enc]
        temporal_features = out.transpose(1, 2)  # [B, T, d_enc]
        temporal_features = self.layer_norm(temporal_features)  # [B, T, d_enc]

        # Global average pooling across time
        pooled = temporal_features.mean(dim=1)  # [B, d_enc]

        return temporal_features, pooled


# %%
# ═══════════════════════════════════════════════════════════════════════════
# Cell 3.3 — Modality-Specific Encoders for Non-Timeseries Inputs
# ═══════════════════════════════════════════════════════════════════════════
# Encoders for metadata (MLP), weather forecasts (lightweight transformer),
# and maintenance history (self-attention).


class MetadataEncoder(nn.Module):
    """MLP encoder for static track metadata.

    Encodes categorical and continuous metadata features through a
    3-layer MLP with ReLU and dropout.

    Architecture: Linear(meta_dim→128) → ReLU → Dropout →
                  Linear(128→128) → ReLU → Dropout → Linear(128→d_enc)

    Input shape:  [B, meta_dim] (default meta_dim=32)
    Output shape: [B, d_enc] (default d_enc=128)

    Args:
        meta_dim: Dimension of metadata input features (default: 32).
        d_enc: Output encoding dimension (default: 128).
        dropout: Dropout probability (default: 0.1).
    """

    def __init__(
        self,
        meta_dim: int = 32,
        d_enc: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(meta_dim, d_enc),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_enc, d_enc),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_enc, d_enc),
            nn.LayerNorm(d_enc),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Metadata tensor of shape [B, meta_dim].

        Returns:
            Encoded tensor of shape [B, d_enc].
        """
        return self.mlp(x)  # [B, d_enc]


class WeatherEncoder(nn.Module):
    """Lightweight transformer encoder for weather forecast sequences.

    Uses a 2-layer transformer encoder with learned positional encoding
    to process 72-hour weather forecasts (6 features per hour).

    Input shape:  [B, weather_hours=72, weather_features=6]
    Output shape: [B, d_enc=128]

    Args:
        weather_features: Number of weather feature channels (default: 6).
        weather_hours: Number of forecast hours (default: 72).
        d_enc: Encoding dimension (default: 128).
        n_heads: Number of attention heads (default: 4).
        n_layers: Number of transformer layers (default: 2).
        dropout: Dropout probability (default: 0.1).
    """

    def __init__(
        self,
        weather_features: int = 6,
        weather_hours: int = 72,
        d_enc: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(weather_features, d_enc)  # [B, 72, 6] → [B, 72, 128]

        # Learned positional encoding for weather time steps
        self.pos_embedding = nn.Parameter(
            torch.randn(1, weather_hours, d_enc) * 0.02
        )  # [1, 72, d_enc]

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_enc,
            nhead=n_heads,
            dim_feedforward=d_enc * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers
        )
        self.layer_norm = nn.LayerNorm(d_enc)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Weather forecast tensor of shape [B, 72, 6].

        Returns:
            Encoded tensor of shape [B, d_enc].
        """
        out = self.input_proj(x)  # [B, 72, d_enc]
        out = out + self.pos_embedding[:, : out.size(1), :]  # [B, 72, d_enc]
        out = self.transformer(out)  # [B, 72, d_enc]
        out = self.layer_norm(out.mean(dim=1))  # [B, d_enc] (global avg pool)
        return out


class MaintenanceHistoryEncoder(nn.Module):
    """Self-attention encoder for maintenance event history.

    Processes a sequence of maintenance events using multi-head
    self-attention to capture dependencies between past maintenance
    actions and their temporal ordering.

    Input shape:  [B, maint_events=16, maint_feat_dim=64]
    Output shape: [B, d_enc=128]

    Args:
        maint_feat_dim: Feature dimension per maintenance event (default: 64).
        maint_events: Maximum number of maintenance events (default: 16).
        d_enc: Encoding dimension (default: 128).
        n_heads: Number of attention heads (default: 4).
        dropout: Dropout probability (default: 0.1).
    """

    def __init__(
        self,
        maint_feat_dim: int = 64,
        maint_events: int = 16,
        d_enc: int = 128,
        n_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(maint_feat_dim, d_enc)  # [B, 16, 64] → [B, 16, 128]

        # Learned positional encoding for event ordering
        self.pos_embedding = nn.Parameter(
            torch.randn(1, maint_events, d_enc) * 0.02
        )  # [1, 16, d_enc]

        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_enc,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ffn = nn.Sequential(
            nn.Linear(d_enc, d_enc * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_enc * 2, d_enc),
        )
        self.norm1 = nn.LayerNorm(d_enc)
        self.norm2 = nn.LayerNorm(d_enc)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Maintenance history tensor of shape [B, 16, 64].

        Returns:
            Encoded tensor of shape [B, d_enc].
        """
        out = self.input_proj(x)  # [B, 16, d_enc]
        out = out + self.pos_embedding[:, : out.size(1), :]  # [B, 16, d_enc]

        # Self-attention with pre-norm
        residual = out
        out = self.norm1(out)  # [B, 16, d_enc]
        out, _ = self.self_attn(out, out, out)  # [B, 16, d_enc]
        out = self.dropout(out) + residual  # [B, 16, d_enc]

        # Feed-forward with pre-norm
        residual = out
        out = self.norm2(out)  # [B, 16, d_enc]
        out = self.ffn(out)  # [B, 16, d_enc]
        out = self.dropout(out) + residual  # [B, 16, d_enc]

        # Global average pool
        out = out.mean(dim=1)  # [B, d_enc]
        return out


# %%
# ═══════════════════════════════════════════════════════════════════════════
# Cell 3.4 — Stage 2: Cross-Modal Fusion Transformer
# ═══════════════════════════════════════════════════════════════════════════
# Fuses information across all modalities using a transformer architecture
# with both temporal self-attention and cross-modal attention mechanisms.
# Uses LEARNED positional encoding (not sinusoidal).


class LearnedPositionalEncoding(nn.Module):
    """Learned positional encoding for transformer inputs.

    Unlike sinusoidal encodings, these are fully learned parameters
    that adapt to the data distribution during training.

    Args:
        max_len: Maximum sequence length (default: 1024).
        d_model: Model dimension (default: 128).
    """

    def __init__(self, max_len: int = 1024, d_model: int = 128) -> None:
        super().__init__()
        self.pos_embedding = nn.Parameter(
            torch.randn(1, max_len, d_model) * 0.02
        )  # [1, max_len, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input.

        Args:
            x: Input tensor of shape [B, T, D].

        Returns:
            Tensor with positional encoding added, shape [B, T, D].
        """
        seq_len = x.size(1)
        return x + self.pos_embedding[:, :seq_len, :]  # [B, T, D]


class CrossModalFusionLayer(nn.Module):
    """Single layer of the cross-modal fusion transformer.

    Architecture:
        1. Temporal Self-Attention: attends across time steps within
           each modality's sequence.
        2. Cross-Modal Attention: attends across different modalities
           at each position, allowing information exchange.
        3. Feed-Forward Network (FFN): position-wise transformation.

    All sub-layers use pre-norm residual connections.

    Args:
        d_model: Model dimension (default: 128).
        n_heads: Number of attention heads (default: 8).
        d_ff: Feed-forward hidden dimension (default: 512).
        dropout: Dropout probability (default: 0.1).
    """

    def __init__(
        self,
        d_model: int = 128,
        n_heads: int = 8,
        d_ff: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        # Temporal self-attention (within-modality, across time)
        self.temporal_self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm_temporal = nn.LayerNorm(d_model)

        # Cross-modal attention (across modalities)
        self.cross_modal_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm_cross = nn.LayerNorm(d_model)

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.norm_ffn = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        cross_modal_context: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass through one fusion layer.

        Args:
            x: Primary modality features [B, T, D].
            cross_modal_context: Concatenated features from all other
                modalities [B, T_ctx, D].

        Returns:
            Updated features of shape [B, T, D].
        """
        # 1. Temporal self-attention (pre-norm)
        residual = x  # [B, T, D]
        x_norm = self.norm_temporal(x)  # [B, T, D]
        attn_out, _ = self.temporal_self_attn(
            x_norm, x_norm, x_norm
        )  # [B, T, D]
        x = residual + self.dropout(attn_out)  # [B, T, D]

        # 2. Cross-modal attention (pre-norm)
        residual = x  # [B, T, D]
        x_norm = self.norm_cross(x)  # [B, T, D]
        ctx_norm = self.norm_cross(cross_modal_context)  # [B, T_ctx, D]
        cross_out, _ = self.cross_modal_attn(
            x_norm, ctx_norm, ctx_norm
        )  # [B, T, D]
        x = residual + self.dropout(cross_out)  # [B, T, D]

        # 3. Feed-forward (pre-norm)
        residual = x  # [B, T, D]
        x_norm = self.norm_ffn(x)  # [B, T, D]
        ff_out = self.ffn(x_norm)  # [B, T, D]
        x = residual + ff_out  # [B, T, D]

        return x  # [B, T, D]


class CrossModalFusionTransformer(nn.Module):
    """6-layer cross-modal fusion transformer.

    Fuses temporal features from multiple modalities (vibration,
    temperature, gauge) along with static encodings (metadata,
    weather, maintenance history) using alternating temporal
    self-attention and cross-modal attention.

    Uses learned positional encodings (NOT sinusoidal).

    Args:
        d_model: Model dimension (default: 128).
        n_heads: Number of attention heads (default: 8).
        d_ff: Feed-forward dimension (default: 512).
        n_layers: Number of transformer layers (default: 6).
        dropout: Dropout rate (default: 0.1).
        max_seq_len: Maximum sequence length (default: 1024).
    """

    def __init__(
        self,
        d_model: int = 128,
        n_heads: int = 8,
        d_ff: int = 512,
        n_layers: int = 6,
        dropout: float = 0.1,
        max_seq_len: int = 1024,
    ) -> None:
        super().__init__()
        self.d_model = d_model

        # Learned positional encoding (NOT sinusoidal)
        self.pos_encoding = LearnedPositionalEncoding(
            max_len=max_seq_len, d_model=d_model
        )

        # Modality-type embeddings (learnable tokens to distinguish modalities)
        self.modality_embeddings = nn.Parameter(
            torch.randn(6, 1, d_model) * 0.02
        )  # [6_modalities, 1, d_model]

        # Stack of fusion layers
        self.layers = nn.ModuleList(
            [
                CrossModalFusionLayer(
                    d_model=d_model,
                    n_heads=n_heads,
                    d_ff=d_ff,
                    dropout=dropout,
                )
                for _ in range(n_layers)
            ]
        )

        self.final_norm = nn.LayerNorm(d_model)

        # Projection for static modalities (expand to pseudo-temporal)
        self.static_expand = nn.Linear(d_model, d_model)

    def forward(
        self,
        vib_features: torch.Tensor,
        temp_features: torch.Tensor,
        gauge_features: torch.Tensor,
        meta_features: torch.Tensor,
        weather_features: torch.Tensor,
        maint_features: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass through the cross-modal fusion transformer.

        Args:
            vib_features: Vibration TCN output [B, T, D].
            temp_features: Temperature TCN output [B, T, D].
            gauge_features: Gauge TCN output [B, T, D].
            meta_features: Metadata encoding [B, D].
            weather_features: Weather encoding [B, D].
            maint_features: Maintenance encoding [B, D].

        Returns:
            Fused representation of shape [B, T, D].
        """
        B = vib_features.size(0)
        T = vib_features.size(1)

        # Add learned positional encoding to temporal modalities
        vib_features = self.pos_encoding(vib_features)  # [B, T, D]
        temp_features = self.pos_encoding(temp_features)  # [B, T, D]
        gauge_features = self.pos_encoding(gauge_features)  # [B, T, D]

        # Add modality-type embeddings
        vib_features = vib_features + self.modality_embeddings[0]  # [B, T, D]
        temp_features = temp_features + self.modality_embeddings[1]  # [B, T, D]
        gauge_features = gauge_features + self.modality_embeddings[2]  # [B, T, D]

        # Expand static modalities to pseudo-temporal sequences (repeat T times)
        meta_exp = self.static_expand(meta_features).unsqueeze(1).expand(
            B, T, self.d_model
        )  # [B, T, D]
        meta_exp = meta_exp + self.modality_embeddings[3]  # [B, T, D]

        weather_exp = self.static_expand(weather_features).unsqueeze(1).expand(
            B, T, self.d_model
        )  # [B, T, D]
        weather_exp = weather_exp + self.modality_embeddings[4]  # [B, T, D]

        maint_exp = self.static_expand(maint_features).unsqueeze(1).expand(
            B, T, self.d_model
        )  # [B, T, D]
        maint_exp = maint_exp + self.modality_embeddings[5]  # [B, T, D]

        # Build cross-modal context by concatenating all modalities
        # Each modality attends to all others
        all_modalities = [
            vib_features,
            temp_features,
            gauge_features,
            meta_exp,
            weather_exp,
            maint_exp,
        ]  # each [B, T, D]

        # Primary: use vibration as anchor; context: all modalities concatenated
        # For efficiency, we concat all modalities along the time dim for context
        cross_ctx = torch.cat(all_modalities, dim=1)  # [B, 6*T, D]

        # Fuse: use mean of all temporal modalities as primary sequence
        fused = torch.stack(all_modalities, dim=0).mean(dim=0)  # [B, T, D]

        # Apply fusion layers
        for layer in self.layers:
            fused = layer(fused, cross_ctx)  # [B, T, D]

        fused = self.final_norm(fused)  # [B, T, D]
        return fused


# %%
# ═══════════════════════════════════════════════════════════════════════════
# Cell 3.5 — Stage 3: Spatial Graph Attention Network
# ═══════════════════════════════════════════════════════════════════════════
# Models spatial relationships between track sections using a 3-layer
# Graph Attention Network (GAT). Falls back to multi-head attention
# if torch_geometric is unavailable.


class FallbackGATLayer(nn.Module):
    """Fallback multi-head attention layer mimicking GAT behaviour.

    Used when torch_geometric is not installed. Implements a dense
    attention mechanism over node features with edge masking.

    Args:
        in_channels: Input feature dimension.
        out_channels: Output feature dimension per head.
        heads: Number of attention heads (default: 4).
        dropout: Dropout probability (default: 0.2).
        concat: Whether to concatenate heads (True) or average (False).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int = 4,
        dropout: float = 0.2,
        concat: bool = True,
    ) -> None:
        super().__init__()
        self.heads = heads
        self.out_channels = out_channels
        self.concat = concat

        self.W = nn.Linear(in_channels, heads * out_channels, bias=False)
        self.attn_src = nn.Parameter(torch.randn(1, heads, out_channels))
        self.attn_dst = nn.Parameter(torch.randn(1, heads, out_channels))
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)

        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.attn_src)
        nn.init.xavier_uniform_(self.attn_dst)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Node features [N, in_channels].
            edge_index: Edge indices [2, E].

        Returns:
            Updated node features [N, heads*out_channels] if concat,
            else [N, out_channels].
        """
        N = x.size(0)
        # Project and reshape to multi-head
        h = self.W(x).view(N, self.heads, self.out_channels)  # [N, H, C]

        # Compute attention scores
        src_score = (h * self.attn_src).sum(dim=-1)  # [N, H]
        dst_score = (h * self.attn_dst).sum(dim=-1)  # [N, H]

        # Build attention coefficient matrix (sparse → dense for simplicity)
        e_src = src_score[edge_index[0]]  # [E, H]
        e_dst = dst_score[edge_index[1]]  # [E, H]
        e = self.leaky_relu(e_src + e_dst)  # [E, H]

        # Softmax per destination node
        alpha = torch.zeros(N, N, self.heads, device=x.device)  # [N, N, H]
        alpha[edge_index[0], edge_index[1]] = e
        alpha = alpha.masked_fill(alpha == 0, float("-inf"))
        alpha = F.softmax(alpha, dim=0)  # softmax over source nodes
        alpha = alpha.masked_fill(torch.isnan(alpha), 0.0)
        alpha = self.dropout(alpha)

        # Aggregate
        # out[j] = Σ_i alpha[i,j] * h[i]
        out = torch.einsum("ijh,ihc->jhc", alpha, h)  # [N, H, C]

        if self.concat:
            return out.reshape(N, self.heads * self.out_channels)  # [N, H*C]
        else:
            return out.mean(dim=1)  # [N, C]


class SpatialGAT(nn.Module):
    """3-layer Graph Attention Network for spatial track topology.

    Uses torch_geometric GATConv layers when available, otherwise
    falls back to a dense attention mechanism.

    Input: section-level fused embeddings + adjacency from track topology.
    Output: graph-enhanced section embeddings.

    Args:
        in_channels: Input feature dimension (default: 128).
        hidden_channels: Hidden dimension (default: 128).
        out_channels: Output dimension (default: 128).
        heads: Number of attention heads (default: 4).
        num_layers: Number of GAT layers (default: 3).
        dropout: Dropout rate (default: 0.2).
    """

    def __init__(
        self,
        in_channels: int = 128,
        hidden_channels: int = 128,
        out_channels: int = 128,
        heads: int = 4,
        num_layers: int = 3,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for i in range(num_layers):
            if i == 0:
                in_ch = in_channels
            else:
                in_ch = hidden_channels * heads  # concat heads

            # Last layer: don't concatenate heads, average them
            is_last = i == num_layers - 1
            out_ch = out_channels if is_last else hidden_channels
            n_heads = 1 if is_last else heads
            concat = not is_last

            if HAS_PYG:
                self.convs.append(
                    GATConv(
                        in_channels=in_ch,
                        out_channels=out_ch,
                        heads=n_heads,
                        dropout=dropout,
                        concat=concat,
                    )
                )
            else:
                self.convs.append(
                    FallbackGATLayer(
                        in_channels=in_ch,
                        out_channels=out_ch,
                        heads=n_heads,
                        dropout=dropout,
                        concat=concat,
                    )
                )
            norm_dim = out_ch * n_heads if concat else out_ch
            self.norms.append(nn.LayerNorm(norm_dim))

        self.residual_proj = nn.Linear(in_channels, out_channels)

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass through the GAT.

        Args:
            x: Node feature matrix [N, in_channels].
            edge_index: Graph edge indices [2, num_edges].

        Returns:
            Updated node features [N, out_channels].
        """
        residual = self.residual_proj(x)  # [N, out_channels]

        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)  # [N, hidden*heads] or [N, out]
            x = self.norms[i](x)  # [N, ...]
            if i < self.num_layers - 1:
                x = F.elu(x, inplace=True)
                x = F.dropout(x, p=self.dropout, training=self.training)

        x = x + residual  # [N, out_channels]
        return x


# %%
# ═══════════════════════════════════════════════════════════════════════════
# Cell 3.6 — Stage 4: Bidirectional LSTM
# ═══════════════════════════════════════════════════════════════════════════
# Captures long-range temporal dependencies using a 2-layer bidirectional
# LSTM, then projects concatenated directions back to target dimension.


class BiLSTMSequencer(nn.Module):
    """2-layer bidirectional LSTM for temporal recurrence.

    Captures long-range temporal dependencies that the TCN + Transformer
    may not fully model. Both forward and backward hidden states are
    concatenated and projected through a linear layer.

    Forward + Backward → Concat [B, T, 2*hidden] → Linear → [B, T, 2*hidden_size]
    Final pooled output: [B, 2*hidden_size] via attention pooling.

    Args:
        input_size: Input dimension (default: 128).
        hidden_size: LSTM hidden size per direction (default: 256).
        num_layers: Number of stacked LSTM layers (default: 2).
        dropout: Dropout between LSTM layers (default: 0.3).

    Returns:
        forward() → (sequence_output [B, T, hidden_size*2],
                      pooled_output [B, hidden_size*2])
    """

    def __init__(
        self,
        input_size: int = 128,
        hidden_size: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Project concatenated forward+backward to output dim
        self.output_proj = nn.Linear(hidden_size * 2, hidden_size * 2)
        self.layer_norm = nn.LayerNorm(hidden_size * 2)

        # Attention-based pooling to get fixed-size representation
        self.attn_pool = nn.Sequential(
            nn.Linear(hidden_size * 2, 1),
        )

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through BiLSTM.

        Args:
            x: Input tensor of shape [B, T, input_size].

        Returns:
            seq_output: Sequence output [B, T, hidden_size*2].
            pooled: Attention-pooled output [B, hidden_size*2].
        """
        # BiLSTM forward pass
        lstm_out, _ = self.lstm(x)  # [B, T, 2*hidden_size]

        # Project
        seq_output = self.output_proj(lstm_out)  # [B, T, 2*hidden_size]
        seq_output = self.layer_norm(seq_output)  # [B, T, 2*hidden_size]

        # Attention pooling
        attn_weights = self.attn_pool(seq_output)  # [B, T, 1]
        attn_weights = F.softmax(attn_weights, dim=1)  # [B, T, 1]
        pooled = (seq_output * attn_weights).sum(dim=1)  # [B, 2*hidden_size]

        return seq_output, pooled


# %%
# ═══════════════════════════════════════════════════════════════════════════
# Cell 3.7 — Stage 5: Multi-Task Prediction Heads
# ═══════════════════════════════════════════════════════════════════════════
# Separate prediction heads for each time horizon (24h, 48h, 72h).
# Each head outputs failure probability, failure category, and TTF.


class PredictionHead(nn.Module):
    """Single prediction head for one time horizon.

    Architecture:
        Linear(input_dim→128) → ReLU → Dropout(0.3) →
        Linear(128→64) → ReLU → Dropout(0.3) →
        ├── P_failure: Linear(64→1) → Sigmoid  [B, 1]
        ├── P_category: Linear(64→8) (logits)  [B, 8]
        └── TTF: Linear(64→1) → ReLU (clamped) [B, 1]

    Args:
        input_dim: Input feature dimension (default: 512, from BiLSTM).
        hidden_dim: Hidden layer dimension (default: 128).
        num_categories: Number of failure categories (default: 8).
        dropout: Dropout probability (default: 0.3).

    Returns:
        forward() → dict with keys:
            'p_failure': Failure probability [B, 1] (sigmoid).
            'p_category': Failure category logits [B, 8].
            'ttf': Time to failure in hours [B, 1] (ReLU clamped).
    """

    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int = 128,
        num_categories: int = 8,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.shared_trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        # Binary failure probability head
        self.failure_head = nn.Linear(64, 1)

        # Multi-class failure category head (logits, softmax applied externally)
        self.category_head = nn.Linear(64, num_categories)

        # Time-to-failure regression head
        self.ttf_head = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass through prediction head.

        Args:
            x: Input features [B, input_dim].

        Returns:
            Dictionary with:
                'p_failure': [B, 1] sigmoid probability.
                'p_category': [B, 8] category logits.
                'ttf': [B, 1] predicted time-to-failure (hours, ≥ 0).
        """
        shared = self.shared_trunk(x)  # [B, 64]

        p_failure = torch.sigmoid(self.failure_head(shared))  # [B, 1]
        p_category = self.category_head(shared)  # [B, 8] (logits)
        ttf = F.relu(self.ttf_head(shared))  # [B, 1] (non-negative hours)

        return {
            "p_failure": p_failure,
            "p_category": p_category,
            "ttf": ttf,
        }


# %%
# ═══════════════════════════════════════════════════════════════════════════
# Cell 3.8 — Full HM-STT Model (6 Stages)
# ═══════════════════════════════════════════════════════════════════════════
# Assembles the complete Hierarchical Multi-Modal Spatio-Temporal
# Transformer for Indian Railways failure prediction.


class HMSTT(nn.Module):
    """Hierarchical Multi-Modal Spatio-Temporal Transformer.

    Complete 6-stage architecture for Indian Railways failure prediction.

    Stage 1: Per-modality TCN encoders (vibration, temperature, gauge)
             + MLP/Transformer encoders for metadata, weather, maintenance
    Stage 2: Cross-modal fusion transformer (6 layers, 8 heads)
    Stage 3: Spatial GAT over track topology (3 layers, 4 heads)
    Stage 4: Bidirectional LSTM (2 layers, hidden=256)
    Stage 5: Multi-task prediction heads (24h, 48h, 72h)
    Stage 6: Uncertainty via MC Dropout (activated at inference)

    Args:
        config: Configuration dictionary with all hyperparameters.

    Returns:
        forward() → dict with keys 'pred_24h', 'pred_48h', 'pred_72h',
        each containing {'p_failure', 'p_category', 'ttf'}.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__()
        self.config = config
        d_enc = config.get("d_enc", 128)
        d_model = config.get("d_model", 128)
        d_ff = config.get("d_ff", 512)
        n_heads = config.get("n_heads", 8)
        n_transformer_layers = config.get("n_transformer_layers", 6)
        dropout = config.get("dropout", 0.1)
        gat_heads = config.get("gat_heads", 4)
        gat_layers = config.get("gat_layers", 3)
        gat_dropout = config.get("gat_dropout", 0.2)
        lstm_hidden = config.get("lstm_hidden", 256)
        lstm_layers = config.get("lstm_layers", 2)
        lstm_dropout = config.get("lstm_dropout", 0.3)
        pred_dropout = config.get("pred_dropout", 0.3)
        num_categories = config.get("num_failure_categories", 8)

        # ──────────────────────────────────────────────────────────
        # Stage 1: Per-Modality Encoders
        # ──────────────────────────────────────────────────────────
        self.vib_encoder = ModalityTCNEncoder(
            input_channels=config.get("vib_channels", 3),
            d_enc=d_enc,
            kernel_size=config.get("tcn_kernel_size", 3),
            dropout=dropout,
            dilation_factors=config.get("tcn_dilation_factors", [1, 2, 4, 8, 16]),
        )
        self.temp_encoder = ModalityTCNEncoder(
            input_channels=config.get("temp_channels", 1),
            d_enc=d_enc,
            kernel_size=config.get("tcn_kernel_size", 3),
            dropout=dropout,
            dilation_factors=config.get("tcn_dilation_factors", [1, 2, 4, 8, 16]),
        )
        self.gauge_encoder = ModalityTCNEncoder(
            input_channels=config.get("gauge_channels", 1),
            d_enc=d_enc,
            kernel_size=config.get("tcn_kernel_size", 3),
            dropout=dropout,
            dilation_factors=config.get("tcn_dilation_factors", [1, 2, 4, 8, 16]),
        )
        self.meta_encoder = MetadataEncoder(
            meta_dim=config.get("meta_dim", 32),
            d_enc=d_enc,
            dropout=dropout,
        )
        self.weather_encoder = WeatherEncoder(
            weather_features=config.get("weather_features", 6),
            weather_hours=config.get("weather_hours", 72),
            d_enc=d_enc,
            dropout=dropout,
        )
        self.maint_encoder = MaintenanceHistoryEncoder(
            maint_feat_dim=config.get("maint_feat_dim", 64),
            maint_events=config.get("maint_events", 16),
            d_enc=d_enc,
            dropout=dropout,
        )

        # ──────────────────────────────────────────────────────────
        # Stage 2: Cross-Modal Fusion Transformer
        # ──────────────────────────────────────────────────────────
        self.fusion_transformer = CrossModalFusionTransformer(
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff,
            n_layers=n_transformer_layers,
            dropout=dropout,
        )

        # ──────────────────────────────────────────────────────────
        # Stage 3: Spatial GAT
        # ──────────────────────────────────────────────────────────
        self.spatial_gat = SpatialGAT(
            in_channels=d_model,
            hidden_channels=d_model,
            out_channels=d_model,
            heads=gat_heads,
            num_layers=gat_layers,
            dropout=gat_dropout,
        )

        # ──────────────────────────────────────────────────────────
        # Stage 4: Bidirectional LSTM
        # ──────────────────────────────────────────────────────────
        self.bilstm = BiLSTMSequencer(
            input_size=d_model,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            dropout=lstm_dropout,
        )

        # ──────────────────────────────────────────────────────────
        # Stage 5: Multi-Task Prediction Heads (24h, 48h, 72h)
        # ──────────────────────────────────────────────────────────
        pred_input_dim = lstm_hidden * 2  # BiLSTM concat: 256*2 = 512
        self.head_24h = PredictionHead(
            input_dim=pred_input_dim,
            num_categories=num_categories,
            dropout=pred_dropout,
        )
        self.head_48h = PredictionHead(
            input_dim=pred_input_dim,
            num_categories=num_categories,
            dropout=pred_dropout,
        )
        self.head_72h = PredictionHead(
            input_dim=pred_input_dim,
            num_categories=num_categories,
            dropout=pred_dropout,
        )

        # ──────────────────────────────────────────────────────────
        # Stage 6: MC Dropout flag (dropout stays active at inference)
        # ──────────────────────────────────────────────────────────
        self._mc_dropout_enabled = False

    def enable_mc_dropout(self) -> None:
        """Enable Monte Carlo Dropout for uncertainty estimation.

        Activates dropout layers during inference (eval mode) to
        produce stochastic forward passes for uncertainty quantification.
        Call this before running MC Dropout inference.
        """
        self._mc_dropout_enabled = True
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.train()

    def disable_mc_dropout(self) -> None:
        """Disable Monte Carlo Dropout (return to standard eval mode).

        Deactivates dropout during inference. Call this to revert
        the model back to deterministic inference after MC Dropout.
        """
        self._mc_dropout_enabled = False
        self.eval()

    def train(self, mode: bool = True) -> "HMSTT":
        """Override train() to respect MC Dropout state.

        Args:
            mode: If True, set to training mode. If False, set to eval
                mode, but keep dropout active if MC Dropout is enabled.

        Returns:
            Self reference for chaining.
        """
        super().train(mode)
        if not mode and self._mc_dropout_enabled:
            # In eval mode with MC Dropout: re-enable all dropouts
            for module in self.modules():
                if isinstance(module, nn.Dropout):
                    module.train()
        return self

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, Dict[str, torch.Tensor]]:
        """Forward pass through the complete HM-STT pipeline.

        Args:
            batch: Dictionary with keys matching RakshakDataset output:
                - 'vibration': [B, 720, 3]
                - 'temperature': [B, 720, 1]
                - 'gauge': [B, 720, 1]
                - 'metadata': [B, 32]
                - 'weather': [B, 72, 6]
                - 'maintenance_history': [B, 16, 64]
                - 'edge_index': [2, num_edges]
                - 'section_id': [B] (integer indices)

        Returns:
            Dictionary with keys 'pred_24h', 'pred_48h', 'pred_72h',
            each containing:
                - 'p_failure': [B, 1]
                - 'p_category': [B, 8]
                - 'ttf': [B, 1]
        """
        # ──── Stage 1: Per-Modality Encoding ────
        vib_temporal, vib_pooled = self.vib_encoder(
            batch["vibration"]
        )  # [B, T, D], [B, D]
        temp_temporal, temp_pooled = self.temp_encoder(
            batch["temperature"]
        )  # [B, T, D], [B, D]
        gauge_temporal, gauge_pooled = self.gauge_encoder(
            batch["gauge"]
        )  # [B, T, D], [B, D]

        meta_enc = self.meta_encoder(batch["metadata"])  # [B, D]
        weather_enc = self.weather_encoder(batch["weather"])  # [B, D]
        maint_enc = self.maint_encoder(
            batch["maintenance_history"]
        )  # [B, D]

        # ──── Stage 2: Cross-Modal Fusion ────
        fused = self.fusion_transformer(
            vib_features=vib_temporal,  # [B, T, D]
            temp_features=temp_temporal,  # [B, T, D]
            gauge_features=gauge_temporal,  # [B, T, D]
            meta_features=meta_enc,  # [B, D]
            weather_features=weather_enc,  # [B, D]
            maint_features=maint_enc,  # [B, D]
        )  # [B, T, D]

        # ──── Stage 3: Spatial GAT ────
        # Pool temporal features to get per-section representations
        B = fused.size(0)
        section_features = fused.mean(dim=1)  # [B, D]

        # Get edge_index (may vary per batch or be shared)
        edge_index = batch["edge_index"]  # [2, E]

        # Run GAT over section nodes
        # Handle case where edge_index has no edges gracefully
        if edge_index.numel() > 0 and edge_index.size(1) > 0:
            # Ensure edge indices are within bounds
            max_node = edge_index.max().item() + 1 if edge_index.numel() > 0 else B
            if max_node > B:
                # Pad section_features if graph is larger than batch
                padding = torch.zeros(
                    max_node - B, section_features.size(1),
                    device=section_features.device,
                    dtype=section_features.dtype,
                )
                node_features = torch.cat([section_features, padding], dim=0)
            else:
                node_features = section_features

            graph_enhanced = self.spatial_gat(
                node_features, edge_index
            )  # [N, D]
            section_features = graph_enhanced[:B]  # [B, D]

        # Expand section features back to temporal dimension for LSTM
        # Combine with fused temporal features
        section_expanded = section_features.unsqueeze(1).expand_as(
            fused
        )  # [B, T, D]
        lstm_input = fused + section_expanded  # [B, T, D]

        # ──── Stage 4: Bidirectional LSTM ────
        _, pooled_repr = self.bilstm(lstm_input)  # [B, 2*hidden_size=512]

        # ──── Stage 5: Multi-Task Prediction Heads ────
        pred_24h = self.head_24h(pooled_repr)  # dict with p_failure, p_category, ttf
        pred_48h = self.head_48h(pooled_repr)
        pred_72h = self.head_72h(pooled_repr)

        return {
            "pred_24h": pred_24h,
            "pred_48h": pred_48h,
            "pred_72h": pred_72h,
        }


# %%
# ═══════════════════════════════════════════════════════════════════════════
# Cell 3.9 — Multi-Task Loss with Uncertainty Weighting
# ═══════════════════════════════════════════════════════════════════════════
# Implements Kendall & Gal (2018) uncertainty weighting with Focal Loss
# for the binary classification task to handle class imbalance.


class FocalLoss(nn.Module):
    """Focal Loss for binary classification with class imbalance.

    Focal Loss = -α_t * (1 - p_t)^γ * log(p_t)

    Down-weights easy examples and focuses training on hard negatives.
    Particularly useful for rare failure events (3% failure rate).

    Args:
        gamma: Focusing parameter (default: 2.0). Higher values increase
            focus on hard examples.
        alpha: Balancing parameter (default: 0.25). Weight for positive class.
        reduction: Reduction method ('mean', 'sum', 'none').

    Returns:
        Focal loss scalar (if reduction='mean' or 'sum') or per-sample
        losses [B] (if reduction='none').
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: float = 0.25,
        reduction: str = "mean",
        pos_weight: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.pos_weight = pos_weight

    def forward(
        self, inputs: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute focal loss.

        Args:
            inputs: Predicted probabilities [B, 1] (after sigmoid).
            targets: Ground truth binary labels [B, 1].

        Returns:
            Focal loss value.

        Raises:
            ValueError: If reduction is not 'mean', 'sum', or 'none'.
        """
        # Clamp for numerical stability
        p = inputs.clamp(min=1e-7, max=1.0 - 1e-7)  # [B, 1]

        # Binary cross entropy
        if self.pos_weight is not None:
            bce = -self.pos_weight * targets * torch.log(p) - (1 - targets) * torch.log(1 - p)
        else:
            bce = -targets * torch.log(p) - (1 - targets) * torch.log(1 - p)

        # Focal weight
        p_t = targets * p + (1 - targets) * (1 - p)  # [B, 1]
        focal_weight = (1 - p_t) ** self.gamma  # [B, 1]

        # Alpha weight
        alpha_t = targets * self.alpha + (1 - targets) * (
            1 - self.alpha
        )  # [B, 1]

        loss = alpha_t * focal_weight * bce  # [B, 1]

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        elif self.reduction == "none":
            return loss
        else:
            raise ValueError(
                f"Invalid reduction '{self.reduction}'. "
                "Use 'mean', 'sum', or 'none'."
            )


class MultiTaskLoss(nn.Module):
    """Multi-task loss with Kendall & Gal uncertainty weighting.

    Automatically balances the three loss components (binary failure,
    category classification, time-to-failure regression) using learnable
    log-variance parameters.

    Loss = (1 / 2σ²_cls) * FocalLoss + (1 / 2σ²_cat) * CE_loss
         + (1 / 2σ²_reg) * HuberLoss
         + log(σ_cls) + log(σ_cat) + log(σ_reg)

    The learnable log-variance parameters (log_sigma²) are optimised
    jointly with the model parameters.

    Args:
        focal_gamma: Gamma parameter for Focal Loss (default: 2.0).
        focal_alpha: Alpha parameter for Focal Loss (default: 0.25).
        num_categories: Number of failure categories (default: 8).

    Returns:
        forward() → (total_loss, loss_dict) with per-component losses.
    """

    def __init__(
        self,
        focal_gamma: float = 2.0,
        focal_alpha: float = 0.25,
        num_categories: int = 8,
        pos_weight: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        # Learnable log-variance parameters (initialised to 0 → σ²=1)
        self.log_var_cls = nn.Parameter(torch.zeros(1))  # Binary classification
        self.log_var_cat = nn.Parameter(torch.zeros(1))  # Category classification
        self.log_var_reg = nn.Parameter(torch.zeros(1))  # Regression

        self.focal_loss = FocalLoss(
            gamma=focal_gamma, alpha=focal_alpha, pos_weight=pos_weight
        )
        self.ce_loss = nn.CrossEntropyLoss()
        self.huber_loss = nn.SmoothL1Loss()  # Huber loss

    def forward(
        self,
        predictions: Dict[str, Dict[str, torch.Tensor]],
        targets: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute multi-task loss with uncertainty weighting.

        Args:
            predictions: Model output dict with keys 'pred_24h',
                'pred_48h', 'pred_72h', each containing
                {'p_failure', 'p_category', 'ttf'}.
            targets: Target dict with keys:
                - 'failure_occurred': [B, 1] binary labels.
                - 'failure_category': [B, 1] category indices (0-7).
                - 'time_to_failure': [B, 1] hours.

        Returns:
            total_loss: Scalar combined loss.
            loss_dict: Dictionary of individual loss components for logging.
        """
        total_loss = torch.tensor(0.0, device=self.log_var_cls.device)
        loss_dict: Dict[str, torch.Tensor] = {}

        horizons = ["pred_24h", "pred_48h", "pred_72h"]

        for horizon in horizons:
            pred = predictions[horizon]

            # ── Binary failure classification (Focal Loss) ──
            cls_loss = self.focal_loss(
                pred["p_failure"], targets["failure_occurred"]
            )

            # ── Failure category classification (Cross-Entropy) ──
            # Only compute on samples that have failures
            failure_mask = targets["failure_occurred"].squeeze(-1) > 0.5  # [B]
            if failure_mask.any():
                cat_pred = pred["p_category"][failure_mask]  # [N_fail, 8]
                cat_target = targets["failure_category"][failure_mask].squeeze(
                    -1
                ).long()  # [N_fail]
                cat_loss = self.ce_loss(cat_pred, cat_target)
            else:
                cat_loss = torch.tensor(
                    0.0, device=self.log_var_cls.device
                )

            # ── Time-to-failure regression (Huber Loss) ──
            reg_loss = self.huber_loss(
                pred["ttf"], targets["time_to_failure"]
            )

            # ── Kendall & Gal uncertainty weighting ──
            precision_cls = torch.exp(-self.log_var_cls)  # 1/σ²
            precision_cat = torch.exp(-self.log_var_cat)
            precision_reg = torch.exp(-self.log_var_reg)

            weighted_cls = 0.5 * precision_cls * cls_loss + 0.5 * self.log_var_cls
            weighted_cat = 0.5 * precision_cat * cat_loss + 0.5 * self.log_var_cat
            weighted_reg = 0.5 * precision_reg * reg_loss + 0.5 * self.log_var_reg

            horizon_loss = weighted_cls + weighted_cat + weighted_reg
            total_loss = total_loss + horizon_loss

            # Log individual components
            loss_dict[f"{horizon}_cls"] = cls_loss.detach()
            loss_dict[f"{horizon}_cat"] = cat_loss.detach()
            loss_dict[f"{horizon}_reg"] = reg_loss.detach()
            loss_dict[f"{horizon}_total"] = horizon_loss.detach()

        # Average across horizons
        total_loss = total_loss / len(horizons)

        # Log sigma values for monitoring
        loss_dict["sigma_cls"] = torch.exp(0.5 * self.log_var_cls).detach()
        loss_dict["sigma_cat"] = torch.exp(0.5 * self.log_var_cat).detach()
        loss_dict["sigma_reg"] = torch.exp(0.5 * self.log_var_reg).detach()

        return total_loss, loss_dict


# %%
# ═══════════════════════════════════════════════════════════════════════════
# Cell 3.10 — Data Augmentation for Time Series
# ═══════════════════════════════════════════════════════════════════════════
# Applies stochastic augmentations to training batches for regularisation.


class TimeSeriesAugmentor:
    """Data augmentation for time series sensor data.

    Applies random augmentations to batched time series tensors
    during training to improve generalisation and robustness.

    Implements:
        1. Gaussian noise injection (σ=0.01)
        2. Time warping (random temporal stretching/compression)
        3. Magnitude scaling (random amplitude scaling ±10%)

    All augmentations are applied with 50% probability each.

    Args:
        noise_sigma: Standard deviation of Gaussian noise (default: 0.01).
        warp_range: Range of time warping factor (default: 0.1, meaning ±10%).
        scale_range: Range of magnitude scaling (default: 0.1, meaning ±10%).
        p: Probability of applying each augmentation (default: 0.5).
    """

    def __init__(
        self,
        noise_sigma: float = 0.01,
        warp_range: float = 0.1,
        scale_range: float = 0.1,
        p: float = 0.5,
    ) -> None:
        self.noise_sigma = noise_sigma
        self.warp_range = warp_range
        self.scale_range = scale_range
        self.p = p

    def gaussian_noise(self, x: torch.Tensor) -> torch.Tensor:
        """Add Gaussian noise to time series.

        Args:
            x: Input tensor of shape [B, T, C].

        Returns:
            Noisy tensor of same shape [B, T, C].
        """
        if torch.rand(1).item() < self.p:
            noise = torch.randn_like(x) * self.noise_sigma  # [B, T, C]
            return x + noise
        return x

    def time_warp(self, x: torch.Tensor) -> torch.Tensor:
        """Apply random temporal stretching/compression via interpolation.

        Generates a random warping factor per sample in the batch and
        resamples the time series using linear interpolation.

        Args:
            x: Input tensor of shape [B, T, C].

        Returns:
            Warped tensor of same shape [B, T, C].
        """
        if torch.rand(1).item() < self.p:
            B, T, C = x.shape
            # Random warp factor per sample: [1-range, 1+range]
            warp_factor = 1.0 + (
                torch.rand(B, device=x.device) * 2 - 1
            ) * self.warp_range  # [B]

            # Generate warped time indices
            orig_steps = torch.arange(T, dtype=torch.float32, device=x.device)  # [T]
            warped_steps = orig_steps.unsqueeze(0) * warp_factor.unsqueeze(1)  # [B, T]

            # Normalise to [-1, 1] for grid_sample
            warped_steps = (
                2.0 * warped_steps / (T - 1) - 1.0
            )  # [B, T], normalised

            # Create grid for F.grid_sample: [B, T, 1, 2] (x=time, y=0)
            grid = torch.zeros(B, T, 1, 2, device=x.device)
            grid[:, :, 0, 0] = warped_steps  # time dimension
            grid[:, :, 0, 1] = 0.0  # no spatial warping

            # Reshape x for grid_sample: [B, C, T, 1]
            x_4d = x.transpose(1, 2).unsqueeze(-1)  # [B, C, T, 1]
            warped = F.grid_sample(
                x_4d, grid, mode="bilinear", align_corners=True, padding_mode="border"
            )  # [B, C, T, 1]
            return warped.squeeze(-1).transpose(1, 2)  # [B, T, C]
        return x

    def magnitude_scale(self, x: torch.Tensor) -> torch.Tensor:
        """Apply random amplitude scaling to time series.

        Args:
            x: Input tensor of shape [B, T, C].

        Returns:
            Scaled tensor of same shape [B, T, C].
        """
        if torch.rand(1).item() < self.p:
            B = x.size(0)
            # Random scale per sample: [1-range, 1+range]
            scale = 1.0 + (
                torch.rand(B, 1, 1, device=x.device) * 2 - 1
            ) * self.scale_range  # [B, 1, 1]
            return x * scale
        return x

    def __call__(
        self, batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Apply all augmentations to a batch of data.

        Args:
            batch: Dictionary with time series tensors. Augmentations
                are applied to 'vibration', 'temperature', and 'gauge'.

        Returns:
            Augmented batch dictionary (in-place modification of tensors).
        """
        augmented = dict(batch)  # Shallow copy
        for key in ["vibration", "temperature", "gauge"]:
            if key in augmented:
                x = augmented[key]  # [B, T, C]
                x = self.gaussian_noise(x)
                x = self.time_warp(x)
                x = self.magnitude_scale(x)
                augmented[key] = x
        return augmented


# %%
# ═══════════════════════════════════════════════════════════════════════════
# Cell 3.11 — Learning Rate Scheduler with Warmup
# ═══════════════════════════════════════════════════════════════════════════
# Cosine annealing with linear warmup for stable training convergence.


class CosineAnnealingWithWarmup:
    """Cosine annealing learning rate schedule with linear warmup.

    During warmup (steps 0 to warmup_steps): LR increases linearly
    from 0 to base_lr.
    After warmup: LR follows cosine annealing from base_lr to min_lr.

    Args:
        optimizer: PyTorch optimizer instance.
        warmup_steps: Number of linear warmup steps (default: 1000).
        total_steps: Total number of training steps.
        min_lr: Minimum learning rate at end of schedule (default: 1e-6).

    Raises:
        ValueError: If warmup_steps >= total_steps.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int = 1000,
        total_steps: int = 10000,
        min_lr: float = 1e-6,
    ) -> None:
        if warmup_steps >= total_steps:
            raise ValueError(
                f"warmup_steps ({warmup_steps}) must be < total_steps ({total_steps})"
            )
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.base_lrs = [pg["lr"] for pg in optimizer.param_groups]
        self.current_step = 0

    def step(self) -> None:
        """Update learning rate based on current step.

        Applies linear warmup for the first `warmup_steps`, then
        cosine annealing for the remaining steps.
        """
        self.current_step += 1
        if self.current_step <= self.warmup_steps:
            # Linear warmup
            scale = self.current_step / max(self.warmup_steps, 1)
        else:
            # Cosine annealing
            progress = (self.current_step - self.warmup_steps) / max(
                self.total_steps - self.warmup_steps, 1
            )
            progress = min(progress, 1.0)
            scale = 0.5 * (1.0 + math.cos(math.pi * progress))

        for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            pg["lr"] = max(self.min_lr, base_lr * scale)

    def get_last_lr(self) -> List[float]:
        """Get the current learning rate for each parameter group.

        Returns:
            List of current learning rates, one per parameter group.
        """
        return [pg["lr"] for pg in self.optimizer.param_groups]

    def state_dict(self) -> Dict[str, Any]:
        """Return scheduler state for checkpointing.

        Returns:
            Dictionary containing current_step and base_lrs.
        """
        return {
            "current_step": self.current_step,
            "base_lrs": self.base_lrs,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load scheduler state from checkpoint.

        Args:
            state_dict: Dictionary containing scheduler state.
        """
        self.current_step = state_dict["current_step"]
        self.base_lrs = state_dict["base_lrs"]


# %%
# ═══════════════════════════════════════════════════════════════════════════
# Cell 3.12 — Training Loop
# ═══════════════════════════════════════════════════════════════════════════
# Full training pipeline for HM-STT with mixed precision, gradient clipping,
# augmentation, warmup scheduler, and early stopping.


def _move_batch_to_device(
    batch: Dict[str, torch.Tensor], device: torch.device
) -> Dict[str, torch.Tensor]:
    """Move all tensors in a batch dictionary to the target device.

    Args:
        batch: Dictionary of tensors from DataLoader.
        device: Target device (CPU or CUDA).

    Returns:
        New dictionary with all tensors on the target device.
    """
    moved = {}
    for key, val in batch.items():
        if isinstance(val, torch.Tensor):
            moved[key] = val.to(device, non_blocking=True)
        else:
            moved[key] = val
    return moved


def _compute_val_auroc(
    model: HMSTT,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> float:
    """Compute validation AUROC for failure prediction (24h horizon).

    Args:
        model: Trained HMSTT model in eval mode.
        val_loader: Validation data loader.
        device: Compute device.

    Returns:
        AUROC score as float. Returns 0.5 if computation fails.
    """
    model.eval()
    all_probs: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []

    with torch.no_grad():
        for batch in val_loader:
            batch = _move_batch_to_device(batch, device)
            with autocast(device_type="cuda", enabled=device.type == "cuda"):
                outputs = model(batch)

            probs = outputs["pred_24h"]["p_failure"].cpu().numpy()  # [B, 1]
            labels = batch["failure_occurred"].cpu().numpy()  # [B, 1]
            all_probs.append(probs)
            all_labels.append(labels)

    all_probs_np = np.concatenate(all_probs, axis=0).ravel()
    all_labels_np = np.concatenate(all_labels, axis=0).ravel()

    try:
        auroc = roc_auc_score(all_labels_np, all_probs_np)
    except ValueError:
        # All labels same class (can happen with small validation sets)
        auroc = 0.5
    return float(auroc)


def train_hmstt(
    model: HMSTT,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    config: Dict[str, Any],
) -> Tuple[HMSTT, Dict[str, List[float]]]:
    """Train the HM-STT failure prediction model.

    Full training pipeline with:
    - AdamW optimizer with lr=1e-4, weight_decay=0.01
    - CosineAnnealingWithWarmup scheduler
    - MultiTaskLoss with learnable sigma parameters
    - TimeSeriesAugmentor applied per batch
    - Mixed precision with autocast + GradScaler
    - Gradient clipping max_norm=1.0
    - tqdm progress bar with per-epoch metrics
    - Save best model checkpoint per horizon
    - EarlyStopping on val_loss

    Args:
        model: HMSTT model instance (on device).
        train_loader: Training data loader.
        val_loader: Validation data loader.
        config: Configuration dictionary.

    Returns:
        model: Trained model (best checkpoint loaded).
        training_history: Dictionary with keys 'train_loss', 'val_loss',
            'val_auroc', 'lr', each mapping to a list of per-epoch values.

    Raises:
        RuntimeError: If device mismatch between model and data.
    """
    device = torch.device(config.get("device", "cuda"))
    epochs = config.get("fpm_epochs", 30)
    lr = config.get("fpm_lr", 1e-4)
    weight_decay = config.get("fpm_weight_decay", 0.01)
    warmup_steps = config.get("fpm_warmup_steps", 1000)
    grad_clip = config.get("fpm_grad_clip", 1.0)
    noise_sigma = config.get("noise_sigma", 0.01)
    focal_gamma = config.get("focal_gamma", 2.0)
    checkpoint_dir = config.get("checkpoint_dir", "./checkpoints/")
    num_categories = config.get("num_failure_categories", 8)

    # Ensure checkpoint directory exists
    import os
    os.makedirs(checkpoint_dir, exist_ok=True)

    # ── Optimizer (AdamW) ──
    optimizer = AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )

    # ── Calculate pos_weight ──
    try:
        num_total = len(train_loader.dataset)
        positive_count = int(train_loader.dataset.failure_mask.sum())
        negative_count = num_total - positive_count
        pos_weight_val = negative_count / max(1, positive_count)
        pos_weight = torch.tensor([pos_weight_val], device=device)
    except AttributeError:
        pos_weight = None

    # ── Loss Function ──
    criterion = MultiTaskLoss(
        focal_gamma=focal_gamma,
        num_categories=num_categories,
        pos_weight=pos_weight,
    ).to(device)

    # ── Scheduler ──
    total_steps = len(train_loader) * epochs
    scheduler = CosineAnnealingWithWarmup(
        optimizer=optimizer,
        warmup_steps=min(warmup_steps, total_steps - 1),
        total_steps=total_steps,
    )

    # ── Mixed Precision Scaler ──
    use_amp = device.type == "cuda"
    grad_scaler = GradScaler(enabled=use_amp)

    # ── Data Augmentor ──
    augmentor = TimeSeriesAugmentor(noise_sigma=noise_sigma)

    # ── Early Stopping ──
    early_stopping = EarlyStopping(patience=5, min_delta=1e-4)

    # ── Training History ──
    history: Dict[str, List[float]] = {
        "train_loss": [],
        "val_loss": [],
        "val_auroc": [],
        "lr": [],
    }

    best_val_loss = float("inf")
    best_model_path = os.path.join(checkpoint_dir, "hmstt_best.pt")

    print(f"\n{'='*70}")
    print(f"  HM-STT Training | {epochs} epochs | device={device}")
    print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"{'='*70}\n")

    for epoch in range(epochs):
        # ──── Training Phase ────
        model.train()
        epoch_losses: List[float] = []

        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{epochs} [Train]",
            leave=False,
        )
        for batch_idx, batch in enumerate(pbar):
            batch = _move_batch_to_device(batch, device)

            # Apply data augmentation
            batch = augmentor(batch)

            optimizer.zero_grad(set_to_none=True)

            with autocast(device_type="cuda", enabled=use_amp):
                outputs = model(batch)
                targets = {
                    "failure_occurred": batch["failure_occurred"],
                    "failure_category": batch["failure_category"],
                    "time_to_failure": batch["time_to_failure"],
                }
                loss, loss_dict = criterion(outputs, targets)

            # Mixed precision backward
            grad_scaler.scale(loss).backward()

            # Gradient clipping (unscale first for correct norm)
            grad_scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=grad_clip
            )

            grad_scaler.step(optimizer)
            grad_scaler.update()

            # Scheduler step (per iteration)
            scheduler.step()

            epoch_losses.append(loss.item())
            current_lr = scheduler.get_last_lr()[0]

            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                lr=f"{current_lr:.2e}",
                σ_cls=f"{loss_dict['sigma_cls'].item():.3f}",
                σ_reg=f"{loss_dict['sigma_reg'].item():.3f}",
            )

        avg_train_loss = np.mean(epoch_losses)

        # ──── Validation Phase ────
        model.eval()
        val_losses: List[float] = []

        with torch.no_grad():
            for batch in tqdm(
                val_loader,
                desc=f"Epoch {epoch+1}/{epochs} [Val]",
                leave=False,
            ):
                batch = _move_batch_to_device(batch, device)
                with autocast(device_type="cuda", enabled=use_amp):
                    outputs = model(batch)
                    targets = {
                        "failure_occurred": batch["failure_occurred"],
                        "failure_category": batch["failure_category"],
                        "time_to_failure": batch["time_to_failure"],
                    }
                    loss, _ = criterion(outputs, targets)
                val_losses.append(loss.item())

        avg_val_loss = np.mean(val_losses)

        # Compute validation AUROC
        val_auroc = _compute_val_auroc(model, val_loader, device)

        # ──── Logging ────
        current_lr = scheduler.get_last_lr()[0]
        history["train_loss"].append(float(avg_train_loss))
        history["val_loss"].append(float(avg_val_loss))
        history["val_auroc"].append(float(val_auroc))
        history["lr"].append(float(current_lr))

        print(
            f"  Epoch {epoch+1:3d}/{epochs} │ "
            f"Train Loss: {avg_train_loss:.4f} │ "
            f"Val Loss: {avg_val_loss:.4f} │ "
            f"Val AUROC: {val_auroc:.4f} │ "
            f"LR: {current_lr:.2e}"
        )

        # ──── Checkpoint Best Model ────
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                loss=avg_val_loss,
                path=best_model_path,
            )
            print(f"    ✓ Saved best model (val_loss={avg_val_loss:.4f})")

        # ──── Early Stopping ────
        early_stopping(avg_val_loss)
        if early_stopping.early_stop:
            print(f"\n  ⚠ Early stopping triggered at epoch {epoch+1}")
            break

    # ──── Load Best Model ────
    if os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"\n  ✓ Loaded best model from epoch with val_loss={best_val_loss:.4f}")

    return model, history


# %%
# ═══════════════════════════════════════════════════════════════════════════
# Cell 3.13 — Uncertainty Wrapper (MC Dropout + Deep Ensemble)
# ═══════════════════════════════════════════════════════════════════════════
# Provides calibrated uncertainty estimates using Monte Carlo Dropout
# and Deep Ensemble methods for safety-critical predictions.


class UncertaintyWrapper:
    """Monte Carlo Dropout + Deep Ensemble uncertainty quantification.

    Provides two complementary uncertainty estimation methods:

    1. MC Dropout: T stochastic forward passes with dropout active
       during inference → mean prediction + predictive uncertainty.
    2. Deep Ensemble: N independently trained models → epistemic
       uncertainty (model disagreement) + aleatoric uncertainty
       (average per-model variance).

    This is critical for safety-related predictions in railway systems
    where the model must "know what it doesn't know".

    Args:
        device: Compute device for inference (default: 'cuda').
    """

    def __init__(self, device: str = "cuda") -> None:
        self.device = torch.device(device)

    @torch.no_grad()
    def mc_dropout_predict(
        self,
        model: HMSTT,
        batch: Dict[str, torch.Tensor],
        T: int = 50,
    ) -> Tuple[Dict[str, Dict[str, torch.Tensor]], Dict[str, Dict[str, torch.Tensor]]]:
        """Run MC Dropout inference with T stochastic forward passes.

        Enables dropout at inference time and performs T forward passes
        to estimate predictive mean and uncertainty (standard deviation).

        Args:
            model: Trained HMSTT model.
            batch: Input batch dictionary (already on device).
            T: Number of stochastic forward passes (default: 50).

        Returns:
            mean_predictions: Dict of mean predictions per horizon.
                Keys: 'pred_24h', 'pred_48h', 'pred_72h', each containing
                {'p_failure': [B,1], 'p_category': [B,8], 'ttf': [B,1]}.
            uncertainties: Dict of standard deviations per horizon.
                Same structure as mean_predictions.
        """
        model.eval()
        model.enable_mc_dropout()

        batch = _move_batch_to_device(batch, self.device)

        # Collect predictions from T stochastic passes
        all_predictions: Dict[str, Dict[str, List[torch.Tensor]]] = {
            horizon: {"p_failure": [], "p_category": [], "ttf": []}
            for horizon in ["pred_24h", "pred_48h", "pred_72h"]
        }

        for _ in range(T):
            with autocast(device_type="cuda", enabled=self.device.type == "cuda"):
                outputs = model(batch)

            for horizon in ["pred_24h", "pred_48h", "pred_72h"]:
                for key in ["p_failure", "p_category", "ttf"]:
                    all_predictions[horizon][key].append(
                        outputs[horizon][key].cpu()
                    )

        # Compute mean and std across T passes
        mean_predictions: Dict[str, Dict[str, torch.Tensor]] = {}
        uncertainties: Dict[str, Dict[str, torch.Tensor]] = {}

        for horizon in ["pred_24h", "pred_48h", "pred_72h"]:
            mean_predictions[horizon] = {}
            uncertainties[horizon] = {}
            for key in ["p_failure", "p_category", "ttf"]:
                stacked = torch.stack(
                    all_predictions[horizon][key], dim=0
                )  # [T, B, ...]
                mean_predictions[horizon][key] = stacked.mean(dim=0)  # [B, ...]
                uncertainties[horizon][key] = stacked.std(dim=0)  # [B, ...]

        model.disable_mc_dropout()
        return mean_predictions, uncertainties

    @torch.no_grad()
    def ensemble_predict(
        self,
        models: List[HMSTT],
        batch: Dict[str, torch.Tensor],
    ) -> Tuple[
        Dict[str, Dict[str, torch.Tensor]],
        Dict[str, Dict[str, torch.Tensor]],
        Dict[str, Dict[str, torch.Tensor]],
    ]:
        """Run Deep Ensemble inference across N independently trained models.

        Each model produces a prediction; the ensemble mean, epistemic
        uncertainty (inter-model variance), and aleatoric uncertainty
        (intra-model variance via MC Dropout) are computed.

        Args:
            models: List of N trained HMSTT models.
            batch: Input batch dictionary (already on device).

        Returns:
            mean_predictions: Ensemble-averaged predictions per horizon.
            epistemic_uncertainty: Standard deviation across models
                (captures model uncertainty / knowledge gaps).
            aleatoric_uncertainty: Average MC Dropout uncertainty per model
                (captures inherent data noise).
        """
        batch = _move_batch_to_device(batch, self.device)

        # Collect deterministic predictions from each model
        all_model_preds: Dict[str, Dict[str, List[torch.Tensor]]] = {
            horizon: {"p_failure": [], "p_category": [], "ttf": []}
            for horizon in ["pred_24h", "pred_48h", "pred_72h"]
        }

        # Collect per-model MC Dropout uncertainties
        all_model_aleatoric: Dict[str, Dict[str, List[torch.Tensor]]] = {
            horizon: {"p_failure": [], "p_category": [], "ttf": []}
            for horizon in ["pred_24h", "pred_48h", "pred_72h"]
        }

        for model in models:
            model.to(self.device)
            model.eval()

            # Deterministic prediction
            with autocast(device_type="cuda", enabled=self.device.type == "cuda"):
                outputs = model(batch)

            for horizon in ["pred_24h", "pred_48h", "pred_72h"]:
                for key in ["p_failure", "p_category", "ttf"]:
                    all_model_preds[horizon][key].append(
                        outputs[horizon][key].cpu()
                    )

            # MC Dropout uncertainty for aleatoric estimation (5 passes)
            model.enable_mc_dropout()
            mc_preds: Dict[str, Dict[str, List[torch.Tensor]]] = {
                h: {"p_failure": [], "p_category": [], "ttf": []}
                for h in ["pred_24h", "pred_48h", "pred_72h"]
            }
            for _ in range(5):
                with autocast(device_type="cuda", enabled=self.device.type == "cuda"):
                    mc_out = model(batch)
                for h in ["pred_24h", "pred_48h", "pred_72h"]:
                    for k in ["p_failure", "p_category", "ttf"]:
                        mc_preds[h][k].append(mc_out[h][k].cpu())

            for h in ["pred_24h", "pred_48h", "pred_72h"]:
                for k in ["p_failure", "p_category", "ttf"]:
                    stacked = torch.stack(mc_preds[h][k], dim=0)  # [5, B, ...]
                    all_model_aleatoric[h][k].append(stacked.std(dim=0))

            model.disable_mc_dropout()

        # Compute ensemble statistics
        mean_predictions: Dict[str, Dict[str, torch.Tensor]] = {}
        epistemic_uncertainty: Dict[str, Dict[str, torch.Tensor]] = {}
        aleatoric_uncertainty: Dict[str, Dict[str, torch.Tensor]] = {}

        for horizon in ["pred_24h", "pred_48h", "pred_72h"]:
            mean_predictions[horizon] = {}
            epistemic_uncertainty[horizon] = {}
            aleatoric_uncertainty[horizon] = {}
            for key in ["p_failure", "p_category", "ttf"]:
                stacked = torch.stack(
                    all_model_preds[horizon][key], dim=0
                )  # [N, B, ...]
                mean_predictions[horizon][key] = stacked.mean(dim=0)
                epistemic_uncertainty[horizon][key] = stacked.std(dim=0)

                aleatoric_stack = torch.stack(
                    all_model_aleatoric[horizon][key], dim=0
                )  # [N, B, ...]
                aleatoric_uncertainty[horizon][key] = aleatoric_stack.mean(dim=0)

        return mean_predictions, epistemic_uncertainty, aleatoric_uncertainty


# %%
# ═══════════════════════════════════════════════════════════════════════════
# Cell 3.14 — Deep Ensemble Training
# ═══════════════════════════════════════════════════════════════════════════
# Trains N independent HMSTT models with different random seeds for
# ensemble-based uncertainty quantification.


def train_deep_ensemble(
    config: Dict[str, Any],
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    num_models: int = 5,
) -> List[str]:
    """Train a deep ensemble of independently initialised HMSTT models.

    Each model is trained with a different random seed to maximise
    diversity in the ensemble, providing better epistemic uncertainty
    estimates.

    Memory is explicitly freed between model trainings to avoid OOM
    on Colab's limited GPU memory.

    Args:
        config: Configuration dictionary with all hyperparameters.
        train_loader: Training data loader.
        val_loader: Validation data loader.
        num_models: Number of models in the ensemble (default: 5).

    Returns:
        model_paths: List of file paths to saved model checkpoints,
            one per ensemble member.

    Raises:
        RuntimeError: If training fails for any ensemble member.
    """
    import os
    import gc

    device = torch.device(config.get("device", "cuda"))
    checkpoint_dir = config.get("checkpoint_dir", "./checkpoints/")
    os.makedirs(checkpoint_dir, exist_ok=True)
    base_seed = config.get("seed", 42)

    model_paths: List[str] = []

    print(f"\n{'='*70}")
    print(f"  Deep Ensemble Training | {num_models} models")
    print(f"{'='*70}")

    for i in range(num_models):
        print(f"\n{'─'*50}")
        print(f"  Training Ensemble Member {i+1}/{num_models}")
        print(f"{'─'*50}")

        # Set unique seed for this ensemble member
        member_seed = base_seed + i * 1000
        set_seed(member_seed)

        # Create fresh model instance
        model = HMSTT(config).to(device)

        # Create model-specific config with unique checkpoint path
        member_config = dict(config)
        member_checkpoint_dir = os.path.join(
            checkpoint_dir, f"ensemble_member_{i}"
        )
        os.makedirs(member_checkpoint_dir, exist_ok=True)
        member_config["checkpoint_dir"] = member_checkpoint_dir

        # Train this ensemble member
        try:
            model, _ = train_hmstt(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                config=member_config,
            )
        except RuntimeError as e:
            print(f"  ✗ Ensemble member {i+1} failed: {e}")
            # Clean up and continue
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            continue

        # Save final checkpoint
        model_path = os.path.join(
            member_checkpoint_dir, "hmstt_best.pt"
        )
        model_paths.append(model_path)
        print(f"  ✓ Saved ensemble member {i+1} → {model_path}")

        # Free GPU memory aggressively
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    print(f"\n  ✓ Ensemble training complete: {len(model_paths)}/{num_models} models")
    return model_paths


# %%
# ═══════════════════════════════════════════════════════════════════════════
# Cell 3.15 — FPM Evaluation
# ═══════════════════════════════════════════════════════════════════════════
# Comprehensive evaluation of the failure prediction model with AUROC,
# MAE, calibration curves, per-class accuracy, and visualisation.

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


def evaluate_fpm(
    model: HMSTT,
    test_loader: torch.utils.data.DataLoader,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Evaluate the HM-STT failure prediction model on the test set.

    Computes and visualises:
    - AUROC per horizon (24h, 48h, 72h) — target ≥ 0.90
    - Time-to-failure MAE in hours — target ≤ 4.2h
    - Calibration curve (reliability diagram) using sklearn
    - Per-class failure category accuracy
    - Summary table of all metrics

    All figures are saved to the configured figures directory.

    Args:
        model: Trained HMSTT model (already on device, in eval mode).
        test_loader: Test data loader.
        config: Optional configuration dictionary. If None, uses defaults.

    Returns:
        metrics: Dictionary containing all computed evaluation metrics:
            - 'auroc_24h', 'auroc_48h', 'auroc_72h': float
            - 'ttf_mae_24h', 'ttf_mae_48h', 'ttf_mae_72h': float
            - 'cat_accuracy': float (overall)
            - 'per_class_accuracy': Dict[str, float]
    """
    if config is None:
        config = {}

    device = next(model.parameters()).device
    figures_dir = config.get("figures_dir", "./figures/")
    import os
    os.makedirs(figures_dir, exist_ok=True)

    num_categories = config.get("num_failure_categories", 8)
    failure_categories = [
        "rail_fracture", "gauge_deviation", "thermal_buckling",
        "ballast_degradation", "weld_failure", "sleeper_damage",
        "drainage_failure", "subgrade_settlement",
    ]

    # ──── Collect all predictions ────
    model.eval()
    horizons = ["pred_24h", "pred_48h", "pred_72h"]

    all_data: Dict[str, Dict[str, List[np.ndarray]]] = {
        h: {
            "p_failure_pred": [],
            "p_failure_true": [],
            "p_category_pred": [],
            "p_category_true": [],
            "ttf_pred": [],
            "ttf_true": [],
        }
        for h in horizons
    }

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating FPM", leave=False):
            batch = _move_batch_to_device(batch, device)
            with autocast(device_type="cuda", enabled=device.type == "cuda"):
                outputs = model(batch)

            for h in horizons:
                all_data[h]["p_failure_pred"].append(
                    outputs[h]["p_failure"].cpu().numpy()
                )
                all_data[h]["p_failure_true"].append(
                    batch["failure_occurred"].cpu().numpy()
                )
                all_data[h]["p_category_pred"].append(
                    outputs[h]["p_category"].cpu().numpy()
                )
                all_data[h]["p_category_true"].append(
                    batch["failure_category"].cpu().numpy()
                )
                all_data[h]["ttf_pred"].append(
                    outputs[h]["ttf"].cpu().numpy()
                )
                all_data[h]["ttf_true"].append(
                    batch["time_to_failure"].cpu().numpy()
                )

    # Concatenate all batches
    for h in horizons:
        for key in all_data[h]:
            all_data[h][key] = np.concatenate(all_data[h][key], axis=0)

    # ──── Compute Metrics ────
    metrics: Dict[str, Any] = {}

    for h in horizons:
        # AUROC for binary failure prediction
        probs = all_data[h]["p_failure_pred"].ravel()
        labels = all_data[h]["p_failure_true"].ravel()
        try:
            auroc = roc_auc_score(labels, probs)
        except ValueError:
            auroc = 0.5
        metrics[f"auroc_{h.split('_')[1]}"] = auroc

        # TTF MAE (hours)
        ttf_pred = all_data[h]["ttf_pred"].ravel()
        ttf_true = all_data[h]["ttf_true"].ravel()
        ttf_mae = np.mean(np.abs(ttf_pred - ttf_true))
        metrics[f"ttf_mae_{h.split('_')[1]}"] = ttf_mae

    # Category accuracy (use 24h horizon predictions)
    cat_pred = all_data["pred_24h"]["p_category_pred"]  # [N, 8]
    cat_true = all_data["pred_24h"]["p_category_true"].ravel().astype(int)  # [N]
    cat_pred_labels = np.argmax(cat_pred, axis=1)  # [N]

    # Only evaluate on failure samples
    failure_mask = all_data["pred_24h"]["p_failure_true"].ravel() > 0.5
    if failure_mask.any():
        cat_accuracy = accuracy_score(
            cat_true[failure_mask], cat_pred_labels[failure_mask]
        )
        metrics["cat_accuracy"] = cat_accuracy

        # Per-class accuracy
        per_class_acc: Dict[str, float] = {}
        for cls_idx in range(num_categories):
            cls_mask = cat_true[failure_mask] == cls_idx
            if cls_mask.any():
                cls_correct = (
                    cat_pred_labels[failure_mask][cls_mask]
                    == cat_true[failure_mask][cls_mask]
                )
                per_class_acc[failure_categories[cls_idx]] = float(
                    cls_correct.mean()
                )
            else:
                per_class_acc[failure_categories[cls_idx]] = float("nan")
        metrics["per_class_accuracy"] = per_class_acc
    else:
        metrics["cat_accuracy"] = float("nan")
        metrics["per_class_accuracy"] = {
            cat: float("nan") for cat in failure_categories
        }

    # ──── Plot 1: AUROC per Horizon ────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    horizon_labels = ["24h", "48h", "72h"]

    for idx, (h, h_label) in enumerate(zip(horizons, horizon_labels)):
        probs = all_data[h]["p_failure_pred"].ravel()
        labels = all_data[h]["p_failure_true"].ravel()

        # ROC curve
        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(labels, probs)
        auroc_val = metrics[f"auroc_{h_label}"]

        axes[idx].plot(fpr, tpr, "b-", linewidth=2, label=f"AUROC = {auroc_val:.4f}")
        axes[idx].plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random")
        axes[idx].fill_between(fpr, tpr, alpha=0.1, color="blue")
        axes[idx].set_xlabel("False Positive Rate", fontsize=12)
        axes[idx].set_ylabel("True Positive Rate", fontsize=12)
        axes[idx].set_title(f"{h_label} Horizon ROC Curve", fontsize=14)
        axes[idx].legend(loc="lower right", fontsize=11)
        axes[idx].grid(True, alpha=0.3)
        # Target line
        axes[idx].axhline(y=0.90, color="red", linestyle=":", alpha=0.5, label="Target 0.90")

    plt.tight_layout()
    roc_path = os.path.join(figures_dir, "fpm_roc_curves.png")
    plt.savefig(roc_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  ✓ ROC curves saved → {roc_path}")

    # ──── Plot 2: Calibration Curves ────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for idx, (h, h_label) in enumerate(zip(horizons, horizon_labels)):
        probs = all_data[h]["p_failure_pred"].ravel()
        labels = all_data[h]["p_failure_true"].ravel()

        try:
            fraction_of_positives, mean_predicted_value = calibration_curve(
                labels, probs, n_bins=10, strategy="uniform"
            )
            axes[idx].plot(
                mean_predicted_value,
                fraction_of_positives,
                "s-",
                color="blue",
                linewidth=2,
                label="Model",
            )
        except ValueError:
            axes[idx].text(
                0.5, 0.5, "Insufficient data",
                ha="center", va="center", fontsize=12,
            )

        axes[idx].plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect calibration")
        axes[idx].set_xlabel("Mean Predicted Probability", fontsize=12)
        axes[idx].set_ylabel("Fraction of Positives", fontsize=12)
        axes[idx].set_title(f"{h_label} Calibration (Reliability Diagram)", fontsize=14)
        axes[idx].legend(loc="upper left", fontsize=11)
        axes[idx].grid(True, alpha=0.3)

    plt.tight_layout()
    cal_path = os.path.join(figures_dir, "fpm_calibration_curves.png")
    plt.savefig(cal_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  ✓ Calibration curves saved → {cal_path}")

    # ──── Plot 3: TTF MAE per Horizon ────
    fig, ax = plt.subplots(figsize=(8, 5))
    mae_values = [
        metrics[f"ttf_mae_{h}"] for h in horizon_labels
    ]
    colors = [
        "green" if mae <= 4.2 else "red" for mae in mae_values
    ]
    bars = ax.bar(horizon_labels, mae_values, color=colors, alpha=0.8, edgecolor="black")
    ax.axhline(y=4.2, color="red", linestyle="--", linewidth=2, label="Target ≤ 4.2h")
    ax.set_xlabel("Prediction Horizon", fontsize=12)
    ax.set_ylabel("MAE (hours)", fontsize=12)
    ax.set_title("Time-to-Failure MAE by Horizon", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis="y")
    for bar, val in zip(bars, mae_values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.1,
            f"{val:.2f}h",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=12,
        )
    plt.tight_layout()
    ttf_path = os.path.join(figures_dir, "fpm_ttf_mae.png")
    plt.savefig(ttf_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  ✓ TTF MAE plot saved → {ttf_path}")

    # ──── Plot 4: Per-Class Failure Category Accuracy ────
    if failure_mask.any():
        fig, ax = plt.subplots(figsize=(12, 5))
        pca = metrics["per_class_accuracy"]
        class_names = list(pca.keys())
        class_accs = [pca[c] for c in class_names]

        # Replace NaN with 0 for plotting
        class_accs_plot = [a if not np.isnan(a) else 0.0 for a in class_accs]

        colors_pca = [
            sns.color_palette("husl", len(class_names))[i]
            for i in range(len(class_names))
        ]
        bars = ax.barh(
            class_names, class_accs_plot, color=colors_pca, alpha=0.85, edgecolor="black"
        )
        ax.set_xlabel("Accuracy", fontsize=12)
        ax.set_title("Per-Class Failure Category Accuracy", fontsize=14)
        ax.set_xlim(0, 1.05)
        ax.grid(True, alpha=0.3, axis="x")

        for bar, acc in zip(bars, class_accs):
            label = f"{acc:.2f}" if not np.isnan(acc) else "N/A"
            ax.text(
                bar.get_width() + 0.01,
                bar.get_y() + bar.get_height() / 2,
                label,
                ha="left",
                va="center",
                fontsize=11,
            )

        plt.tight_layout()
        pca_path = os.path.join(figures_dir, "fpm_per_class_accuracy.png")
        plt.savefig(pca_path, dpi=150, bbox_inches="tight")
        plt.show()
        print(f"  ✓ Per-class accuracy plot saved → {pca_path}")

    # ──── Summary Table ────
    print(f"\n{'='*60}")
    print(f"  HM-STT Failure Prediction Model — Evaluation Summary")
    print(f"{'='*60}")
    print(f"  {'Metric':<30} {'Value':>12} {'Target':>12} {'Status':>8}")
    print(f"  {'─'*62}")

    for h_label in horizon_labels:
        auroc_val = metrics[f"auroc_{h_label}"]
        status = "✓" if auroc_val >= 0.90 else "✗"
        print(
            f"  AUROC ({h_label})                  "
            f"   {auroc_val:>8.4f}     ≥ 0.90     {status}"
        )

    for h_label in horizon_labels:
        mae_val = metrics[f"ttf_mae_{h_label}"]
        status = "✓" if mae_val <= 4.2 else "✗"
        print(
            f"  TTF MAE ({h_label})                "
            f"  {mae_val:>8.2f}h     ≤ 4.2h     {status}"
        )

    if not np.isnan(metrics.get("cat_accuracy", float("nan"))):
        print(
            f"  Category Accuracy              "
            f"   {metrics['cat_accuracy']:>8.4f}       —        —"
        )

    print(f"{'='*60}\n")

    return metrics


# %%
# ═══════════════════════════════════════════════════════════════════════════
# Cell 3.16 — Section 3 Checkpoint: Train, Evaluate, Report
# ═══════════════════════════════════════════════════════════════════════════
# Instantiate the model, run single-model training (Colab mode),
# evaluate, and print final metrics.

def run_section_3_checkpoint(
    config: Dict[str, Any],
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    device: torch.device,
    return_history: bool = False,
) -> Union[Tuple[HMSTT, Dict[str, Any]], Tuple[HMSTT, Dict[str, Any], Dict[str, List[float]]]]:
    """Section 3 checkpoint: train single HMSTT model and evaluate.

    In Colab mode, trains a single model (not the full 5-model ensemble)
    to stay within GPU memory and time constraints. The full ensemble
    can be trained by calling train_deep_ensemble() separately.

    Args:
        config: Full configuration dictionary.
        train_loader: Training data loader from Section 1.
        val_loader: Validation data loader from Section 1.
        test_loader: Test data loader from Section 1.
        device: Compute device (CPU/CUDA).

    Returns:
        model: Trained HMSTT model.
        metrics: Evaluation metrics dictionary.
        history: Returned as the third value when return_history=True.
    """
    print("\n" + "█" * 70)
    print("█  SECTION 3 — HM-STT Failure Prediction Model                      █")
    print("█" * 70)

    # Set seed for reproducibility
    set_seed(config.get("seed", 42))

    # ──── Instantiate Model ────
    print("\n  ► Instantiating HM-STT model...")
    model = HMSTT(config).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"    Total parameters:     {total_params:>12,}")
    print(f"    Trainable parameters: {trainable_params:>12,}")

    # ──── Train Single Model ────
    print("\n  ► Training HM-STT (single model, Colab mode)...")
    model, history = train_hmstt(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
    )

    # ──── Plot Training Curves ────
    import os
    figures_dir = config.get("figures_dir", "./figures/")
    os.makedirs(figures_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Loss curves
    axes[0].plot(history["train_loss"], "b-", label="Train", linewidth=2)
    axes[0].plot(history["val_loss"], "r-", label="Validation", linewidth=2)
    axes[0].set_xlabel("Epoch", fontsize=12)
    axes[0].set_ylabel("Loss", fontsize=12)
    axes[0].set_title("Training & Validation Loss", fontsize=14)
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)

    # AUROC curve
    axes[1].plot(history["val_auroc"], "g-", linewidth=2)
    axes[1].axhline(y=0.90, color="red", linestyle="--", linewidth=1.5, label="Target 0.90")
    axes[1].set_xlabel("Epoch", fontsize=12)
    axes[1].set_ylabel("Validation AUROC", fontsize=12)
    axes[1].set_title("Validation AUROC", fontsize=14)
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)

    # Learning rate schedule
    axes[2].plot(history["lr"], "m-", linewidth=2)
    axes[2].set_xlabel("Epoch", fontsize=12)
    axes[2].set_ylabel("Learning Rate", fontsize=12)
    axes[2].set_title("Learning Rate Schedule", fontsize=14)
    axes[2].set_yscale("log")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    train_fig_path = os.path.join(figures_dir, "fpm_training_curves.png")
    plt.savefig(train_fig_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  ✓ Training curves saved → {train_fig_path}")

    # ──── Evaluate on Test Set ────
    print("\n  ► Evaluating on test set...")
    metrics = evaluate_fpm(model, test_loader, config)

    # ──── MC Dropout Uncertainty Demo ────
    print("\n  ► Running MC Dropout uncertainty estimation (T=10 demo)...")
    uncertainty_wrapper = UncertaintyWrapper(
        device=str(device),
    )

    # Get a single batch for demo
    demo_batch = next(iter(test_loader))
    demo_batch = _move_batch_to_device(demo_batch, device)
    mc_preds, mc_unc = uncertainty_wrapper.mc_dropout_predict(
        model=model,
        batch=demo_batch,
        T=10,  # Reduced for demo speed
    )

    # Print uncertainty statistics
    print(f"\n  MC Dropout Uncertainty (24h, first 5 samples):")
    print(f"    {'P(failure) mean':>20} {'±std':>10} {'TTF mean':>12} {'±std':>10}")
    for i in range(min(5, mc_preds["pred_24h"]["p_failure"].size(0))):
        pf_mean = mc_preds["pred_24h"]["p_failure"][i, 0].item()
        pf_std = mc_unc["pred_24h"]["p_failure"][i, 0].item()
        ttf_mean = mc_preds["pred_24h"]["ttf"][i, 0].item()
        ttf_std = mc_unc["pred_24h"]["ttf"][i, 0].item()
        print(f"    {pf_mean:>20.4f} {pf_std:>10.4f} {ttf_mean:>12.2f}h {ttf_std:>10.2f}h")

    print("\n" + "█" * 70)
    print("█  Section 3 Complete ✓                                              █")
    print("█" * 70 + "\n")

    if return_history:
        return model, metrics, history
    return model, metrics


# ──── Execute Section 3 Checkpoint ────
# Uncomment the following line when running in the notebook:
# hmstt_model, fpm_metrics = run_section_3_checkpoint(
#     config=CONFIG,
#     train_loader=train_loader,
#     val_loader=val_loader,
#     test_loader=test_loader,
#     device=device,
# )
