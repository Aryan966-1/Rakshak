# %% [markdown]
# # 🛤️ Project Rakshak — AI-Powered Predictive Maintenance for Indian Railways
#
# ## Section 0: Environment Setup & Configuration
#
# This section initializes the complete runtime environment for the Rakshak
# predictive maintenance pipeline, including:
#
# | Step | Description |
# |------|-------------|
# | 0.1  | Title and overview |
# | 0.2  | GPU detection and device assignment |
# | 0.3  | Package installation |
# | 0.4  | All imports |
# | 0.5  | Reproducibility (seed setting) |
# | 0.6  | Google Drive mount |
# | 0.7  | Global CONFIG dict |
# | 0.8  | Mixed precision setup |
# | 0.9  | Helper utilities (EarlyStopping, checkpointing, plotting) |
# | 0.10 | Section 0 checkpoint |
#
# **Target Hardware:** Google Colab T4 GPU (15 GB VRAM)

# %% [markdown]
# ### Cell 0.2 — GPU Detection & Device Assignment

# %%
import torch
import os
import sys
import warnings

warnings.filterwarnings('ignore')

# ── GPU Detection & Assertion ──────────────────────────────────────────────────
if torch.cuda.is_available():
    device = torch.device('cuda')
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f'✅ GPU: {gpu_name} ({gpu_mem:.1f} GB)')
    print(f'   CUDA version : {torch.version.cuda}')
    print(f'   PyTorch       : {torch.__version__}')
else:
    device = torch.device('cpu')
    print('⚠️  WARNING: No GPU detected. Training will be extremely slow.')
    print('   Recommend: Runtime → Change runtime type → T4 GPU')

print(f'\n   Device selected: {device}')

# %% [markdown]
# ### Cell 0.3 — Package Installation

# %%
# ── Install all required packages ──────────────────────────────────────────────
# torch & torchvision are pre-installed in Colab; we install the extras.
import subprocess

_PACKAGES = [
    'numpy>=1.24',
    'pandas>=2.0',
    'scikit-learn>=1.3',
    'matplotlib>=3.7',
    'seaborn>=0.13',
    'tqdm>=4.66',
    'scipy>=1.11',
    'pydantic>=2.0',
    'torch-geometric>=2.4',       # for GAT / heterogeneous GNN
    'networkx>=3.1',
]

_OPTIONAL_PACKAGES = {'torch-geometric'}

if os.environ.get('RAKSHAK_SKIP_PACKAGE_INSTALL') == '1':
    print('Package installation skipped by RAKSHAK_SKIP_PACKAGE_INSTALL=1.')
    _PACKAGES = []

for _pkg in _PACKAGES:
    try:
        subprocess.check_call(
            [sys.executable, '-m', 'pip', 'install', '-q', _pkg],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except subprocess.CalledProcessError:
        _pkg_name = _pkg.split('>=', 1)[0].split('==', 1)[0]
        if _pkg_name in _OPTIONAL_PACKAGES:
            print(f'Optional package install failed: {_pkg}. Fallback code will be used.')
            continue
        raise

print('✅ All packages installed successfully.')

# %% [markdown]
# ### Cell 0.4 — All Imports

# %%
# ── Standard library ───────────────────────────────────────────────────────────
import json
import math
import copy
import time
import random
import hashlib
import pathlib
import logging
import datetime
import functools
import collections
from typing import (
    Any, Dict, List, Optional, Sequence, Tuple, Union, Callable
)
from dataclasses import dataclass, field

# ── Numerics & data ───────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
from scipy import signal as sp_signal
from scipy import stats as sp_stats

# ── Scikit-learn ──────────────────────────────────────────────────────────────
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    f1_score,
    precision_recall_curve,
    average_precision_score,
)

# ── PyTorch core ──────────────────────────────────────────────────────────────
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torch.cuda.amp import GradScaler, autocast

# ── PyTorch Geometric ─────────────────────────────────────────────────────────
try:
    import torch_geometric
    from torch_geometric.nn import GATConv, HeteroConv, Linear as PyGLinear
    from torch_geometric.data import HeteroData
    _HAS_PYG = True
    print(f'   torch_geometric : {torch_geometric.__version__}')
except ImportError:
    _HAS_PYG = False
    print('⚠️  torch_geometric not available — GAT/HGNN features disabled.')

# ── Visualisation ─────────────────────────────────────────────────────────────
import matplotlib
matplotlib.rcParams['figure.dpi'] = 120
matplotlib.rcParams['savefig.dpi'] = 150
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style='whitegrid', palette='muted', font_scale=1.05)

# ── Progress bars ─────────────────────────────────────────────────────────────
from tqdm.auto import tqdm

# ── NetworkX (graph utilities) ────────────────────────────────────────────────
import networkx as nx

print('✅ All imports completed.')

# %% [markdown]
# ### Cell 0.5 — Seed Setting for Reproducibility

# %%
def set_all_seeds(seed: int = 42) -> None:
    """Set random seeds across all libraries for reproducibility.

    Args:
        seed: Integer seed value to use across all random number generators.

    Returns:
        None

    Raises:
        TypeError: If seed is not an integer.
    """
    if not isinstance(seed, int):
        raise TypeError(f"Seed must be an integer, got {type(seed)}")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # multi-GPU safety
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f'🔒 All seeds set to {seed} (deterministic mode ON)')


# Call immediately
set_all_seeds(42)

# Compatibility alias used by downstream exported notebook sections.
set_seed = set_all_seeds

# %% [markdown]
# ### Cell 0.6 — Google Drive Mount

# %%
# ── Mount Google Drive (Colab only) ───────────────────────────────────────────
_IN_COLAB = 'google.colab' in sys.modules

if _IN_COLAB:
    from google.colab import drive
    drive.mount('/content/drive', force_remount=False)
    print('✅ Google Drive mounted at /content/drive/')
else:
    print('ℹ️  Not running in Colab — skipping Drive mount.')
    print('   Checkpoints will be saved to local ./rakshak_v1/ directory.')

# %% [markdown]
# ### Cell 0.7 — Global CONFIG Dictionary
#
# **Every** hyper-parameter and path lives here. Downstream sections import
# `CONFIG` and never hard-code magic numbers.

# %%
# Determine Colab mode (reduced dataset size for free-tier GPU)
COLAB_MODE: bool = True

CONFIG: Dict[str, Any] = {
    # ── General ────────────────────────────────────────────────────────────────
    'seed': 42,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'dtype': torch.float32,
    'colab_mode': True,

    # ── Data ───────────────────────────────────────────────────────────────────
    'num_sections': 50 if COLAB_MODE else 500,
    'num_years': 1 if COLAB_MODE else 5,
    'seq_len': 720,
    'num_stations': 12,
    'failure_rate': 0.18,
    'num_failure_categories': 8,
    'time_window_hrs': 720,
    'train_ratio': 0.6,
    'val_ratio': 0.2,
    'test_ratio': 0.2,
    'batch_size': 32,

    # ── Vibration ──────────────────────────────────────────────────────────────
    'vib_channels': 3,
    # ── Temperature ────────────────────────────────────────────────────────────
    'temp_channels': 1,
    # ── Gauge ──────────────────────────────────────────────────────────────────
    'gauge_channels': 1,
    # ── Metadata ───────────────────────────────────────────────────────────────
    'meta_dim': 32,
    # ── Weather ────────────────────────────────────────────────────────────────
    'weather_hours': 72,
    'weather_features': 6,
    # ── Maintenance history ────────────────────────────────────────────────────
    'maint_events': 16,
    'maint_feat_dim': 64,

    # ── ADE (Anomaly Detection Engine) ─────────────────────────────────────────
    'ade_if_trees': 200,
    'ade_if_contamination': 0.03,
    'ade_vae_latent_dim': 32,
    'ade_vae_beta': 0.5,
    'ade_vae_epochs': 50,
    'ade_vae_lr': 1e-3,
    'ade_vae_patience': 10,
    'ade_target_f1': 0.92,

    # ── HM-STT (Failure Prediction Model) ─────────────────────────────────────
    'tcn_dilation_factors': [1, 2, 4, 8, 16],
    'tcn_kernel_size': 3,
    'd_enc': 128,
    'd_model': 128,
    'd_ff': 512,
    'n_heads': 8,
    'n_transformer_layers': 6,
    'dropout': 0.1,
    'gat_heads': 4,
    'gat_layers': 3,
    'gat_dropout': 0.2,
    'lstm_hidden': 256,
    'lstm_layers': 2,
    'lstm_dropout': 0.3,
    'pred_dropout': 0.3,
    'fpm_lr': 1e-4,
    'fpm_weight_decay': 0.01,
    'fpm_warmup_steps': 1000,
    'fpm_epochs': 30 if COLAB_MODE else 100,
    'fpm_grad_clip': 1.0,
    'focal_gamma': 2.0,
    'mc_dropout_passes': 50,
    'ensemble_size': 5,
    'noise_sigma': 0.01,

    # ── HGNN (Root-Cause Analysis) ─────────────────────────────────────────────
    'hgnn_layers': 4,
    'hgnn_hidden': 128,
    'hgnn_lr': 1e-3,
    'hgnn_epochs': 50,
    'num_node_types': 4,
    'num_relation_types': 4,

    # ── Paths ──────────────────────────────────────────────────────────────────
    'drive_path': '/content/drive/MyDrive/rakshak_v1/',
    'checkpoint_dir': '/content/drive/MyDrive/rakshak_v1/checkpoints/',
    'figures_dir': '/content/drive/MyDrive/rakshak_v1/figures/',
}

# ── Explicit Training Modes ───────────────────────────────────────────────────
TRAINING_MODE = "mini"

if TRAINING_MODE == "smoke":
    CONFIG["num_sections"] = 6
    CONFIG["batch_size"] = 2
    CONFIG["ade_vae_epochs"] = 1
    CONFIG["fpm_epochs"] = 1
    CONFIG["hgnn_epochs"] = 1
elif TRAINING_MODE == "mini":
    CONFIG["num_sections"] = 100
    CONFIG["batch_size"] = 8
    CONFIG["ade_vae_epochs"] = 10
    CONFIG["fpm_epochs"] = 20
    CONFIG["hgnn_epochs"] = 20
elif TRAINING_MODE == "full":
    CONFIG["num_sections"] = 50
    CONFIG["batch_size"] = 32
    CONFIG["ade_vae_epochs"] = 50
    CONFIG["fpm_epochs"] = 100
    CONFIG["hgnn_epochs"] = 50

# ── Ensure directories exist ──────────────────────────────────────────────────
if _IN_COLAB:
    for _dir_key in ('drive_path', 'checkpoint_dir', 'figures_dir'):
        os.makedirs(CONFIG[_dir_key], exist_ok=True)
else:
    # Local fallback paths
    CONFIG['drive_path'] = './rakshak_v1/'
    CONFIG['checkpoint_dir'] = './rakshak_v1/checkpoints/'
    CONFIG['figures_dir'] = './rakshak_v1/figures/'
    for _dir_key in ('drive_path', 'checkpoint_dir', 'figures_dir'):
        os.makedirs(CONFIG[_dir_key], exist_ok=True)

print("=================================================")
print(f"Training Mode : {TRAINING_MODE.upper()}")
print(f"Sections      : {CONFIG['num_sections']}")
print(f"Batch Size    : {CONFIG['batch_size']}")
print(f"ADE Epochs    : {CONFIG['ade_vae_epochs']}")
print(f"FPM Epochs    : {CONFIG['fpm_epochs']}")
print(f"HGNN Epochs   : {CONFIG['hgnn_epochs']}")
print("=================================================")
print('✅ CONFIG dict created with', len(CONFIG), 'entries.')
print(f'   Dataset size  : {CONFIG["num_sections"]} sections × {CONFIG["num_years"]} year(s)')
print(f'   Batch size    : {CONFIG["batch_size"]}')
print(f'   Device        : {CONFIG["device"]}')

# %% [markdown]
# ### Cell 0.8 — Mixed Precision Setup

# %%
# ── Mixed precision training setup (AMP) ──────────────────────────────────────
# GradScaler prevents underflow in FP16 gradients on GPU
_USE_AMP: bool = (CONFIG['device'] == 'cuda')

grad_scaler = GradScaler(enabled=_USE_AMP)

def amp_autocast():
    """Return an autocast context manager appropriate for the current device.

    Uses float16 on CUDA for ~2× throughput on Tensor Cores; falls back to
    a no-op context on CPU.

    Args:
        None

    Returns:
        torch.cuda.amp.autocast context manager.

    Raises:
        Nothing.
    """
    return autocast(enabled=_USE_AMP, dtype=torch.float16)

print(f'⚡ Mixed precision (AMP) : {"ENABLED — float16 on CUDA" if _USE_AMP else "DISABLED — CPU mode"}')

# %% [markdown]
# ### Cell 0.9 — Helper Utilities
#
# Reusable utilities used by **every** downstream section:
# - `EarlyStopping` — patience-based training termination
# - `save_checkpoint` / `load_checkpoint` — robust model persistence
# - `plot_metrics` — training curve visualisation

# %%
class EarlyStopping:
    """Monitors a validation metric and stops training after patience epochs
    without improvement.

    Args:
        patience: Number of epochs to wait after last improvement before
            stopping.  Defaults to 10.
        min_delta: Minimum change in the monitored metric to qualify as an
            improvement.  Defaults to 1e-4.
        mode: One of ``'min'`` or ``'max'``.  ``'min'`` means lower is better
            (e.g. loss), ``'max'`` means higher is better (e.g. F1).
            Defaults to ``'min'``.
        verbose: If ``True``, prints a message on each improvement / stop.
            Defaults to ``True``.

    Returns:
        None (used via its ``__call__`` method).

    Raises:
        ValueError: If *mode* is not ``'min'`` or ``'max'``.
    """

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 1e-4,
        mode: str = 'min',
        verbose: bool = True,
    ) -> None:
        if mode not in ('min', 'max'):
            raise ValueError(f"mode must be 'min' or 'max', got '{mode}'")

        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose

        self.counter: int = 0
        self.best_score: Optional[float] = None
        self.early_stop: bool = False

        self._compare = (lambda cur, best: cur < best - min_delta) \
            if mode == 'min' \
            else (lambda cur, best: cur > best + min_delta)

    def __call__(self, metric: float) -> bool:
        """Check whether training should stop.

        Args:
            metric: Current epoch's monitored metric value.

        Returns:
            ``True`` if training should stop, ``False`` otherwise.

        Raises:
            Nothing.
        """
        if self.best_score is None:
            self.best_score = metric
            if self.verbose:
                print(f'   EarlyStopping: baseline {metric:.6f}')
            return False

        if self._compare(metric, self.best_score):
            self.best_score = metric
            self.counter = 0
            if self.verbose:
                print(f'   EarlyStopping: improved to {metric:.6f}')
        else:
            self.counter += 1
            if self.verbose:
                print(f'   EarlyStopping: no improvement ({self.counter}/{self.patience})')
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print('   🛑 EarlyStopping triggered.')
                return True
        return False

    def reset(self) -> None:
        """Reset internal state so the instance can be reused.

        Args:
            None

        Returns:
            None

        Raises:
            Nothing.
        """
        self.counter = 0
        self.best_score = None
        self.early_stop = False


def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    epoch: int,
    metrics: Optional[Union[Dict[str, float], float]] = None,
    filepath: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
    *,
    loss: Optional[float] = None,
    path: Optional[str] = None,
) -> str:
    """Save a training checkpoint to disk.

    Args:
        model: The PyTorch model whose ``state_dict`` will be saved.
        optimizer: The optimizer whose ``state_dict`` will be saved.
        epoch: Current epoch number (0-indexed).
        metrics: Dictionary of metric name → value at this epoch.
        filepath: Destination file path (e.g. ``checkpoints/best.pt``).
        extra: Optional dictionary of additional data to persist (e.g.
            scheduler state, scaler state).

    Returns:
        The absolute path of the saved checkpoint file as a string.

    Raises:
        OSError: If the file cannot be written.
    """
    if filepath is None:
        filepath = path
    if filepath is None:
        raise ValueError("save_checkpoint requires filepath=... or path=...")

    if metrics is None and loss is not None:
        metrics = {'loss': float(loss)}
    elif isinstance(metrics, (int, float)):
        metrics = {'loss': float(metrics)}
    elif metrics is None:
        metrics = {}

    os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)

    payload: Dict[str, Any] = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
        'timestamp': datetime.datetime.utcnow().isoformat(),
        'torch_version': torch.__version__,
    }
    if extra is not None:
        payload.update(extra)

    torch.save(payload, filepath)
    abs_path = os.path.abspath(filepath)
    print(f'💾 Checkpoint saved → {abs_path}  (epoch {epoch})')
    return abs_path


def load_checkpoint(
    filepath: str,
    model: nn.Module,
    optimizer: Optional[optim.Optimizer] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Load a training checkpoint from disk.

    Args:
        filepath: Path to the ``.pt`` checkpoint file.
        model: Model instance to load weights into (in-place).
        optimizer: Optional optimizer instance to restore state into.
        device: Device to map tensors to.  Defaults to ``CONFIG['device']``.

    Returns:
        The full checkpoint dictionary (contains ``epoch``, ``metrics``,
        ``timestamp``, and any ``extra`` fields).

    Raises:
        FileNotFoundError: If *filepath* does not exist.
        RuntimeError: If the checkpoint is incompatible with the model.
    """
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"No checkpoint at {filepath}")

    map_location = device or torch.device(CONFIG['device'])
    ckpt: Dict[str, Any] = torch.load(filepath, map_location=map_location, weights_only=False)

    model.load_state_dict(ckpt['model_state_dict'])
    if optimizer is not None and 'optimizer_state_dict' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])

    print(f'📂 Checkpoint loaded ← {filepath}  (epoch {ckpt.get("epoch", "?")})')
    return ckpt


def plot_metrics(
    history: Dict[str, List[float]],
    title: str = 'Training Metrics',
    save_path: Optional[str] = None,
) -> None:
    """Plot training & validation metric curves.

    Args:
        history: Dictionary mapping metric names (e.g. ``'train_loss'``,
            ``'val_f1'``) to lists of per-epoch values.
        title: Plot super-title.
        save_path: If provided, the figure is saved to this path in addition
            to being displayed.

    Returns:
        None (displays matplotlib figure).

    Raises:
        ValueError: If *history* is empty.
    """
    if not history:
        raise ValueError("history dict is empty — nothing to plot.")

    # Group metrics by base name (e.g. 'loss' groups 'train_loss' & 'val_loss')
    base_names: Dict[str, List[str]] = collections.defaultdict(list)
    for key in history:
        # Remove train_/val_ prefix to find base name
        base = key.replace('train_', '').replace('val_', '')
        base_names[base].append(key)

    n_plots = len(base_names)
    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 4), squeeze=False)
    axes = axes.flatten()

    for idx, (base, keys) in enumerate(sorted(base_names.items())):
        ax = axes[idx]
        for key in sorted(keys):
            label = key.replace('_', ' ').title()
            ax.plot(history[key], label=label, linewidth=1.5)
        ax.set_xlabel('Epoch')
        ax.set_ylabel(base.replace('_', ' ').title())
        ax.set_title(base.replace('_', ' ').title())
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        fig.savefig(save_path, bbox_inches='tight')
        print(f'📊 Figure saved → {save_path}')

    plt.show()
    plt.close(fig)


print('✅ Helper utilities loaded: EarlyStopping, save_checkpoint, load_checkpoint, plot_metrics')

# %% [markdown]
# ### Cell 0.10 — Section 0 Checkpoint ✅
#
# **Environment is ready.**
#
# | Component | Status |
# |-----------|--------|
# | GPU       | Detected & assigned |
# | Packages  | Installed |
# | Imports   | All loaded |
# | Seeds     | Locked to 42 |
# | Drive     | Mounted (or local fallback) |
# | CONFIG    | 50+ parameters defined |
# | AMP       | GradScaler + autocast ready |
# | Helpers   | EarlyStopping, checkpointing, plotting |
#
# ➡️ Proceed to **Section 1 — Synthetic Data Generator**
