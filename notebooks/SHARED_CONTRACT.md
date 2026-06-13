# SHARED INTERFACE CONTRACT — All Subagents Must Follow This

## CONFIG Dict Structure (defined in Section 0, used everywhere)
```python
CONFIG = {
    # General
    'seed': 42,
    'device': 'cuda',
    'dtype': torch.float32,
    'colab_mode': True,  # Reduced dataset for Colab
    
    # Data
    'num_sections': 50 if COLAB_MODE else 500,
    'num_years': 1 if COLAB_MODE else 5,
    'seq_len': 720,
    'num_stations': 12,
    'failure_rate': 0.03,
    'num_failure_categories': 8,
    'train_ratio': 0.6,
    'val_ratio': 0.2,
    'test_ratio': 0.2,
    'batch_size': 32,
    
    # Vibration
    'vib_channels': 3,
    # Temperature
    'temp_channels': 1,
    # Gauge
    'gauge_channels': 1,
    # Metadata
    'meta_dim': 32,
    # Weather
    'weather_hours': 72,
    'weather_features': 6,
    # Maintenance history
    'maint_events': 16,
    'maint_feat_dim': 64,
    
    # ADE
    'ade_if_trees': 200,
    'ade_if_contamination': 0.03,
    'ade_vae_latent_dim': 32,
    'ade_vae_beta': 0.5,
    'ade_vae_epochs': 50,
    'ade_vae_lr': 1e-3,
    'ade_vae_patience': 10,
    'ade_target_f1': 0.92,
    
    # HM-STT
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
    
    # HGNN
    'hgnn_layers': 4,
    'hgnn_hidden': 128,
    'hgnn_lr': 1e-3,
    'hgnn_epochs': 50,
    'num_node_types': 4,
    'num_relation_types': 4,
    
    # Paths
    'drive_path': '/content/drive/MyDrive/rakshak_v1/',
    'checkpoint_dir': '/content/drive/MyDrive/rakshak_v1/checkpoints/',
    'figures_dir': '/content/drive/MyDrive/rakshak_v1/figures/',
}
```

## Failure Categories (8 classes)
```python
FAILURE_CATEGORIES = [
    'rail_fracture',
    'gauge_deviation',
    'thermal_buckling',
    'ballast_degradation',
    'weld_failure',
    'sleeper_damage',
    'drainage_failure',
    'subgrade_settlement'
]
```

## 12 Phase 1 Stations
```python
STATIONS = [
    'DLI', 'GZB', 'MERT', 'HPJN', 'ALJN', 'KOSI',
    'MATH', 'AGC', 'TDL', 'FRD', 'BRJ', 'MTJ'
]
```

## Dataset Output Format
The `RakshakDataset.__getitem__` returns a dict:
```python
{
    'vibration': torch.FloatTensor,        # [720, 3]
    'temperature': torch.FloatTensor,      # [720, 1]
    'gauge': torch.FloatTensor,            # [720, 1]
    'metadata': torch.FloatTensor,         # [32]
    'weather': torch.FloatTensor,          # [72, 6]
    'maintenance_history': torch.FloatTensor, # [16, 64]
    'section_id': int,
    'edge_index': torch.LongTensor,        # [2, num_edges] (graph adjacency)
    'failure_occurred': torch.FloatTensor,  # [1]
    'failure_category': torch.LongTensor,   # [1] (0-7)
    'time_to_failure': torch.FloatTensor,   # [1] (hours)
}
```

## Key Class Names (must match across sections)
- `RakshakDataset` — Section 1
- `StatisticalDetector` — Section 2
- `IsolationForestDetector` — Section 2
- `SensorVAE` — Section 2
- `AnomalyMetaClassifier` — Section 2
- `TemporalConvBlock` — Section 3
- `ModalityTCNEncoder` — Section 3
- `CrossModalFusionLayer` — Section 3
- `CrossModalFusionTransformer` — Section 3
- `SpatialGAT` — Section 3
- `BiLSTMSequencer` — Section 3
- `PredictionHead` — Section 3
- `HMSTT` — Section 3
- `MultiTaskLoss` — Section 3
- `UncertaintyWrapper` — Section 3
- `RootCauseHGNN` — Section 4
- `BaseAgent` — Section 5
- `MockMessageBus` — Section 5
- `SensorIngestionAgent`, `AnomalyDetectionAgent`, `FailurePredictionAgent`,
  `RootCauseAgent`, `MaintenanceDispatchAgent`, `SpeedRestrictionAgent`,
  `NetworkHealthAgent`, `ExplainabilityAgent`, `LearningAgent`, `OrchestratorAgent` — Section 5

## Event Schema Classes (Pydantic v2, Section 5)
- `SensorPacket`, `SensorPacketValidated`, `SensorFaultEvent`
- `AnomalyEvent`, `FailurePredictionEvent`, `RootCauseReport`
- `MaintenanceTicket`, `TSRAdvisory`, `NetworkHealthUpdate`
- `ExplanationRecord`, `ModelUpdateEvent`, `HITLEscalation`
```
