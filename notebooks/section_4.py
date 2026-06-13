# %% [markdown]
# # Section 4 — Root Cause Analysis with Heterogeneous Graph Neural Network (HGNN)
#
# ## Architecture Overview
#
# This section implements a **Heterogeneous Graph Neural Network** for automated root cause
# analysis of railway infrastructure failures. The approach models the railway system as a
# heterogeneous knowledge graph with four node types and four relation types:
#
# ### Node Types
# | Type | Description | Feature Dim |
# |------|-------------|-------------|
# | **SensorNode** | Individual sensors with reading statistics | 128 |
# | **SectionNode** | Track sections with health embeddings | 128 |
# | **MaintenanceEventNode** | Historical maintenance actions | 128 |
# | **FailureEventNode** | Known failure events with category labels | 128 |
#
# ### Relation Types
# | Relation | Source → Target | Semantics |
# |----------|----------------|-----------|
# | **MEASURES** | SensorNode → SectionNode | Sensor monitors section |
# | **ADJACENT_TO** | SectionNode ↔ SectionNode | Physical adjacency |
# | **APPLIED_TO** | MaintenanceEventNode → SectionNode | Maintenance on section |
# | **PRECEDED_BY** | FailureEventNode → MaintenanceEventNode | Temporal precedence |
#
# ### Model Architecture
# - **4-layer RGCN** (Relational Graph Convolutional Network) with basis decomposition
# - **Layer-wise aggregation** per relation type with skip connections
# - **Link prediction head** using dot-product + MLP scoring for causal ranking
# - **Negative sampling** for efficient training
#
# ### Training Objective
# Binary cross-entropy on positive (known causal) vs. negative (sampled non-causal) edges,
# enabling the model to rank root causes by predicted causal strength.

# %%
# Cell 4.2 — Heterogeneous Graph Builder
# Constructs heterogeneous knowledge graphs with 4 node types and 4 relation types
# for root cause analysis of railway infrastructure failures.

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import math
import json
import hashlib
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict
from tqdm.auto import tqdm

try:
    from torch_geometric.data import HeteroData
    from torch_geometric.nn import RGCNConv
    HAS_PYG = True
except ImportError:
    HAS_PYG = False
    print("[WARN] torch_geometric not found. Using fallback implementations.")


class FallbackHeteroData:
    """Minimal HeteroData replacement when torch_geometric is unavailable.

    Stores node features and edge indices keyed by type/relation tuples,
    mirroring the torch_geometric HeteroData API used in this notebook.

    Attributes:
        _node_store: Dict mapping node type str to a dict of tensor attributes.
        _edge_store: Dict mapping (src, rel, dst) tuples to a dict of tensor attributes.
    """

    def __init__(self) -> None:
        self._node_store: Dict[str, Dict[str, torch.Tensor]] = defaultdict(dict)
        self._edge_store: Dict[Tuple[str, str, str], Dict[str, torch.Tensor]] = defaultdict(dict)
        self._metadata_cache: Optional[Tuple[List[str], List[Tuple[str, str, str]]]] = None

    # --- node access --------------------------------------------------------
    def __getitem__(self, key: str) -> '_NodeView':
        return _NodeView(self._node_store[key])

    def __setitem__(self, key: str, value: Dict[str, torch.Tensor]) -> None:
        self._node_store[key] = value

    # --- edge access --------------------------------------------------------
    def __getattr__(self, name: str) -> Any:
        if name.startswith('_'):
            raise AttributeError(name)
        return self.__dict__.get(name, None)

    def set_edge(self, src_type: str, rel: str, dst_type: str,
                 attr: str, tensor: torch.Tensor) -> None:
        """Set an edge attribute for a given relation triple.

        Args:
            src_type: Source node type name.
            rel: Relation name.
            dst_type: Destination node type name.
            attr: Attribute name (e.g. 'edge_index').
            tensor: Tensor value to store.
        """
        self._edge_store[(src_type, rel, dst_type)][attr] = tensor
        self._metadata_cache = None

    def get_edge(self, src_type: str, rel: str, dst_type: str,
                 attr: str) -> Optional[torch.Tensor]:
        """Retrieve an edge attribute for a given relation triple.

        Args:
            src_type: Source node type name.
            rel: Relation name.
            dst_type: Destination node type name.
            attr: Attribute name.

        Returns:
            The stored tensor, or None if not found.
        """
        return self._edge_store.get((src_type, rel, dst_type), {}).get(attr, None)

    def metadata(self) -> Tuple[List[str], List[Tuple[str, str, str]]]:
        """Return (node_types, edge_types) metadata tuple.

        Returns:
            A tuple of (list of node type strings, list of (src, rel, dst) tuples).
        """
        if self._metadata_cache is None:
            node_types = list(self._node_store.keys())
            edge_types = list(self._edge_store.keys())
            self._metadata_cache = (node_types, edge_types)
        return self._metadata_cache

    @property
    def node_types(self) -> List[str]:
        """List of node type names."""
        return list(self._node_store.keys())

    @property
    def edge_types(self) -> List[Tuple[str, str, str]]:
        """List of edge type triples."""
        return list(self._edge_store.keys())

    def num_nodes_dict(self) -> Dict[str, int]:
        """Return dict mapping node type to number of nodes.

        Returns:
            Dict with node type keys and integer counts.
        """
        result = {}
        for ntype, store in self._node_store.items():
            if 'x' in store:
                result[ntype] = store['x'].shape[0]
            elif 'num_nodes' in store:
                result[ntype] = store['num_nodes']
        return result


class _NodeView:
    """Proxy object for accessing node attributes by type."""

    def __init__(self, store: Dict[str, torch.Tensor]) -> None:
        self._store = store

    def __getattr__(self, name: str) -> torch.Tensor:
        if name.startswith('_'):
            raise AttributeError(name)
        return self._store.get(name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name.startswith('_'):
            super().__setattr__(name, value)
        else:
            self._store[name] = value

    def __contains__(self, key: str) -> bool:
        return key in self._store


if not HAS_PYG:
    HeteroData = FallbackHeteroData  # type: ignore[misc]


# Relation type constants
RELATION_MEASURES = 0       # SensorNode → SectionNode
RELATION_ADJACENT_TO = 1    # SectionNode ↔ SectionNode
RELATION_APPLIED_TO = 2     # MaintenanceEventNode → SectionNode
RELATION_PRECEDED_BY = 3    # FailureEventNode → MaintenanceEventNode

RELATION_NAMES = ['MEASURES', 'ADJACENT_TO', 'APPLIED_TO', 'PRECEDED_BY']

NODE_TYPES = ['sensor', 'section', 'maintenance', 'failure']

EDGE_TYPE_TRIPLES = [
    ('sensor', 'MEASURES', 'section'),
    ('section', 'ADJACENT_TO', 'section'),
    ('maintenance', 'APPLIED_TO', 'section'),
    ('failure', 'PRECEDED_BY', 'maintenance'),
]


class HeterogeneousGraphBuilder:
    """Builds a heterogeneous knowledge graph for root cause analysis.

    Constructs a multi-relational graph representing the railway infrastructure,
    with four distinct node types and four relation types. The graph captures
    sensor-section monitoring relationships, track adjacency, maintenance history,
    and failure-maintenance temporal precedence.

    Node types:
        - SensorNode: Individual sensors with feature vectors
        - SectionNode: Track sections with health embeddings
        - MaintenanceEventNode: Past maintenance actions
        - FailureEventNode: Known failure events with labels

    Relation types:
        - MEASURES: SensorNode → SectionNode
        - ADJACENT_TO: SectionNode ↔ SectionNode
        - APPLIED_TO: MaintenanceEventNode → SectionNode
        - PRECEDED_BY: FailureEventNode → MaintenanceEventNode

    Args:
        config: Global CONFIG dict with graph hyperparameters.
        feature_dim: Dimension of node feature vectors (default 128).
    """

    def __init__(self, config: Dict[str, Any], feature_dim: int = 128) -> None:
        self.config = config
        self.feature_dim = feature_dim
        self.seed = config.get('seed', 42)
        self.initializer = NodeFeatureInitializer(feature_dim=feature_dim)

    def build_synthetic_graph(
        self,
        num_sections: int = 50,
        num_sensors_per_section: int = 5,
        num_events: int = 200
    ) -> HeteroData:
        """Build a synthetic heterogeneous graph for root cause analysis.

        Creates a complete heterogeneous graph with realistic structure including
        sensor-section assignments, track adjacency chains, maintenance events,
        and failure events with known causal links.

        Args:
            num_sections: Number of track sections (SectionNode count).
            num_sensors_per_section: Sensors per section (total SensorNodes =
                num_sections * num_sensors_per_section).
            num_events: Total number of maintenance + failure events to generate.

        Returns:
            HeteroData object containing node features and edge indices for all
            node types and relation types. Includes ground-truth causal labels
            on failure nodes.
        """
        rng = np.random.RandomState(self.seed)

        num_sensors = num_sections * num_sensors_per_section
        num_maintenance = int(num_events * 0.6)  # 60% maintenance events
        num_failures = num_events - num_maintenance  # 40% failure events

        # --- Node feature generation ---
        sensor_types = ['accelerometer', 'thermometer', 'gauge_meter',
                        'strain_gauge', 'acoustic_emission']

        sensor_features_list = []
        for i in range(num_sensors):
            stype = sensor_types[i % len(sensor_types)]
            readings_stats = {
                'mean': rng.uniform(-1, 1),
                'std': rng.uniform(0.1, 2.0),
                'max': rng.uniform(1.0, 5.0),
                'min': rng.uniform(-5.0, -1.0),
                'kurtosis': rng.uniform(-1, 3),
                'skew': rng.uniform(-1, 1),
            }
            feat = self.initializer.sensor_features(stype, readings_stats)
            sensor_features_list.append(feat)
        sensor_x = torch.stack(sensor_features_list)  # [num_sensors, 128]

        section_features_list = []
        for i in range(num_sections):
            metadata = {
                'station_idx': i % len(STATIONS) if 'STATIONS' in dir() else i % 12,
                'km_marker': rng.uniform(0, 500),
                'track_age_years': rng.uniform(1, 40),
                'curve_radius': rng.uniform(200, 5000),
                'gradient': rng.uniform(-0.02, 0.02),
                'sleeper_type': rng.randint(0, 3),
                'rail_weight': rng.choice([52, 60, 68]),
            }
            health_scores = {
                'vibration_health': rng.uniform(0.3, 1.0),
                'thermal_stability': rng.uniform(0.4, 1.0),
                'gauge_compliance': rng.uniform(0.5, 1.0),
                'overall_thi': rng.uniform(0.3, 1.0),
            }
            feat = self.initializer.section_features(metadata, health_scores)
            section_features_list.append(feat)
        section_x = torch.stack(section_features_list)  # [num_sections, 128]

        maintenance_types = ['rail_grinding', 'sleeper_replacement', 'tamping',
                             'welding_repair', 'ballast_renewal', 'drainage_clearing',
                             'rail_replacement', 'inspection']

        maint_features_list = []
        maint_section_map = []
        for i in range(num_maintenance):
            etype = maintenance_types[i % len(maintenance_types)]
            severity = rng.choice(['minor', 'moderate', 'major', 'critical'])
            outcome = rng.choice(['resolved', 'partial', 'deferred', 'escalated'])
            feat = self.initializer.maintenance_features(etype, severity, outcome)
            maint_features_list.append(feat)
            maint_section_map.append(rng.randint(0, num_sections))
        maint_x = torch.stack(maint_features_list)  # [num_maintenance, 128]

        failure_cats = FAILURE_CATEGORIES if 'FAILURE_CATEGORIES' in dir() else [
            'rail_fracture', 'gauge_deviation', 'thermal_buckling',
            'ballast_degradation', 'weld_failure', 'sleeper_damage',
            'drainage_failure', 'subgrade_settlement'
        ]

        failure_features_list = []
        failure_section_map = []
        failure_category_labels = []
        failure_maint_links = []  # (failure_idx, maint_idx) — ground truth causal links
        for i in range(num_failures):
            cat = failure_cats[i % len(failure_cats)]
            severity = rng.choice(['low', 'medium', 'high', 'critical'])
            ts_feats = {
                'hour_sin': np.sin(2 * np.pi * rng.uniform(0, 24) / 24),
                'hour_cos': np.cos(2 * np.pi * rng.uniform(0, 24) / 24),
                'day_of_week': rng.randint(0, 7) / 7.0,
                'month_sin': np.sin(2 * np.pi * rng.uniform(1, 12) / 12),
                'month_cos': np.cos(2 * np.pi * rng.uniform(1, 12) / 12),
            }
            feat = self.initializer.failure_features(cat, severity, ts_feats)
            failure_features_list.append(feat)

            section_idx = rng.randint(0, num_sections)
            failure_section_map.append(section_idx)
            failure_category_labels.append(failure_cats.index(cat))

            # Link to 1-3 preceding maintenance events on the same section
            same_section_maint = [j for j, s in enumerate(maint_section_map) if s == section_idx]
            if same_section_maint:
                num_links = min(rng.randint(1, 4), len(same_section_maint))
                linked = rng.choice(same_section_maint, size=num_links, replace=False).tolist()
                for m_idx in linked:
                    failure_maint_links.append((i, m_idx))
            else:
                # Link to a random maintenance event as fallback
                m_idx = rng.randint(0, num_maintenance)
                failure_maint_links.append((i, m_idx))

        failure_x = torch.stack(failure_features_list)  # [num_failures, 128]

        # --- Edge construction ---
        # MEASURES: each sensor → its assigned section
        sensor_src = torch.arange(num_sensors, dtype=torch.long)
        sensor_dst = torch.arange(num_sections, dtype=torch.long).repeat_interleave(
            num_sensors_per_section
        )
        measures_edge_index = torch.stack([sensor_src, sensor_dst])  # [2, num_sensors]

        # ADJACENT_TO: linear chain + some random skip connections
        adj_src_list, adj_dst_list = [], []
        for i in range(num_sections - 1):
            adj_src_list.extend([i, i + 1])
            adj_dst_list.extend([i + 1, i])
        # Add ~10% random skip connections for realism
        num_skip = max(1, num_sections // 10)
        for _ in range(num_skip):
            a, b = rng.randint(0, num_sections, size=2)
            if a != b:
                adj_src_list.extend([a, b])
                adj_dst_list.extend([b, a])
        adjacent_edge_index = torch.tensor(
            [adj_src_list, adj_dst_list], dtype=torch.long
        )  # [2, num_adj_edges]

        # APPLIED_TO: maintenance → section
        applied_src = torch.arange(num_maintenance, dtype=torch.long)
        applied_dst = torch.tensor(maint_section_map, dtype=torch.long)
        applied_edge_index = torch.stack([applied_src, applied_dst])  # [2, num_maintenance]

        # PRECEDED_BY: failure → maintenance (ground truth causal edges)
        if failure_maint_links:
            preceded_src = torch.tensor([f for f, m in failure_maint_links], dtype=torch.long)
            preceded_dst = torch.tensor([m for f, m in failure_maint_links], dtype=torch.long)
            preceded_edge_index = torch.stack([preceded_src, preceded_dst])
        else:
            preceded_edge_index = torch.zeros(2, 0, dtype=torch.long)

        # --- Assemble HeteroData ---
        if HAS_PYG:
            graph = HeteroData()
            graph['sensor'].x = sensor_x
            graph['section'].x = section_x
            graph['maintenance'].x = maint_x
            graph['failure'].x = failure_x
            graph['failure'].y = torch.tensor(failure_category_labels, dtype=torch.long)
            graph['sensor', 'MEASURES', 'section'].edge_index = measures_edge_index
            graph['section', 'ADJACENT_TO', 'section'].edge_index = adjacent_edge_index
            graph['maintenance', 'APPLIED_TO', 'section'].edge_index = applied_edge_index
            graph['failure', 'PRECEDED_BY', 'maintenance'].edge_index = preceded_edge_index
        else:
            graph = FallbackHeteroData()
            graph['sensor'].x = sensor_x
            graph['section'].x = section_x
            graph['maintenance'].x = maint_x
            graph['failure'].x = failure_x
            graph['failure'].y = torch.tensor(failure_category_labels, dtype=torch.long)
            graph.set_edge('sensor', 'MEASURES', 'section', 'edge_index', measures_edge_index)
            graph.set_edge('section', 'ADJACENT_TO', 'section', 'edge_index', adjacent_edge_index)
            graph.set_edge('maintenance', 'APPLIED_TO', 'section', 'edge_index', applied_edge_index)
            graph.set_edge('failure', 'PRECEDED_BY', 'maintenance', 'edge_index', preceded_edge_index)

        # Store metadata for later use
        graph._num_sections = num_sections
        graph._num_sensors = num_sensors
        graph._num_maintenance = num_maintenance
        graph._num_failures = num_failures
        graph._failure_section_map = failure_section_map
        graph._maint_section_map = maint_section_map
        graph._failure_maint_links = failure_maint_links

        print(f"[GraphBuilder] Built heterogeneous graph:")
        print(f"  Sensor nodes:      {num_sensors}")
        print(f"  Section nodes:     {num_sections}")
        print(f"  Maintenance nodes: {num_maintenance}")
        print(f"  Failure nodes:     {num_failures}")
        print(f"  MEASURES edges:    {measures_edge_index.shape[1]}")
        print(f"  ADJACENT_TO edges: {adjacent_edge_index.shape[1]}")
        print(f"  APPLIED_TO edges:  {applied_edge_index.shape[1]}")
        print(f"  PRECEDED_BY edges: {preceded_edge_index.shape[1]}")

        return graph

    def inject_known_causes(
        self,
        graph: HeteroData,
        cause_pairs: List[Tuple[int, int]]
    ) -> HeteroData:
        """Inject known causal edges (failure → maintenance) into the graph.

        Adds additional ground-truth causal links between failure events and
        maintenance events to enrich the training signal.

        Args:
            graph: An existing HeteroData graph to augment.
            cause_pairs: List of (failure_node_idx, maintenance_node_idx) pairs
                representing known causal relationships.

        Returns:
            The same HeteroData graph with additional PRECEDED_BY edges and
            updated causal metadata.
        """
        if not cause_pairs:
            return graph

        new_src = torch.tensor([f for f, m in cause_pairs], dtype=torch.long)
        new_dst = torch.tensor([m for f, m in cause_pairs], dtype=torch.long)

        if HAS_PYG:
            existing = graph['failure', 'PRECEDED_BY', 'maintenance'].edge_index
        else:
            existing = graph.get_edge('failure', 'PRECEDED_BY', 'maintenance', 'edge_index')

        if existing is not None and existing.shape[1] > 0:
            combined_src = torch.cat([existing[0], new_src])
            combined_dst = torch.cat([existing[1], new_dst])
        else:
            combined_src = new_src
            combined_dst = new_dst

        new_edge_index = torch.stack([combined_src, combined_dst])

        if HAS_PYG:
            graph['failure', 'PRECEDED_BY', 'maintenance'].edge_index = new_edge_index
        else:
            graph.set_edge('failure', 'PRECEDED_BY', 'maintenance',
                           'edge_index', new_edge_index)

        # Update metadata
        if hasattr(graph, '_failure_maint_links'):
            graph._failure_maint_links = graph._failure_maint_links + cause_pairs

        print(f"[GraphBuilder] Injected {len(cause_pairs)} known causal edges. "
              f"Total PRECEDED_BY edges: {new_edge_index.shape[1]}")

        return graph


print("[Section 4] HeterogeneousGraphBuilder defined.")

# %%
# Cell 4.3 — Node Feature Initializers
# Creates fixed-dimension feature vectors for each of the four node types
# using learned-style encodings (one-hot + continuous concatenation).


class NodeFeatureInitializer:
    """Initializes feature vectors for each node type in the heterogeneous graph.

    Maps raw node attributes (sensor type, health scores, event metadata) into
    fixed-dimension feature vectors suitable for GNN processing. Uses a combination
    of one-hot encoding for categorical features and normalized continuous features,
    padded or truncated to the target dimension.

    Args:
        feature_dim: Target dimension for all feature vectors (default 128).
    """

    # Class-level type vocabularies for consistent encoding
    SENSOR_TYPES = ['accelerometer', 'thermometer', 'gauge_meter',
                    'strain_gauge', 'acoustic_emission']
    MAINTENANCE_TYPES = ['rail_grinding', 'sleeper_replacement', 'tamping',
                         'welding_repair', 'ballast_renewal', 'drainage_clearing',
                         'rail_replacement', 'inspection']
    SEVERITY_LEVELS = ['minor', 'moderate', 'major', 'critical',
                       'low', 'medium', 'high']
    OUTCOME_TYPES = ['resolved', 'partial', 'deferred', 'escalated']
    FAILURE_CATEGORIES = [
        'rail_fracture', 'gauge_deviation', 'thermal_buckling',
        'ballast_degradation', 'weld_failure', 'sleeper_damage',
        'drainage_failure', 'subgrade_settlement'
    ]

    def __init__(self, feature_dim: int = 128) -> None:
        self.feature_dim = feature_dim

    def _one_hot(self, value: str, vocabulary: List[str]) -> List[float]:
        """Create a one-hot encoding for a categorical value.

        Args:
            value: The categorical value to encode.
            vocabulary: List of possible values.

        Returns:
            List of floats representing the one-hot vector.
        """
        vec = [0.0] * len(vocabulary)
        if value in vocabulary:
            vec[vocabulary.index(value)] = 1.0
        return vec

    def _pad_or_truncate(self, features: List[float]) -> torch.Tensor:
        """Pad with zeros or truncate to reach target feature_dim.

        Args:
            features: List of float feature values.

        Returns:
            Tensor of shape [feature_dim].
        """
        if len(features) >= self.feature_dim:
            return torch.tensor(features[:self.feature_dim], dtype=torch.float32)
        padded = features + [0.0] * (self.feature_dim - len(features))
        return torch.tensor(padded, dtype=torch.float32)

    def sensor_features(
        self,
        sensor_type: str,
        readings_stats: Dict[str, float]
    ) -> torch.Tensor:
        """Create feature vector for a sensor node.

        Encodes the sensor type as one-hot and concatenates statistical summaries
        of the sensor readings (mean, std, max, min, kurtosis, skew).

        Args:
            sensor_type: Type of sensor (e.g., 'accelerometer', 'thermometer').
            readings_stats: Dictionary with keys 'mean', 'std', 'max', 'min',
                'kurtosis', 'skew' containing float values.

        Returns:
            Feature tensor of shape [128].
        """
        features: List[float] = []
        features.extend(self._one_hot(sensor_type, self.SENSOR_TYPES))  # [5]
        features.append(readings_stats.get('mean', 0.0))
        features.append(readings_stats.get('std', 1.0))
        features.append(readings_stats.get('max', 0.0))
        features.append(readings_stats.get('min', 0.0))
        features.append(readings_stats.get('kurtosis', 0.0))
        features.append(readings_stats.get('skew', 0.0))  # [11 total]
        # Add derived features
        features.append(readings_stats.get('max', 0.0) - readings_stats.get('min', 0.0))  # range
        features.append(abs(readings_stats.get('mean', 0.0)) / max(readings_stats.get('std', 1.0), 1e-8))  # SNR proxy
        # Hash-based embedding of sensor type for additional discrimination
        type_hash = int(hashlib.md5(sensor_type.encode()).hexdigest()[:8], 16)
        for bit in range(16):
            features.append(float((type_hash >> bit) & 1))  # [29 total]
        return self._pad_or_truncate(features)  # [128]

    def section_features(
        self,
        metadata: Dict[str, Any],
        health_scores: Dict[str, float]
    ) -> torch.Tensor:
        """Create feature vector for a track section node.

        Encodes physical track properties (age, curvature, gradient) and
        current health indicator scores.

        Args:
            metadata: Dictionary with keys like 'station_idx', 'km_marker',
                'track_age_years', 'curve_radius', 'gradient', 'sleeper_type',
                'rail_weight'.
            health_scores: Dictionary with keys 'vibration_health',
                'thermal_stability', 'gauge_compliance', 'overall_thi'.

        Returns:
            Feature tensor of shape [128].
        """
        features: List[float] = []
        # Station one-hot (12 stations)
        station_vec = [0.0] * 12
        station_idx = int(metadata.get('station_idx', 0)) % 12
        station_vec[station_idx] = 1.0
        features.extend(station_vec)  # [12]
        # Continuous physical properties
        features.append(metadata.get('km_marker', 0.0) / 500.0)  # normalized
        features.append(metadata.get('track_age_years', 10.0) / 40.0)
        features.append(metadata.get('curve_radius', 1000.0) / 5000.0)
        features.append(metadata.get('gradient', 0.0))
        # Sleeper type one-hot (3 types)
        sleeper_vec = [0.0] * 3
        sleeper_vec[int(metadata.get('sleeper_type', 0)) % 3] = 1.0
        features.extend(sleeper_vec)  # [19]
        # Rail weight normalized
        features.append(metadata.get('rail_weight', 60) / 68.0)  # [20]
        # Health scores
        features.append(health_scores.get('vibration_health', 0.5))
        features.append(health_scores.get('thermal_stability', 0.5))
        features.append(health_scores.get('gauge_compliance', 0.5))
        features.append(health_scores.get('overall_thi', 0.5))  # [24]
        # Derived health features
        health_vals = [health_scores.get(k, 0.5) for k in
                       ['vibration_health', 'thermal_stability', 'gauge_compliance']]
        features.append(min(health_vals))  # worst indicator
        features.append(max(health_vals) - min(health_vals))  # health spread
        features.append(np.std(health_vals))  # health variability  # [27]
        return self._pad_or_truncate(features)  # [128]

    def maintenance_features(
        self,
        event_type: str,
        severity: str,
        outcome: str
    ) -> torch.Tensor:
        """Create feature vector for a maintenance event node.

        Encodes the maintenance action type, severity level, and resolution
        outcome as a combination of one-hot and ordinal features.

        Args:
            event_type: Type of maintenance (e.g., 'rail_grinding', 'tamping').
            severity: Severity level ('minor', 'moderate', 'major', 'critical').
            outcome: Resolution outcome ('resolved', 'partial', 'deferred', 'escalated').

        Returns:
            Feature tensor of shape [128].
        """
        features: List[float] = []
        features.extend(self._one_hot(event_type, self.MAINTENANCE_TYPES))  # [8]
        features.extend(self._one_hot(severity, self.SEVERITY_LEVELS))  # [15]
        features.extend(self._one_hot(outcome, self.OUTCOME_TYPES))  # [19]
        # Ordinal severity encoding
        sev_map = {'minor': 0.25, 'moderate': 0.5, 'major': 0.75, 'critical': 1.0,
                   'low': 0.25, 'medium': 0.5, 'high': 0.75}
        features.append(sev_map.get(severity, 0.5))  # [20]
        # Outcome quality score (higher = better resolution)
        outcome_map = {'resolved': 1.0, 'partial': 0.6, 'deferred': 0.3, 'escalated': 0.1}
        features.append(outcome_map.get(outcome, 0.5))  # [21]
        # Interaction features
        features.append(sev_map.get(severity, 0.5) * outcome_map.get(outcome, 0.5))  # [22]
        return self._pad_or_truncate(features)  # [128]

    def failure_features(
        self,
        category: str,
        severity: str,
        timestamp_features: Dict[str, float]
    ) -> torch.Tensor:
        """Create feature vector for a failure event node.

        Encodes the failure category, severity, and temporal features
        (cyclical time encodings) into a fixed-dimension vector.

        Args:
            category: Failure category from FAILURE_CATEGORIES list.
            severity: Severity level ('low', 'medium', 'high', 'critical').
            timestamp_features: Dictionary with cyclical time encodings:
                'hour_sin', 'hour_cos', 'day_of_week', 'month_sin', 'month_cos'.

        Returns:
            Feature tensor of shape [128].
        """
        features: List[float] = []
        features.extend(self._one_hot(category, self.FAILURE_CATEGORIES))  # [8]
        features.extend(self._one_hot(severity, self.SEVERITY_LEVELS))  # [15]
        # Temporal features
        features.append(timestamp_features.get('hour_sin', 0.0))
        features.append(timestamp_features.get('hour_cos', 0.0))
        features.append(timestamp_features.get('day_of_week', 0.0))
        features.append(timestamp_features.get('month_sin', 0.0))
        features.append(timestamp_features.get('month_cos', 0.0))  # [20]
        # Severity ordinal
        sev_map = {'low': 0.25, 'medium': 0.5, 'high': 0.75, 'critical': 1.0}
        features.append(sev_map.get(severity, 0.5))  # [21]
        # Category risk priors (domain knowledge)
        risk_priors = {
            'rail_fracture': 1.0, 'gauge_deviation': 0.8, 'thermal_buckling': 0.9,
            'ballast_degradation': 0.5, 'weld_failure': 0.85, 'sleeper_damage': 0.6,
            'drainage_failure': 0.4, 'subgrade_settlement': 0.7,
        }
        features.append(risk_priors.get(category, 0.5))  # [22]
        return self._pad_or_truncate(features)  # [128]


print("[Section 4] NodeFeatureInitializer defined.")

# %%
# Cell 4.4 — RootCauseHGNN Model
# 4-layer RGCN with link prediction head for causal ranking.
# Uses homogeneous RGCN formulation operating on a unified edge_index + edge_type.


class CausalLinkPredictor(nn.Module):
    """MLP-based link prediction head for causal edge scoring.

    Takes concatenated source and target node embeddings and produces a scalar
    score indicating the likelihood of a causal relationship.

    Architecture:
        Linear(2*dim, dim) → ReLU → Dropout → Linear(dim, dim//2) → ReLU →
        Dropout → Linear(dim//2, 1) → Sigmoid

    Args:
        embed_dim: Dimension of node embeddings.
        dropout: Dropout rate for regularization (default 0.2).
    """

    def __init__(self, embed_dim: int, dropout: float = 0.2) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2 * embed_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, 1),
        )

    def forward(
        self,
        src_emb: torch.Tensor,
        dst_emb: torch.Tensor
    ) -> torch.Tensor:
        """Score candidate causal links between source and destination nodes.

        Args:
            src_emb: Source (query) node embeddings, shape [N, D].
            dst_emb: Destination (candidate cause) node embeddings, shape [N, D].

        Returns:
            Causal likelihood scores in [0, 1], shape [N].
        """
        combined = torch.cat([src_emb, dst_emb], dim=-1)  # [N, 2*D]
        scores = self.mlp(combined).squeeze(-1)  # [N]
        return scores


class RGCNLayerFallback(nn.Module):
    """Fallback RGCN layer when torch_geometric is not available.

    Implements a basic Relational Graph Convolutional layer with per-relation
    weight matrices and basis decomposition.

    Args:
        in_channels: Input feature dimension.
        out_channels: Output feature dimension.
        num_relations: Number of distinct relation types.
        num_bases: Number of basis matrices for weight decomposition (default 4).
    """

    def __init__(self, in_channels: int, out_channels: int,
                 num_relations: int, num_bases: int = 4) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_relations = num_relations
        self.num_bases = min(num_bases, num_relations)

        # Basis decomposition: W_r = sum_b a_{r,b} * V_b
        self.bases = nn.Parameter(
            torch.randn(self.num_bases, in_channels, out_channels) * 0.01
        )
        self.att = nn.Parameter(
            torch.randn(num_relations, self.num_bases) * 0.01
        )
        self.bias = nn.Parameter(torch.zeros(out_channels))
        self.self_loop = nn.Linear(in_channels, out_channels, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass through the RGCN layer.

        Args:
            x: Node feature matrix, shape [N, in_channels].
            edge_index: Edge index tensor, shape [2, E].
            edge_type: Edge type labels, shape [E].

        Returns:
            Updated node features, shape [N, out_channels].
        """
        N = x.size(0)
        out = torch.zeros(N, self.out_channels, device=x.device, dtype=x.dtype)

        # Compute per-relation weight matrices via basis decomposition
        # att: [num_relations, num_bases], bases: [num_bases, in, out]
        weights = torch.einsum('rb,bio->rio', self.att, self.bases)  # [R, in, out]

        for r in range(self.num_relations):
            mask = edge_type == r  # [E]
            if mask.sum() == 0:
                continue
            src = edge_index[0, mask]  # [E_r]
            dst = edge_index[1, mask]  # [E_r]
            msg = x[src] @ weights[r]  # [E_r, out_channels]

            # Degree-normalized aggregation
            out.index_add_(0, dst, msg)

        # Degree normalization
        _, counts = torch.unique(edge_index[1], return_counts=True)
        degree = torch.zeros(N, 1, device=x.device, dtype=x.dtype)
        unique_dst = torch.unique(edge_index[1])
        degree[unique_dst] = counts.float().unsqueeze(1)
        degree = degree.clamp(min=1.0)
        out = out / degree

        # Self-loop + bias
        out = out + self.self_loop(x) + self.bias

        return out


class RootCauseHGNN(nn.Module):
    """Heterogeneous Graph Neural Network for root cause analysis.

    Implements a multi-layer Relational Graph Convolutional Network (RGCN) that
    operates on the unified homogeneous representation of the heterogeneous graph.
    Includes a link prediction head for causal ranking of root causes.

    Architecture:
        - 4-layer RGCN with basis decomposition per relation type
        - Skip connections between layers with learnable gating
        - Layer normalization after each convolution
        - Link prediction head using dot-product + MLP scoring

    Args:
        in_channels: Input feature dimension (128).
        hidden_channels: Hidden dimension (128).
        out_channels: Output embedding dimension (64).
        num_relations: Number of relation types (4).
        num_layers: Number of RGCN layers (4).
        dropout: Dropout rate (default 0.2).
    """

    def __init__(
        self,
        in_channels: int = 128,
        hidden_channels: int = 128,
        out_channels: int = 64,
        num_relations: int = 4,
        num_layers: int = 4,
        dropout: float = 0.2
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_relations = num_relations
        self.num_layers = num_layers
        self.dropout_rate = dropout

        # Input projection
        self.input_proj = nn.Linear(in_channels, hidden_channels)  # [B, 128] → [B, 128]
        self.input_norm = nn.LayerNorm(hidden_channels)

        # RGCN layers
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.gates = nn.ModuleList()  # Learnable skip connection gates

        for i in range(num_layers):
            if HAS_PYG:
                conv = RGCNConv(
                    in_channels=hidden_channels,
                    out_channels=hidden_channels,
                    num_relations=num_relations,
                    num_bases=min(4, num_relations),
                    aggr='mean',
                )
            else:
                conv = RGCNLayerFallback(
                    in_channels=hidden_channels,
                    out_channels=hidden_channels,
                    num_relations=num_relations,
                    num_bases=min(4, num_relations),
                )
            self.convs.append(conv)
            self.norms.append(nn.LayerNorm(hidden_channels))
            # Gate for residual connection: g * new + (1-g) * old
            self.gates.append(nn.Sequential(
                nn.Linear(2 * hidden_channels, 1),
                nn.Sigmoid(),
            ))

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, out_channels),
        )

        # Link prediction head for causal scoring
        self.causal_predictor = CausalLinkPredictor(
            embed_dim=out_channels,
            dropout=dropout,
        )

        self.dropout = nn.Dropout(dropout)

        # Initialize weights
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize model weights using Xavier uniform for linear layers."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass: encode all nodes through RGCN layers.

        Args:
            x: Concatenated node feature matrix, shape [N, in_channels]
                where N = total nodes across all types.
            edge_index: Global edge index, shape [2, E].
            edge_type: Edge type labels, shape [E], values in [0, num_relations).

        Returns:
            Node embeddings, shape [N, out_channels].
        """
        # Input projection
        h = self.input_proj(x)       # [N, hidden_channels]
        h = self.input_norm(h)       # [N, hidden_channels]
        h = F.relu(h)
        h = self.dropout(h)

        # RGCN layers with gated residual connections
        for i in range(self.num_layers):
            h_new = self.convs[i](h, edge_index, edge_type)  # [N, hidden_channels]
            h_new = self.norms[i](h_new)                     # [N, hidden_channels]
            h_new = F.relu(h_new)
            h_new = self.dropout(h_new)

            # Learnable gated residual
            gate_input = torch.cat([h, h_new], dim=-1)  # [N, 2*hidden_channels]
            gate = self.gates[i](gate_input)             # [N, 1]
            h = gate * h_new + (1 - gate) * h            # [N, hidden_channels]

        # Output projection
        embeddings = self.output_proj(h)  # [N, out_channels]

        return embeddings

    def predict_cause(
        self,
        query_node_emb: torch.Tensor,
        candidate_node_embs: torch.Tensor
    ) -> torch.Tensor:
        """Rank candidate root causes for a query failure node.

        Uses both dot-product similarity and the MLP-based causal predictor
        to produce a fused ranking score for each candidate.

        Args:
            query_node_emb: Embedding of the failure node being queried,
                shape [D] or [1, D].
            candidate_node_embs: Embeddings of candidate cause nodes,
                shape [K, D] where K is the number of candidates.

        Returns:
            Ranked cause scores in [0, 1], shape [K], higher = more likely cause.
        """
        if query_node_emb.dim() == 1:
            query_node_emb = query_node_emb.unsqueeze(0)  # [1, D]

        K = candidate_node_embs.size(0)

        # Dot-product similarity scores
        dot_scores = torch.matmul(
            candidate_node_embs, query_node_emb.squeeze(0)
        )  # [K]
        dot_scores = torch.sigmoid(dot_scores)  # [K], normalize to [0, 1]

        # MLP causal prediction scores
        query_expanded = query_node_emb.expand(K, -1)  # [K, D]
        mlp_scores = self.causal_predictor(
            query_expanded, candidate_node_embs
        )  # [K]

        # Fused score: weighted average of dot-product and MLP
        fused_scores = 0.4 * dot_scores + 0.6 * mlp_scores  # [K]

        return fused_scores

    def compute_link_loss(
        self,
        node_embs: torch.Tensor,
        pos_edge_index: torch.Tensor,
        neg_edge_index: torch.Tensor
    ) -> torch.Tensor:
        """Compute BCE loss for link prediction (positive vs negative edges).

        Args:
            node_embs: All node embeddings, shape [N, D].
            pos_edge_index: Positive (ground truth) edge index, shape [2, P].
            neg_edge_index: Negative (sampled) edge index, shape [2, Q].

        Returns:
            Scalar BCE loss value.
        """
        # Positive edge scores
        pos_src_emb = node_embs[pos_edge_index[0]]  # [P, D]
        pos_dst_emb = node_embs[pos_edge_index[1]]  # [P, D]
        pos_scores = self.causal_predictor(pos_src_emb, pos_dst_emb)  # [P]

        # Negative edge scores
        neg_src_emb = node_embs[neg_edge_index[0]]  # [Q, D]
        neg_dst_emb = node_embs[neg_edge_index[1]]  # [Q, D]
        neg_scores = self.causal_predictor(neg_src_emb, neg_dst_emb)  # [Q]

        # BCE loss
        pos_labels = torch.ones_like(pos_scores)
        neg_labels = torch.zeros_like(neg_scores)

        scores = torch.cat([pos_scores, neg_scores], dim=0)
        labels = torch.cat([pos_labels, neg_labels], dim=0).float()

        assert torch.isfinite(scores).all(), \
        "Non-finite logits detected"

        assert torch.isfinite(labels).all(), \
        "Non-finite labels detected"

        loss = F.binary_cross_entropy_with_logits(scores, labels)
        return loss


print("[Section 4] RootCauseHGNN model defined.")

# %%
# Cell 4.5 — HGNN Training
# Trains the RootCauseHGNN with link prediction loss using negative sampling.
# Converts heterogeneous graph to homogeneous representation for RGCN.


def convert_hetero_to_homo(
    graph: HeteroData,
    device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Tuple[int, int]]]:
    """Convert a heterogeneous graph to a homogeneous representation.

    Concatenates all node features into a single matrix and remaps all edge
    indices to use global node IDs, with an accompanying edge_type tensor.

    Args:
        graph: HeteroData object with per-type node features and edge indices.
        device: Target torch device.

    Returns:
        Tuple of:
            - x: Concatenated node features, shape [N_total, feature_dim].
            - edge_index: Global edge index, shape [2, E_total].
            - edge_type: Edge type labels, shape [E_total].
            - node_offsets: Dict mapping node type str to (start_idx, count).
    """
    node_features = []
    node_offsets: Dict[str, Tuple[int, int]] = {}
    current_offset = 0

    for ntype in NODE_TYPES:
        if HAS_PYG:
            feats = graph[ntype].x
        else:
            feats = graph[ntype].x
        if feats is None:
            continue
        n_nodes = feats.shape[0]
        node_features.append(feats)
        node_offsets[ntype] = (current_offset, n_nodes)
        current_offset += n_nodes

    x = torch.cat(node_features, dim=0).to(device)  # [N_total, feat_dim]

    all_edges_src = []
    all_edges_dst = []
    all_edge_types = []

    for rel_idx, (src_type, rel_name, dst_type) in enumerate(EDGE_TYPE_TRIPLES):
        if HAS_PYG:
            ei = graph[src_type, rel_name, dst_type].edge_index
        else:
            ei = graph.get_edge(src_type, rel_name, dst_type, 'edge_index')
        if ei is None or ei.shape[1] == 0:
            continue

        src_offset, _ = node_offsets[src_type]
        dst_offset, _ = node_offsets[dst_type]

        all_edges_src.append(ei[0] + src_offset)
        all_edges_dst.append(ei[1] + dst_offset)
        all_edge_types.append(torch.full((ei.shape[1],), rel_idx, dtype=torch.long))

    if all_edges_src:
        edge_index = torch.stack([
            torch.cat(all_edges_src),
            torch.cat(all_edges_dst),
        ]).to(device)  # [2, E_total]
        edge_type = torch.cat(all_edge_types).to(device)  # [E_total]
    else:
        edge_index = torch.zeros(2, 0, dtype=torch.long, device=device)
        edge_type = torch.zeros(0, dtype=torch.long, device=device)

    return x, edge_index, edge_type, node_offsets


def sample_negative_edges(
    pos_edge_index: torch.Tensor,
    num_nodes: int,
    num_neg_samples: int,
    seed: int = 42
) -> torch.Tensor:
    """Sample negative edges that do not exist in the positive edge set.

    Uses rejection sampling to generate edges between node pairs that have
    no existing positive connection.

    Args:
        pos_edge_index: Positive edge index, shape [2, P].
        num_nodes: Total number of nodes in the graph.
        num_neg_samples: Number of negative edges to sample.
        seed: Random seed for reproducibility.

    Returns:
        Negative edge index, shape [2, num_neg_samples].

    Raises:
        ValueError: If num_neg_samples exceeds available negative edge space.
    """
    rng = np.random.RandomState(seed)

    # Build set of existing edges for O(1) lookup
    pos_set = set()
    for i in range(pos_edge_index.shape[1]):
        src = pos_edge_index[0, i].item()
        dst = pos_edge_index[1, i].item()
        pos_set.add((src, dst))

    neg_src, neg_dst = [], []
    attempts = 0
    max_attempts = num_neg_samples * 10

    while len(neg_src) < num_neg_samples and attempts < max_attempts:
        s = rng.randint(0, num_nodes)
        d = rng.randint(0, num_nodes)
        if s != d and (s, d) not in pos_set:
            neg_src.append(s)
            neg_dst.append(d)
            pos_set.add((s, d))  # prevent duplicates
        attempts += 1

    # Pad with random pairs if we couldn't find enough unique negatives
    while len(neg_src) < num_neg_samples:
        s = rng.randint(0, num_nodes)
        d = rng.randint(0, num_nodes)
        neg_src.append(s)
        neg_dst.append(d)

    return torch.tensor([neg_src[:num_neg_samples],
                         neg_dst[:num_neg_samples]], dtype=torch.long)


def train_hgnn(
    model: RootCauseHGNN,
    graph_data: HeteroData,
    config: Dict[str, Any],
    device: Optional[torch.device] = None
) -> Dict[str, List[float]]:
    """Train the RootCauseHGNN with link prediction loss.

    Builds the homogeneous representation, creates negative samples, and
    trains with BCE loss on positive vs. negative causal edges.

    Args:
        model: RootCauseHGNN model instance.
        graph_data: Heterogeneous graph with ground-truth causal edges.
        config: CONFIG dict with training hyperparameters ('hgnn_epochs',
            'hgnn_lr', 'seed', 'checkpoint_dir').
        device: Target torch device (default: from config or CPU).

    Returns:
        Dictionary with training history:
            - 'train_loss': List of per-epoch loss values.
            - 'lr': List of per-epoch learning rates.
    """
    if device is None:
        device = torch.device(config.get('device', 'cpu'))
        if device.type == 'cuda' and not torch.cuda.is_available():
            device = torch.device('cpu')

    model = model.to(device)

    # Convert heterogeneous graph to homogeneous
    x, edge_index, edge_type, node_offsets = convert_hetero_to_homo(graph_data, device)
    num_total_nodes = x.shape[0]

    print(f"[HGNN Train] Total nodes: {num_total_nodes}, Total edges: {edge_index.shape[1]}")
    print(f"[HGNN Train] Node offsets: {node_offsets}")

    # Extract positive causal edges (PRECEDED_BY relation in global indices)
    failure_offset, num_failures = node_offsets.get('failure', (0, 0))
    maint_offset, num_maint = node_offsets.get('maintenance', (0, 0))

    # Get PRECEDED_BY edges in global coordinates
    preceded_by_rel_idx = 3  # PRECEDED_BY
    preceded_mask = edge_type == preceded_by_rel_idx
    pos_edge_index = edge_index[:, preceded_mask]  # [2, P]

    if pos_edge_index.shape[1] == 0:
        print("[WARN] No positive causal edges found. Generating synthetic ones.")
        num_synthetic = min(50, num_failures * 3)
        rng = np.random.RandomState(config.get('seed', 42))
        syn_src = torch.tensor(
            [failure_offset + rng.randint(0, max(1, num_failures))
             for _ in range(num_synthetic)], dtype=torch.long
        )
        syn_dst = torch.tensor(
            [maint_offset + rng.randint(0, max(1, num_maint))
             for _ in range(num_synthetic)], dtype=torch.long
        )
        pos_edge_index = torch.stack([syn_src, syn_dst]).to(device)

    num_pos = pos_edge_index.shape[1]
    num_neg = num_pos * 3  # 3:1 negative to positive ratio

    print(f"[HGNN Train] Positive causal edges: {num_pos}")
    print(f"[HGNN Train] Negative samples per epoch: {num_neg}")
    print(f"[HGNN Train] Maintenance nodes: {num_maint}")
    print(f"[HGNN Train] Failure nodes: {num_failures}")
    
    if num_pos <= num_failures:
        print(f"[WARN] Positive causal edges ({num_pos}) <= Failure nodes ({num_failures}). Graph may lack sufficient causal links.")

    # Optimizer setup
    epochs = config.get('hgnn_epochs', 50)
    lr = config.get('hgnn_lr', 1e-3)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=0.01,
        betas=(0.9, 0.999),
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=lr * 0.01
    )

    # Mixed precision scaler
    use_amp = device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    # Training history
    history: Dict[str, List[float]] = {'train_loss': [], 'lr': []}

    model.train()

    pbar = tqdm(range(epochs), desc="HGNN Training", unit="epoch")
    for epoch in pbar:
        # Re-sample negatives each epoch for diversity
        neg_edge_index = sample_negative_edges(
            pos_edge_index.cpu(),
            num_nodes=num_total_nodes,
            num_neg_samples=num_neg,
            seed=config.get('seed', 42) + epoch,
        ).to(device)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast('cuda', enabled=use_amp):
            # Forward pass: get node embeddings
            node_embs = model(x, edge_index, edge_type)  # [N, out_channels]

            # Link prediction loss on causal edges
            loss = model.compute_link_loss(
                node_embs, pos_edge_index, neg_edge_index
            )

        scaler.scale(loss).backward()

        # Gradient clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        loss_val = loss.item()
        current_lr = optimizer.param_groups[0]['lr']
        history['train_loss'].append(loss_val)
        history['lr'].append(current_lr)

        pbar.set_postfix({
            'loss': f'{loss_val:.4f}',
            'lr': f'{current_lr:.2e}',
        })

    # Save checkpoint
    checkpoint_dir = config.get('checkpoint_dir', './checkpoints/')
    import os
    os.makedirs(checkpoint_dir, exist_ok=True)
    ckpt_path = os.path.join(checkpoint_dir, 'hgnn_root_cause.pt')
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': {k: v for k, v in config.items()
                   if isinstance(v, (int, float, str, bool, list))},
        'epoch': epochs,
        'final_loss': history['train_loss'][-1] if history['train_loss'] else float('inf'),
        'node_offsets': node_offsets,
    }, ckpt_path)
    print(f"[HGNN Train] Checkpoint saved to {ckpt_path}")
    print(f"[HGNN Train] Final loss: {history['train_loss'][-1]:.4f}")

    return history


print("[Section 4] train_hgnn function defined.")

# %%
# Cell 4.6 — HGNN Evaluation
# Evaluates root cause prediction accuracy with Top-1 and Top-5 metrics,
# prints a detailed results table, and visualizes sample causal chains.

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def evaluate_hgnn(
    model: RootCauseHGNN,
    test_graph: HeteroData,
    config: Optional[Dict[str, Any]] = None,
    device: Optional[torch.device] = None,
    top_k_values: List[int] = [1, 3, 5],
    verbose: bool = True
) -> Dict[str, float]:
    """Evaluate HGNN root cause prediction accuracy.

    For each failure node in the test graph, queries the model to rank all
    maintenance nodes by causal score and checks whether the true cause
    appears in the top-k predictions.

    Args:
        model: Trained RootCauseHGNN model.
        test_graph: HeteroData graph with ground-truth causal edges.
        config: CONFIG dict (optional, used for device/paths).
        device: Target torch device.
        top_k_values: List of k values for top-k accuracy (default [1, 3, 5]).
        verbose: Whether to print detailed results (default True).

    Returns:
        Dictionary with evaluation metrics:
            - 'top_1_acc': Top-1 accuracy.
            - 'top_3_acc': Top-3 accuracy.
            - 'top_5_acc': Top-5 accuracy.
            - 'mrr': Mean Reciprocal Rank.
            - 'mean_rank': Mean rank of true cause.
    """
    if config is None:
        config = {}
    if device is None:
        device = torch.device(config.get('device', 'cpu'))
        if device.type == 'cuda' and not torch.cuda.is_available():
            device = torch.device('cpu')

    model = model.to(device)
    model.eval()

    # Convert to homogeneous
    x, edge_index, edge_type, node_offsets = convert_hetero_to_homo(test_graph, device)

    failure_offset, num_failures = node_offsets.get('failure', (0, 0))
    maint_offset, num_maint = node_offsets.get('maintenance', (0, 0))

    if num_failures == 0 or num_maint == 0:
        print("[WARN] No failure or maintenance nodes in test graph.")
        return {'top_1_acc': 0.0, 'top_3_acc': 0.0, 'top_5_acc': 0.0,
                'mrr': 0.0, 'mean_rank': float('inf')}

    # Get ground truth causal links
    gt_links: Dict[int, List[int]] = defaultdict(list)
    if hasattr(test_graph, '_failure_maint_links'):
        for f_idx, m_idx in test_graph._failure_maint_links:
            gt_links[f_idx].append(m_idx)
    else:
        # Extract from PRECEDED_BY edges
        preceded_mask = edge_type == 3
        preceded_edges = edge_index[:, preceded_mask]
        for i in range(preceded_edges.shape[1]):
            f_global = preceded_edges[0, i].item()
            m_global = preceded_edges[1, i].item()
            f_local = f_global - failure_offset
            m_local = m_global - maint_offset
            if 0 <= f_local < num_failures and 0 <= m_local < num_maint:
                gt_links[f_local].append(m_local)

    with torch.no_grad():
        node_embs = model(x, edge_index, edge_type)  # [N, D]

    # Candidate maintenance embeddings (all maintenance nodes)
    maint_embs = node_embs[maint_offset:maint_offset + num_maint]  # [M, D]

    # Evaluate per failure node
    top_k_hits = {k: 0 for k in top_k_values}
    reciprocal_ranks = []
    ranks = []
    per_failure_results = []

    for f_idx in range(num_failures):
        if f_idx not in gt_links or len(gt_links[f_idx]) == 0:
            continue

        f_global = failure_offset + f_idx
        query_emb = node_embs[f_global]  # [D]

        with torch.no_grad():
            scores = model.predict_cause(query_emb, maint_embs)  # [M]

        # Rank candidates by score (descending)
        sorted_indices = torch.argsort(scores, descending=True).cpu().numpy()
        true_causes = set(gt_links[f_idx])

        # Find best rank of any true cause
        best_rank = num_maint + 1
        for rank_pos, m_idx in enumerate(sorted_indices):
            if m_idx in true_causes:
                best_rank = rank_pos + 1  # 1-indexed
                break

        ranks.append(best_rank)
        reciprocal_ranks.append(1.0 / best_rank)

        for k in top_k_values:
            top_k_set = set(sorted_indices[:k].tolist())
            if true_causes & top_k_set:
                top_k_hits[k] += 1

        # Store for detailed reporting
        top5_preds = sorted_indices[:5].tolist()
        top5_scores_vals = scores[sorted_indices[:5]].cpu().numpy().tolist()
        per_failure_results.append({
            'failure_idx': f_idx,
            'true_causes': list(true_causes),
            'best_rank': best_rank,
            'top5_preds': top5_preds,
            'top5_scores': top5_scores_vals,
        })

    num_evaluated = len(ranks)

    if num_evaluated == 0:
        print("[WARN] No evaluatable failure nodes with ground truth links.")
        return {'top_1_acc': 0.0, 'top_3_acc': 0.0, 'top_5_acc': 0.0,
                'mrr': 0.0, 'mean_rank': float('inf')}

    # Compute metrics
    metrics = {}
    for k in top_k_values:
        metrics[f'top_{k}_acc'] = top_k_hits[k] / num_evaluated
    metrics['mrr'] = np.mean(reciprocal_ranks)
    metrics['mean_rank'] = np.mean(ranks)

    if verbose:
        print("\n" + "=" * 70)
        print("  ROOT CAUSE HGNN — EVALUATION RESULTS")
        print("=" * 70)
        print(f"  Evaluated failure nodes: {num_evaluated}/{num_failures}")
        print(f"  Candidate pool size:     {num_maint} maintenance nodes")
        print("-" * 70)
        print(f"  {'Metric':<25} {'Value':>10} {'Target':>10} {'Status':>10}")
        print("-" * 70)

        target_map = {'top_1_acc': 0.75, 'top_3_acc': None, 'top_5_acc': 0.92}
        for key, val in metrics.items():
            target = target_map.get(key, None)
            if target is not None:
                status = "✓ PASS" if val >= target else "✗ FAIL"
                print(f"  {key:<25} {val:>10.4f} {target:>10.2f} {status:>10}")
            else:
                print(f"  {key:<25} {val:>10.4f} {'—':>10} {'—':>10}")

        print("-" * 70)

        # Detailed per-failure table (show first 10)
        print("\n  Sample Failure Analysis (first 10):")
        print(f"  {'F_idx':>5} {'TrueCause':>10} {'BestRank':>9} {'Top-5 Predictions':>30}")
        print("  " + "-" * 60)
        for result in per_failure_results[:10]:
            tc_str = str(result['true_causes'][:3])
            preds_str = str(result['top5_preds'])
            print(f"  {result['failure_idx']:>5} {tc_str:>10} {result['best_rank']:>9} {preds_str:>30}")

        print("=" * 70)

    # --- Visualize sample causal chains ---
    _visualize_causal_chains(per_failure_results[:6], test_graph, config)

    return metrics


def _visualize_causal_chains(
    results: List[Dict[str, Any]],
    graph: HeteroData,
    config: Dict[str, Any]
) -> None:
    """Visualize sample causal chains as a ranked bar chart.

    Creates a multi-panel figure showing the top-5 ranked causes for each
    sample failure event, with the true cause highlighted.

    Args:
        results: List of per-failure result dicts from evaluate_hgnn.
        graph: The test HeteroData graph (for metadata).
        config: CONFIG dict (for figure save path).
    """
    if not results:
        return

    num_panels = min(len(results), 6)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    failure_cats = FAILURE_CATEGORIES if 'FAILURE_CATEGORIES' in dir() else [
        'rail_fracture', 'gauge_deviation', 'thermal_buckling',
        'ballast_degradation', 'weld_failure', 'sleeper_damage',
        'drainage_failure', 'subgrade_settlement'
    ]

    for i in range(num_panels):
        ax = axes[i]
        result = results[i]
        true_set = set(result['true_causes'])

        labels = [f"M-{idx}" for idx in result['top5_preds']]
        scores = result['top5_scores']
        colors = ['#e74c3c' if idx in true_set else '#3498db'
                  for idx in result['top5_preds']]

        bars = ax.barh(range(len(labels)), scores, color=colors, edgecolor='white', height=0.6)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlabel('Causal Score', fontsize=9)
        ax.set_title(f'Failure #{result["failure_idx"]} (rank={result["best_rank"]})',
                     fontsize=10, fontweight='bold')
        ax.invert_yaxis()
        ax.set_xlim(0, 1.05)
        ax.grid(axis='x', alpha=0.3)

    # Hide unused panels
    for i in range(num_panels, len(axes)):
        axes[i].set_visible(False)

    # Legend
    true_patch = mpatches.Patch(color='#e74c3c', label='True Cause')
    pred_patch = mpatches.Patch(color='#3498db', label='Predicted Candidate')
    fig.legend(handles=[true_patch, pred_patch], loc='upper center',
               ncol=2, fontsize=11, framealpha=0.9)

    plt.suptitle('Root Cause Analysis — Top-5 Ranked Causes per Failure',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    # Save figure
    figures_dir = config.get('figures_dir', './figures/')
    import os
    os.makedirs(figures_dir, exist_ok=True)
    fig_path = os.path.join(figures_dir, 'hgnn_causal_chains.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"[HGNN Eval] Figure saved to {fig_path}")


print("[Section 4] evaluate_hgnn function defined.")

# %%
# Cell 4.7 — Section 4 Checkpoint
# Build graph, instantiate model, train, evaluate, and save final checkpoint.

def run_section_4_checkpoint(config: Dict[str, Any]) -> Dict[str, Any]:
    """Execute the full Section 4 pipeline: build, train, evaluate.

    Args:
        config: Global CONFIG dict with all hyperparameters.

    Returns:
        Dictionary containing all Section 4 artifacts:
            - 'graph': Built HeteroData graph.
            - 'model': Trained RootCauseHGNN model.
            - 'history': Training loss history.
            - 'metrics': Evaluation metrics dict.
    """
    print("=" * 70)
    print("  SECTION 4 CHECKPOINT — Root Cause HGNN")
    print("=" * 70)

    # Determine device
    device = torch.device(config.get('device', 'cpu'))
    if device.type == 'cuda' and not torch.cuda.is_available():
        device = torch.device('cpu')
    print(f"[Device] Using: {device}")

    # Set seed
    seed = config.get('seed', 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Step 1: Build heterogeneous graph
    print("\n[Step 1/4] Building heterogeneous knowledge graph...")
    num_sections = config.get('num_sections', 50)
    builder = HeterogeneousGraphBuilder(config=config, feature_dim=128)
    graph = builder.build_synthetic_graph(
        num_sections=num_sections,
        num_sensors_per_section=5,
        num_events=num_sections * 4,
    )

    # Inject additional known causes for richer training signal
    rng = np.random.RandomState(seed)
    num_extra = min(20, graph._num_failures)
    extra_pairs = []
    for _ in range(num_extra):
        f_idx = rng.randint(0, graph._num_failures)
        m_idx = rng.randint(0, graph._num_maintenance)
        extra_pairs.append((f_idx, m_idx))
    graph = builder.inject_known_causes(graph, extra_pairs)

    # Step 2: Instantiate model
    print("\n[Step 2/4] Instantiating RootCauseHGNN model...")
    hgnn_model = RootCauseHGNN(
        in_channels=128,
        hidden_channels=config.get('hgnn_hidden', 128),
        out_channels=64,
        num_relations=config.get('num_relation_types', 4),
        num_layers=config.get('hgnn_layers', 4),
        dropout=config.get('dropout', 0.2),
    )
    total_params = sum(p.numel() for p in hgnn_model.parameters())
    trainable_params = sum(p.numel() for p in hgnn_model.parameters() if p.requires_grad)
    print(f"  Total parameters:     {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    # Step 3: Train
    print("\n[Step 3/4] Training HGNN...")
    history = train_hgnn(hgnn_model, graph, config, device=device)

    # Step 4: Evaluate
    print("\n[Step 4/4] Evaluating HGNN...")
    metrics = evaluate_hgnn(hgnn_model, graph, config, device=device)

    # Summary
    print("\n" + "=" * 70)
    print("  SECTION 4 CHECKPOINT — COMPLETE")
    print(f"  Final Training Loss:  {history['train_loss'][-1]:.4f}")
    print(f"  Top-1 Accuracy:       {metrics.get('top_1_acc', 0):.4f}")
    print(f"  Top-5 Accuracy:       {metrics.get('top_5_acc', 0):.4f}")
    print(f"  MRR:                  {metrics.get('mrr', 0):.4f}")
    print("=" * 70)

    return {
        'graph': graph,
        'model': hgnn_model,
        'history': history,
        'metrics': metrics,
    }


# Execute checkpoint
# section_4_results = run_section_4_checkpoint(CONFIG)
# hgnn_model = section_4_results['model']

print("\n[Section 4] ✓ All cells defined. Call run_section_4_checkpoint(CONFIG) to execute.")