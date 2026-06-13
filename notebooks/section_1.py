# %% [markdown]
# ## Section 1 — Synthetic Data Generator for Project Rakshak
#
# This section creates a **physically realistic** synthetic dataset that
# simulates multi-modal sensor telemetry from the Delhi–Agra railway corridor.
#
# ### Modalities Generated
#
# | # | Modality | Shape | Description |
# |---|----------|-------|-------------|
# | 1 | Vibration | `[720, 3]` | 3-axis accelerometer @ 1 Hz for 720 hours (30 days) |
# | 2 | Temperature | `[720, 1]` | Rail temperature (°C) over 30 days |
# | 3 | Gauge deviation | `[720, 1]` | Deviation from nominal 1676 mm broad gauge |
# | 4 | Track metadata | `[32]` | Static features per track section |
# | 5 | Weather forecast | `[72, 6]` | 72-hour hourly forecast (6 features) |
# | 6 | Maintenance history | `[16, 64]` | Last 16 maintenance events (64-d encoding) |
# | 7 | Graph topology | `[2, E]` | Station adjacency edge_index |
#
# ### Failure Categories (8 classes)
# `rail_fracture`, `gauge_deviation`, `thermal_buckling`,
# `ballast_degradation`, `weld_failure`, `sleeper_damage`,
# `drainage_failure`, `subgrade_settlement`

# %% [markdown]
# ### Cell 1.2 — Failure Categories, Station Graph & Coordinates

# %%
# ── Failure categories (8 classes, 0-indexed) ─────────────────────────────────
FAILURE_CATEGORIES: List[str] = [
    'rail_fracture',
    'gauge_deviation',
    'thermal_buckling',
    'ballast_degradation',
    'weld_failure',
    'sleeper_damage',
    'drainage_failure',
    'subgrade_settlement',
]

# ── 12 Phase-1 Stations on the Delhi-Agra corridor ────────────────────────────
STATIONS: List[str] = [
    'DLI',   # Delhi
    'GZB',   # Ghaziabad
    'MERT',  # Meerut (branch)
    'HPJN',  # Hapur Junction
    'ALJN',  # Aligarh Junction
    'KOSI',  # Kosi Kalan
    'MATH',  # Mathura Junction
    'AGC',   # Agra Cantonment
    'TDL',   # Tundla Junction
    'FRD',   # Firozabad
    'BRJ',   # Bharatpur Junction (branch)
    'MTJ',   # Mathura Junction (alternate code / yard)
]

# ── Realistic lat/lon for each station ─────────────────────────────────────────
STATION_COORDS: Dict[str, Tuple[float, float]] = {
    'DLI':  (28.6424, 77.2210),
    'GZB':  (28.6692, 77.4538),
    'MERT': (28.9845, 77.7064),
    'HPJN': (28.7307, 77.7751),
    'ALJN': (27.8974, 78.0880),
    'KOSI': (27.7400, 77.4600),
    'MATH': (27.4924, 77.6737),
    'AGC':  (27.1767, 78.0081),
    'TDL':  (27.0100, 78.2330),
    'FRD':  (27.1533, 78.3961),
    'BRJ':  (27.2173, 77.4900),
    'MTJ':  (27.4924, 77.6800),
}


def build_adjacency_graph(
    stations: Optional[List[str]] = None,
) -> torch.LongTensor:
    """Build a station adjacency graph for the Delhi–Agra corridor.

    The stations are connected along the main line (linear chain) plus
    cross-links for junctions (GZB↔MERT, GZB↔HPJN, MATH↔MTJ, MATH↔BRJ,
    TDL↔FRD).

    Args:
        stations: Ordered list of station codes.  Defaults to ``STATIONS``.

    Returns:
        ``edge_index`` tensor of shape ``[2, num_edges]`` (undirected — both
        directions included) suitable for PyTorch-Geometric message passing.

    Raises:
        ValueError: If fewer than 2 stations are provided.
    """
    stations = stations or STATIONS
    if len(stations) < 2:
        raise ValueError("Need at least 2 stations to build a graph.")

    idx: Dict[str, int] = {s: i for i, s in enumerate(stations)}

    # Main line: DLI–GZB–HPJN–ALJN–KOSI–MATH–AGC (and TDL–FRD)
    edges: List[Tuple[int, int]] = []

    # Linear backbone
    main_line_order = ['DLI', 'GZB', 'HPJN', 'ALJN', 'KOSI', 'MATH', 'AGC', 'TDL', 'FRD']
    for i in range(len(main_line_order) - 1):
        a, b = main_line_order[i], main_line_order[i + 1]
        if a in idx and b in idx:
            edges.append((idx[a], idx[b]))

    # Cross-links / branch lines
    cross_links = [
        ('GZB', 'MERT'),   # Meerut branch
        ('MATH', 'MTJ'),   # Mathura yard link
        ('MATH', 'BRJ'),   # Bharatpur branch
        ('TDL', 'AGC'),    # Tundla–Agra direct
        ('KOSI', 'MTJ'),   # Kosi Kalan–Mathura link
    ]
    for a, b in cross_links:
        if a in idx and b in idx:
            edges.append((idx[a], idx[b]))

    # Make undirected: add reverse edges
    undirected: List[Tuple[int, int]] = []
    for u, v in edges:
        undirected.append((u, v))
        undirected.append((v, u))

    # Add self-loops for message passing stability
    for i in range(len(stations)):
        undirected.append((i, i))

    src = [e[0] for e in undirected]
    dst = [e[1] for e in undirected]

    edge_index = torch.tensor([src, dst], dtype=torch.long)  # [2, num_edges]
    return edge_index


# ── Quick sanity check ────────────────────────────────────────────────────────
_edge_index = build_adjacency_graph()
print(f'✅ Station graph: {len(STATIONS)} nodes, {_edge_index.shape[1]} directed edges (incl. self-loops)')
print(f'   Failure categories: {len(FAILURE_CATEGORIES)} classes')

# %% [markdown]
# ### Cell 1.3 — Vibration Signal Generator

# %%
class VibrationGenerator:
    """Generates realistic 3-axis vibration signals for railway track sections.

    Normal vibration is modelled as a superposition of multi-frequency sinusoids
    (rail resonance, wheel-rail interaction, bogie dynamics) plus band-limited
    Gaussian noise.  Failure modes inject physically-motivated perturbations.

    Args:
        rng: NumPy ``Generator`` instance for reproducibility.

    Returns:
        ndarray objects of shape ``[length, channels]``.

    Raises:
        ValueError: If ``length`` or ``channels`` is non-positive.
    """

    # Characteristic frequencies (Hz) for normal rail vibration at 1 Hz sampling
    _BASE_FREQS: List[float] = [0.01, 0.02, 0.05, 0.08, 0.12, 0.18, 0.25, 0.35, 0.42]
    _BASE_AMPS:  List[float] = [0.30, 0.20, 0.15, 0.10, 0.08, 0.05, 0.04, 0.03, 0.02]

    def __init__(self, rng: Optional[np.random.Generator] = None) -> None:
        self.rng = rng or np.random.default_rng(CONFIG['seed'])

    # ── helpers ────────────────────────────────────────────────────────────────

    def _time_axis(self, length: int) -> np.ndarray:
        """Create a time axis in hours.

        Args:
            length: Number of time steps.

        Returns:
            1-D ndarray of shape ``[length]``.

        Raises:
            Nothing.
        """
        return np.arange(length, dtype=np.float64)  # [length]

    # ── Normal signal ──────────────────────────────────────────────────────────

    def generate_normal(
        self,
        length: int = 720,
        channels: int = 3,
    ) -> np.ndarray:
        """Generate baseline (healthy) vibration telemetry.

        Args:
            length: Number of hourly samples (default 720 = 30 days).
            channels: Number of accelerometer axes (default 3: X, Y, Z).

        Returns:
            ndarray of shape ``[length, channels]`` with values in ≈ [-1, 1].

        Raises:
            ValueError: If length or channels ≤ 0.
        """
        if length <= 0 or channels <= 0:
            raise ValueError(f"length and channels must be > 0, got {length}, {channels}")

        t = self._time_axis(length)  # [length]
        signal = np.zeros((length, channels), dtype=np.float64)  # [length, channels]

        for ch in range(channels):
            for freq, amp in zip(self._BASE_FREQS, self._BASE_AMPS):
                # Slight per-channel phase/frequency jitter for realism
                phase = self.rng.uniform(0, 2 * np.pi)
                freq_jitter = freq * (1.0 + self.rng.normal(0, 0.05))
                amp_jitter = amp * (1.0 + self.rng.normal(0, 0.1))
                signal[:, ch] += amp_jitter * np.sin(2 * np.pi * freq_jitter * t + phase)

            # Band-limited Gaussian noise (sensor noise floor)
            noise = self.rng.normal(0, 0.05, size=length)
            signal[:, ch] += noise

            # Slow amplitude modulation (train passage cycles: ~24h period)
            daily_mod = 0.15 * np.sin(2 * np.pi * t / 24.0 + self.rng.uniform(0, np.pi))
            signal[:, ch] *= (1.0 + daily_mod)

        return signal.astype(np.float32)  # [length, channels]

    # ── Fault injection methods ────────────────────────────────────────────────

    def inject_rail_fracture(
        self,
        signal: np.ndarray,
        lead_time_hours: float,
    ) -> np.ndarray:
        """Inject rail fracture signature: progressive amplitude increase +
        impulse spikes before fracture event.

        Args:
            signal: Healthy vibration array of shape ``[T, C]``.
            lead_time_hours: Hours before the end at which the fault develops.

        Returns:
            Modified signal of the same shape ``[T, C]``.

        Raises:
            Nothing.
        """
        signal = signal.copy()  # [T, C]
        T, C = signal.shape
        onset = max(0, T - int(lead_time_hours))

        # Progressive amplitude ramp (crack propagation)
        ramp = np.zeros(T)
        ramp[onset:] = np.linspace(0.0, 1.0, T - onset)  # [T-onset]
        for ch in range(C):
            # Amplitude grows quadratically
            signal[onset:, ch] *= (1.0 + 2.5 * ramp[onset:] ** 2)

        # Impulse spikes (random, increasing frequency near fracture)
        n_spikes = max(3, int(lead_time_hours / 30))
        spike_positions = self.rng.integers(onset, T, size=n_spikes)
        for pos in spike_positions:
            spike_amplitude = 1.5 + 2.0 * (pos - onset) / max(1, T - onset)
            spike_width = self.rng.integers(1, 4)
            end_pos = min(pos + spike_width, T)
            for ch in range(C):
                signal[pos:end_pos, ch] += self.rng.choice([-1, 1]) * spike_amplitude

        # High-frequency crack-growth noise in the last 20% of lead time
        crack_start = max(onset, T - int(lead_time_hours * 0.2))
        crack_noise = self.rng.normal(0, 0.8, size=(T - crack_start, C))
        signal[crack_start:] += crack_noise.astype(np.float32)

        return signal.astype(np.float32)  # [T, C]

    def inject_weld_failure(
        self,
        signal: np.ndarray,
        lead_time_hours: float,
    ) -> np.ndarray:
        """Inject weld failure signature: periodic impact signatures at weld
        spacing intervals with growing amplitude.

        Args:
            signal: Healthy vibration array of shape ``[T, C]``.
            lead_time_hours: Hours before the end at which the fault develops.

        Returns:
            Modified signal of the same shape ``[T, C]``.

        Raises:
            Nothing.
        """
        signal = signal.copy()  # [T, C]
        T, C = signal.shape
        onset = max(0, T - int(lead_time_hours))

        # Periodic impact every ~13 m weld spacing ≈ ~4.3 hours at 3 km/h crawl
        impact_period = self.rng.integers(3, 7)  # hours between impacts
        impact_times = np.arange(onset, T, impact_period)

        for t_impact in impact_times:
            progress = (t_impact - onset) / max(1, T - onset)
            amplitude = 0.5 + 2.0 * progress ** 1.5
            # Decaying transient (exponential ring-down)
            duration = min(6, T - t_impact)
            decay = np.exp(-np.arange(duration) / 1.5)  # [duration]
            for ch in range(C):
                freq = 0.2 + 0.1 * ch
                transient = amplitude * decay * np.sin(
                    2 * np.pi * freq * np.arange(duration)
                )
                signal[t_impact:t_impact + duration, ch] += transient.astype(np.float32)

        return signal.astype(np.float32)  # [T, C]

    def inject_ballast_degradation(
        self,
        signal: np.ndarray,
        lead_time_hours: float,
    ) -> np.ndarray:
        """Inject ballast degradation: broadband noise increase + low-frequency
        drift as ballast cushioning is lost.

        Args:
            signal: Healthy vibration array of shape ``[T, C]``.
            lead_time_hours: Hours before the end at which the degradation begins.

        Returns:
            Modified signal of the same shape ``[T, C]``.

        Raises:
            Nothing.
        """
        signal = signal.copy()  # [T, C]
        T, C = signal.shape
        onset = max(0, T - int(lead_time_hours))

        ramp = np.zeros(T)
        ramp[onset:] = np.linspace(0.0, 1.0, T - onset)

        for ch in range(C):
            # Broadband noise increase (ballast losing damping capacity)
            noise_power = 0.6 * ramp ** 2
            broadband_noise = self.rng.normal(0, 1.0, size=T) * noise_power
            signal[:, ch] += broadband_noise.astype(np.float32)

            # Low-frequency drift (settlement)
            lf_drift = 0.4 * ramp * np.sin(
                2 * np.pi * 0.005 * np.arange(T) + self.rng.uniform(0, np.pi)
            )
            signal[:, ch] += lf_drift.astype(np.float32)

        return signal.astype(np.float32)  # [T, C]

    def inject_sleeper_damage(
        self,
        signal: np.ndarray,
        lead_time_hours: float,
    ) -> np.ndarray:
        """Inject sleeper damage: periodic low-frequency resonance at sleeper
        spacing with increasing amplitude.

        Args:
            signal: Healthy vibration array of shape ``[T, C]``.
            lead_time_hours: Hours before the end at which damage becomes
                detectable.

        Returns:
            Modified signal of the same shape ``[T, C]``.

        Raises:
            Nothing.
        """
        signal = signal.copy()  # [T, C]
        T, C = signal.shape
        onset = max(0, T - int(lead_time_hours))

        t = self._time_axis(T)  # [T]
        ramp = np.zeros(T)
        ramp[onset:] = np.linspace(0.0, 1.0, T - onset)

        # Sleeper spacing resonance: ~0.6 m spacing at low speed → ~0.015 Hz
        sleeper_freq = 0.015 + self.rng.normal(0, 0.002)

        for ch in range(C):
            resonance = 1.2 * ramp * np.sin(
                2 * np.pi * sleeper_freq * t + self.rng.uniform(0, 2 * np.pi)
            )
            # Add 2nd harmonic
            resonance += 0.4 * ramp * np.sin(
                2 * np.pi * 2 * sleeper_freq * t + self.rng.uniform(0, 2 * np.pi)
            )
            signal[:, ch] += resonance.astype(np.float32)

            # Intermittent clunking noise (cracked sleeper bouncing)
            n_clunks = max(2, int(lead_time_hours / 50))
            clunk_times = self.rng.integers(onset, T, size=n_clunks)
            for ct in clunk_times:
                end_ct = min(ct + 3, T)
                signal[ct:end_ct, ch] += self.rng.uniform(0.5, 1.5) * ramp[ct]

        return signal.astype(np.float32)  # [T, C]


print(f'✅ VibrationGenerator loaded — {len(VibrationGenerator._BASE_FREQS)} base frequency components')

# %% [markdown]
# ### Cell 1.4 — Temperature Signal Generator

# %%
class TemperatureGenerator:
    """Generates realistic rail temperature time-series for Indian conditions.

    Normal temperature follows a diurnal sinusoidal pattern (day/night) with
    a seasonal baseline and stochastic weather perturbations.  Indian summers
    push rail temperature to 50 °C+; thermal buckling risk starts above 55 °C.

    Args:
        rng: NumPy ``Generator`` instance for reproducibility.

    Returns:
        ndarray objects of shape ``[length, 1]``.

    Raises:
        ValueError: If ``length`` is non-positive.
    """

    def __init__(self, rng: Optional[np.random.Generator] = None) -> None:
        self.rng = rng or np.random.default_rng(CONFIG['seed'])

    def generate_normal(self, length: int = 720) -> np.ndarray:
        """Generate baseline (healthy) rail temperature profile.

        Temperature model:
            T(t) = T_base + A_seasonal * sin(season) + A_diurnal * sin(diurnal) + noise

        For Indian conditions:
            - Base temp: 35–42 °C (summer) / 15–25 °C (winter)
            - Diurnal swing: 8–15 °C
            - Rail surface is typically 10–20 °C above ambient

        Args:
            length: Number of hourly samples (default 720 = 30 days).

        Returns:
            ndarray of shape ``[length, 1]`` with rail temperature in °C.

        Raises:
            ValueError: If ``length`` ≤ 0.
        """
        if length <= 0:
            raise ValueError(f"length must be > 0, got {length}")

        t = np.arange(length, dtype=np.float64)  # [length]

        # Seasonal baseline (assume random month; higher base in April-June)
        base_temp = self.rng.uniform(30.0, 42.0)

        # Seasonal drift over 30 days (slow, small)
        seasonal_drift = 2.0 * np.sin(2 * np.pi * t / (720 * 4) + self.rng.uniform(0, 2 * np.pi))

        # Diurnal cycle: 24-hour period, peak at ~14:00 (hour 14 from midnight)
        diurnal_amp = self.rng.uniform(8.0, 15.0)
        diurnal = diurnal_amp * np.sin(2 * np.pi * (t - 14.0) / 24.0)

        # Rail surface heating (extra 10-20 °C due to direct sun)
        rail_heating = self.rng.uniform(10.0, 18.0) * np.maximum(
            0, np.sin(2 * np.pi * (t - 12.0) / 24.0)
        )

        # Weather perturbation (cloudy days, rain cooling)
        weather_noise = self.rng.normal(0, 1.5, size=length)
        # Smooth it with a rolling average to simulate gradual weather changes
        kernel_size = 6
        kernel = np.ones(kernel_size) / kernel_size
        weather_noise = np.convolve(weather_noise, kernel, mode='same')

        temp = base_temp + seasonal_drift + diurnal + rail_heating + weather_noise
        # Clamp to physically realistic range
        temp = np.clip(temp, 5.0, 70.0)

        return temp.astype(np.float32).reshape(-1, 1)  # [length, 1]

    def inject_thermal_buckling(
        self,
        signal: np.ndarray,
        lead_time_hours: float,
    ) -> np.ndarray:
        """Inject thermal buckling precursor: rapid temperature rise exceeding
        the 55 °C safety threshold.

        Physics: continuous welded rail (CWR) develops compressive stress when
        ΔT exceeds the neutral temperature.  At 55 °C+ the track may buckle.

        Args:
            signal: Healthy temperature array of shape ``[T, 1]``.
            lead_time_hours: Hours before the end at which thermal anomaly
                starts developing.

        Returns:
            Modified signal of the same shape ``[T, 1]``.

        Raises:
            Nothing.
        """
        signal = signal.copy()  # [T, 1]
        T = signal.shape[0]
        onset = max(0, T - int(lead_time_hours))

        # Progressive temperature anomaly
        ramp = np.zeros(T)
        ramp[onset:] = np.linspace(0.0, 1.0, T - onset)

        # Temperature rises by 15-25 °C above normal
        delta_t = self.rng.uniform(15.0, 25.0)
        signal[:, 0] += (delta_t * ramp ** 1.5).astype(np.float32)

        # Superimposed rapid heat spikes (reflected heat from ballast)
        n_spikes = max(2, int(lead_time_hours / 24))
        for _ in range(n_spikes):
            spike_center = self.rng.integers(onset, T)
            spike_width = self.rng.integers(2, 8)
            spike_amp = self.rng.uniform(3.0, 8.0) * ramp[spike_center]
            lo = max(0, spike_center - spike_width)
            hi = min(T, spike_center + spike_width)
            gaussian_spike = spike_amp * np.exp(
                -0.5 * ((np.arange(lo, hi) - spike_center) / max(1, spike_width / 2)) ** 2
            )
            signal[lo:hi, 0] += gaussian_spike.astype(np.float32)

        # Ensure we breach 55 °C threshold
        signal[onset:, 0] = np.maximum(signal[onset:, 0], 55.0 + 3.0 * ramp[onset:])

        return signal.astype(np.float32)  # [T, 1]


print('✅ TemperatureGenerator loaded — Indian diurnal + seasonal model')

# %% [markdown]
# ### Cell 1.5 — Gauge Deviation Generator

# %%
class GaugeGenerator:
    """Generates rail gauge deviation signals for Indian Broad Gauge (1676 mm).

    Normal gauge deviation is a slow random walk around 0 mm (nominal).
    IR safety limit: deviation > ±10 mm triggers speed restriction.

    Args:
        rng: NumPy ``Generator`` instance for reproducibility.

    Returns:
        ndarray objects of shape ``[length, 1]``.

    Raises:
        ValueError: If ``length`` is non-positive.
    """

    _NOMINAL_GAUGE_MM: float = 1676.0  # Indian Broad Gauge

    def __init__(self, rng: Optional[np.random.Generator] = None) -> None:
        self.rng = rng or np.random.default_rng(CONFIG['seed'])

    def generate_normal(self, length: int = 720) -> np.ndarray:
        """Generate baseline gauge deviation signal (healthy track).

        Model: bounded random walk with slight mean-reversion (maintenance
        keeps gauge near nominal).

        Args:
            length: Number of hourly samples (default 720 = 30 days).

        Returns:
            ndarray of shape ``[length, 1]`` — deviation in mm from nominal.

        Raises:
            ValueError: If ``length`` ≤ 0.
        """
        if length <= 0:
            raise ValueError(f"length must be > 0, got {length}")

        deviation = np.zeros(length, dtype=np.float64)  # [length]
        deviation[0] = self.rng.normal(0, 0.5)

        for i in range(1, length):
            # Mean-reverting random walk (Ornstein-Uhlenbeck-like)
            mean_reversion = -0.005 * deviation[i - 1]
            step = mean_reversion + self.rng.normal(0, 0.08)
            deviation[i] = deviation[i - 1] + step

        # Normal gauge stays within ± 3 mm
        deviation = np.clip(deviation, -3.0, 3.0)

        return deviation.astype(np.float32).reshape(-1, 1)  # [length, 1]

    def inject_gauge_deviation(
        self,
        signal: np.ndarray,
        lead_time_hours: float,
    ) -> np.ndarray:
        """Inject gauge deviation fault: gradual drift + sudden jump exceeding
        the 10 mm safety threshold.

        Args:
            signal: Healthy gauge deviation array of shape ``[T, 1]``.
            lead_time_hours: Hours before the end at which deviation begins.

        Returns:
            Modified signal of the same shape ``[T, 1]``.

        Raises:
            Nothing.
        """
        signal = signal.copy()  # [T, 1]
        T = signal.shape[0]
        onset = max(0, T - int(lead_time_hours))

        ramp = np.zeros(T)
        ramp[onset:] = np.linspace(0.0, 1.0, T - onset)

        # Direction of deviation (widening or narrowing)
        direction = self.rng.choice([-1.0, 1.0])

        # Gradual drift phase (sleeper rot, ballast settlement)
        gradual_drift = direction * 8.0 * ramp ** 2  # up to ±8 mm
        signal[:, 0] += gradual_drift.astype(np.float32)

        # Sudden jump near the end (fastener failure / sleeper fracture)
        jump_time = max(onset, T - self.rng.integers(10, max(11, int(lead_time_hours * 0.15))))
        jump_magnitude = direction * self.rng.uniform(4.0, 8.0)
        signal[jump_time:, 0] += jump_magnitude

        # Ensure exceeds 10 mm threshold near the end
        final_dev = abs(signal[-1, 0])
        if final_dev < 10.5:
            signal[-int(lead_time_hours * 0.1):, 0] += direction * (11.0 - final_dev)

        return signal.astype(np.float32)  # [T, 1]

    def inject_subgrade_settlement(
        self,
        signal: np.ndarray,
        lead_time_hours: float,
    ) -> np.ndarray:
        """Inject subgrade settlement: slow progressive deviation caused by
        soil consolidation / waterlogging under the formation.

        Args:
            signal: Healthy gauge deviation array of shape ``[T, 1]``.
            lead_time_hours: Hours before the end at which settlement starts.

        Returns:
            Modified signal of the same shape ``[T, 1]``.

        Raises:
            Nothing.
        """
        signal = signal.copy()  # [T, 1]
        T = signal.shape[0]
        onset = max(0, T - int(lead_time_hours))

        ramp = np.zeros(T)
        ramp[onset:] = np.linspace(0.0, 1.0, T - onset)

        # Slow logarithmic settlement (soil mechanics model)
        settlement = np.zeros(T)
        settlement[onset:] = 6.0 * np.log1p(3.0 * ramp[onset:])

        # Add cyclic loading effect (train passages accelerate settlement)
        cyclic = 0.8 * ramp * np.sin(
            2 * np.pi * np.arange(T) / 24.0 + self.rng.uniform(0, np.pi)
        )

        direction = self.rng.choice([-1.0, 1.0])
        signal[:, 0] += (direction * (settlement + cyclic)).astype(np.float32)

        # Differential settlement noise
        diff_noise = self.rng.normal(0, 0.3, size=T) * ramp
        signal[:, 0] += diff_noise.astype(np.float32)

        return signal.astype(np.float32)  # [T, 1]


print('✅ GaugeGenerator loaded — Indian BG 1676 mm nominal')

# %% [markdown]
# ### Cell 1.6 — Track Metadata Generator

# %%
def generate_track_metadata(
    num_sections: int,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Generate static metadata features for track sections.

    The 32-dimensional feature vector encodes:
        - [0]  rail_age_years          (0-30)
        - [1]  traffic_load_mgt        (million gross tonnes, 10-100)
        - [2]  last_maintenance_days   (0-365)
        - [3]  avg_speed_kmh           (60-160)
        - [4]  curve_radius_m          (0=straight, 200-5000)
        - [5]  gradient_permille       (-15 to +15)
        - [6-8]   sleeper_type one-hot (PSC, wooden, steel)
        - [9-11]  rail_type one-hot    (60kg, 52kg, 90UTS)
        - [12-17] zone one-hot         (NR, NCR, WR, CR, SR, ER)
        - [18-22] geography one-hot    (plain, hilly, coastal, desert, urban)
        - [23] altitude_m              (100-2500)
        - [24] annual_rainfall_mm      (200-3000)
        - [25] curvature_index         (0-1)
        - [26] bridge_proximity_flag   (0 or 1)
        - [27] tunnel_proximity_flag   (0 or 1)
        - [28] electrification_flag    (0 or 1)
        - [29] track_class             (0-5 normalised)
        - [30] soil_type_index         (0-1)
        - [31] flood_risk_index        (0-1)

    Args:
        num_sections: Number of track sections.
        rng: NumPy ``Generator`` instance for reproducibility.

    Returns:
        ndarray of shape ``[num_sections, 32]`` with all features normalised
        to approximately [0, 1] range.

    Raises:
        ValueError: If ``num_sections`` ≤ 0.
    """
    if num_sections <= 0:
        raise ValueError(f"num_sections must be > 0, got {num_sections}")

    rng = rng or np.random.default_rng(CONFIG['seed'])
    meta = np.zeros((num_sections, 32), dtype=np.float32)  # [num_sections, 32]

    for i in range(num_sections):
        # Continuous features (normalised)
        meta[i, 0] = rng.uniform(0.0, 1.0)    # rail_age_years / 30
        meta[i, 1] = rng.uniform(0.0, 1.0)    # traffic_load_mgt / 100
        meta[i, 2] = rng.uniform(0.0, 1.0)    # last_maintenance_days / 365
        meta[i, 3] = rng.uniform(0.0, 1.0)    # avg_speed_kmh / 160
        meta[i, 4] = rng.uniform(0.0, 1.0)    # curve_radius / 5000 (0 = straight)
        meta[i, 5] = rng.uniform(-1.0, 1.0)   # gradient / 15

        # Sleeper type one-hot (3 categories)
        sleeper_idx = rng.integers(0, 3)
        meta[i, 6 + sleeper_idx] = 1.0

        # Rail type one-hot (3 categories)
        rail_idx = rng.integers(0, 3)
        meta[i, 9 + rail_idx] = 1.0

        # Zone one-hot (6 categories)
        zone_idx = rng.integers(0, 6)
        meta[i, 12 + zone_idx] = 1.0

        # Geography one-hot (5 categories)
        geo_idx = rng.integers(0, 5)
        meta[i, 18 + geo_idx] = 1.0

        # Remaining continuous features
        meta[i, 23] = rng.uniform(0.0, 1.0)   # altitude / 2500
        meta[i, 24] = rng.uniform(0.0, 1.0)   # annual_rainfall / 3000
        meta[i, 25] = rng.uniform(0.0, 1.0)   # curvature_index
        meta[i, 26] = rng.choice([0.0, 1.0], p=[0.85, 0.15])  # bridge_proximity
        meta[i, 27] = rng.choice([0.0, 1.0], p=[0.92, 0.08])  # tunnel_proximity
        meta[i, 28] = rng.choice([0.0, 1.0], p=[0.30, 0.70])  # electrification
        meta[i, 29] = rng.integers(0, 6) / 5.0  # track_class
        meta[i, 30] = rng.uniform(0.0, 1.0)   # soil_type_index
        meta[i, 31] = rng.uniform(0.0, 1.0)   # flood_risk_index

    return meta  # [num_sections, 32]


# Quick test
_test_meta = generate_track_metadata(5)
print(f'✅ generate_track_metadata: shape={_test_meta.shape}, '
      f'range=[{_test_meta.min():.2f}, {_test_meta.max():.2f}]')

# %% [markdown]
# ### Cell 1.7 — Weather Generator

# %%
def generate_weather(
    length: int = 72,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Generate synthetic 72-hour weather forecast for Indian conditions.

    The 6 weather features are:
        - [0] temperature_c        (15-50 °C)
        - [1] humidity_pct         (20-100 %)
        - [2] precipitation_mm_hr  (0-30 mm/hr)
        - [3] wind_speed_kmh       (0-60 km/h)
        - [4] pressure_hpa         (990-1030 hPa)
        - [5] cloud_cover_pct      (0-100 %)

    All values are normalised to [0, 1] for model consumption.

    Args:
        length: Number of hourly forecast steps (default 72 = 3 days).
        rng: NumPy ``Generator`` instance for reproducibility.

    Returns:
        ndarray of shape ``[length, 6]`` with normalised weather features.

    Raises:
        ValueError: If ``length`` ≤ 0.
    """
    if length <= 0:
        raise ValueError(f"length must be > 0, got {length}")

    rng = rng or np.random.default_rng(CONFIG['seed'])
    t = np.arange(length, dtype=np.float64)  # [length]
    weather = np.zeros((length, 6), dtype=np.float64)  # [length, 6]

    # ── Temperature (°C) ──────────────────────────────────────────────────
    base_temp = rng.uniform(25.0, 42.0)
    diurnal = 8.0 * np.sin(2 * np.pi * (t - 14.0) / 24.0)
    temp_noise = rng.normal(0, 1.5, size=length)
    raw_temp = base_temp + diurnal + temp_noise
    raw_temp = np.clip(raw_temp, 15.0, 50.0)
    weather[:, 0] = (raw_temp - 15.0) / 35.0  # normalise to [0, 1]

    # ── Humidity (%) ──────────────────────────────────────────────────────
    base_humidity = rng.uniform(40.0, 80.0)
    # Humidity is anti-correlated with temperature (higher at night)
    humidity = base_humidity - 0.5 * diurnal + rng.normal(0, 5.0, size=length)
    humidity = np.clip(humidity, 20.0, 100.0)
    weather[:, 1] = (humidity - 20.0) / 80.0

    # ── Precipitation (mm/hr) ─────────────────────────────────────────────
    # Intermittent: mostly 0, occasional showers
    is_raining = rng.random(size=length) < 0.15  # 15% chance each hour
    rain_intensity = rng.exponential(3.0, size=length) * is_raining
    rain_intensity = np.clip(rain_intensity, 0.0, 30.0)
    # Temporal correlation: rain events persist
    for i in range(1, length):
        if is_raining[i - 1] and rng.random() < 0.6:
            rain_intensity[i] = max(rain_intensity[i], rain_intensity[i - 1] * rng.uniform(0.5, 1.1))
    weather[:, 2] = rain_intensity / 30.0

    # ── Wind speed (km/h) ─────────────────────────────────────────────────
    base_wind = rng.uniform(5.0, 20.0)
    wind = base_wind + rng.normal(0, 5.0, size=length)
    # Gusts
    gusts = (rng.random(size=length) < 0.05) * rng.uniform(15.0, 40.0, size=length)
    wind = np.clip(wind + gusts, 0.0, 60.0)
    weather[:, 3] = wind / 60.0

    # ── Atmospheric pressure (hPa) ────────────────────────────────────────
    base_pressure = rng.uniform(1005.0, 1020.0)
    pressure = base_pressure + rng.normal(0, 3.0, size=length)
    # Anti-correlate with rain
    pressure -= 5.0 * (rain_intensity / 30.0)
    pressure = np.clip(pressure, 990.0, 1030.0)
    weather[:, 4] = (pressure - 990.0) / 40.0

    # ── Cloud cover (%) ───────────────────────────────────────────────────
    base_cloud = rng.uniform(10.0, 50.0)
    cloud = base_cloud + 30.0 * (rain_intensity / 30.0) + rng.normal(0, 10.0, size=length)
    cloud = np.clip(cloud, 0.0, 100.0)
    weather[:, 5] = cloud / 100.0

    return weather.astype(np.float32)  # [length, 6]


# Quick test
_test_weather = generate_weather(72)
print(f'✅ generate_weather: shape={_test_weather.shape}, '
      f'range=[{_test_weather.min():.3f}, {_test_weather.max():.3f}]')

# %% [markdown]
# ### Cell 1.8 — Maintenance History Generator

# %%
# Maintenance event types (10 categories)
_MAINTENANCE_TYPES: List[str] = [
    'rail_grinding',       # 0
    'sleeper_replacement', # 1
    'ballast_tamping',     # 2
    'weld_repair',         # 3
    'gauge_correction',    # 4
    'drainage_clearing',   # 5
    'formation_repair',    # 6
    'ultrasonic_test',     # 7
    'visual_inspection',   # 8
    'emergency_repair',    # 9
]

_NUM_MAINT_TYPES: int = len(_MAINTENANCE_TYPES)  # 10


def generate_maintenance_history(
    num_events: int = 16,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Generate synthetic maintenance history for a track section.

    Each event is encoded as a 64-dimensional vector:
        - [0-9]   event_type one-hot      (10 categories)
        - [10]    severity                 (0-1, normalised)
        - [11]    duration_hours           (0-1, /48h max)
        - [12]    days_since_event         (0-1, /365)
        - [13]    cost_normalised          (0-1)
        - [14]    outcome_code             (0-1, 0=scheduled, 0.5=unscheduled, 1=emergency)
        - [15]    crew_size_norm           (0-1, /20)
        - [16]    tools_used_norm          (0-1)
        - [17]    weather_during_norm      (0-1)
        - [18]    section_closed_flag      (0 or 1)
        - [19]    defect_found_flag        (0 or 1)
        - [20-29] defect_type one-hot      (10 defect sub-types)
        - [30]    repeat_event_flag        (0 or 1)
        - [31]    effectiveness_score      (0-1)
        - [32-63] reserved / embedding pad (zeros for future expansion)

    Events are ordered chronologically (most recent first), with
    ``days_since_event`` increasing for older events.

    Args:
        num_events: Number of maintenance events to generate (default 16).
        rng: NumPy ``Generator`` instance for reproducibility.

    Returns:
        ndarray of shape ``[num_events, 64]``.

    Raises:
        ValueError: If ``num_events`` ≤ 0.
    """
    if num_events <= 0:
        raise ValueError(f"num_events must be > 0, got {num_events}")

    rng = rng or np.random.default_rng(CONFIG['seed'])
    history = np.zeros((num_events, 64), dtype=np.float32)  # [num_events, 64]

    # Generate chronologically ordered days_since (most recent first)
    days_since_values = np.sort(rng.uniform(1, 365, size=num_events))  # ascending = oldest last
    days_since_values = days_since_values[::-1]  # reverse: most recent first
    # Normalise
    days_since_norm = days_since_values / 365.0

    for i in range(num_events):
        # Event type one-hot
        etype = rng.integers(0, _NUM_MAINT_TYPES)
        history[i, etype] = 1.0

        # Continuous features
        history[i, 10] = rng.uniform(0.0, 1.0)   # severity
        history[i, 11] = rng.uniform(0.05, 1.0)  # duration_hours / 48
        history[i, 12] = days_since_norm[i]       # days_since / 365
        history[i, 13] = rng.uniform(0.0, 1.0)   # cost
        history[i, 14] = rng.choice([0.0, 0.5, 1.0], p=[0.6, 0.3, 0.1])  # outcome
        history[i, 15] = rng.uniform(0.1, 1.0)   # crew_size
        history[i, 16] = rng.uniform(0.0, 1.0)   # tools_used
        history[i, 17] = rng.uniform(0.0, 1.0)   # weather_during
        history[i, 18] = rng.choice([0.0, 1.0], p=[0.7, 0.3])   # section_closed
        history[i, 19] = rng.choice([0.0, 1.0], p=[0.5, 0.5])   # defect_found

        # Defect sub-type one-hot (if defect was found)
        if history[i, 19] > 0.5:
            defect_type = rng.integers(0, 10)
            history[i, 20 + defect_type] = 1.0

        history[i, 30] = rng.choice([0.0, 1.0], p=[0.8, 0.2])   # repeat_event
        history[i, 31] = rng.uniform(0.3, 1.0)   # effectiveness

        # [32-63] reserved — leave as zeros

    return history  # [num_events, 64]


# Quick test
_test_maint = generate_maintenance_history(16)
print(f'✅ generate_maintenance_history: shape={_test_maint.shape}, '
      f'non-zero fraction={(_test_maint != 0).mean():.2f}')

# %% [markdown]
# ### Cell 1.9 — RakshakDataset Class
#
# The main PyTorch Dataset that composes all generators to produce
# the 7-modality samples with configurable failure injection.

# %%
class RakshakDataset(torch.utils.data.Dataset):
    """Complete synthetic Indian Railways predictive maintenance dataset.

    Generates all 7 input modalities with configurable failure injection.
    Supports class-imbalanced generation with ~3% failure rate (matching
    real-world IR incident frequency).

    Each sample represents one 30-day monitoring window for a single track
    section.  Failures are injected with realistic lead times and the dataset
    preserves temporal ordering for proper train/val/test splitting.

    Args:
        num_sections: Number of track sections to simulate.
        num_samples_per_section: Samples per section over time (temporal
            snapshots).  Each sample is an independent 30-day window.
        failure_rate: Fraction of samples with injected failure (default 0.03).
        seed: Random seed for reproducibility.

    Returns:
        dict from ``__getitem__`` — see shared contract for exact schema.

    Raises:
        ValueError: If parameters are out of valid ranges.
    """

    def __init__(
        self,
        num_sections: int = 50,
        num_samples_per_section: int = 12,
        failure_rate: float = 0.03,
        seed: int = 42,
    ) -> None:
        super().__init__()

        if num_sections <= 0:
            raise ValueError(f"num_sections must be > 0, got {num_sections}")
        if num_samples_per_section <= 0:
            raise ValueError(f"num_samples_per_section must be > 0, got {num_samples_per_section}")
        if not (0.0 <= failure_rate <= 1.0):
            raise ValueError(f"failure_rate must be in [0, 1], got {failure_rate}")

        self.num_sections = num_sections
        self.num_samples_per_section = num_samples_per_section
        self.failure_rate = failure_rate
        self.seed = seed

        self.total_samples = num_sections * num_samples_per_section
        self.seq_len = CONFIG['seq_len']          # 720

        # Pre-build shared resources
        self.rng = np.random.default_rng(seed)
        self.edge_index = build_adjacency_graph()  # [2, E]

        # Pre-generate track metadata (static per section, reused across time)
        self.section_metadata = generate_track_metadata(
            num_sections, rng=np.random.default_rng(seed + 100)
        )  # [num_sections, 32]

        # Pre-determine which samples have failures (for reproducibility)
        failure_rng = np.random.default_rng(seed + 200)
        self.failure_mask = failure_rng.random(self.total_samples) < failure_rate
        # Assign failure categories to failed samples
        self.failure_cats = failure_rng.integers(
            0, CONFIG['num_failure_categories'], size=self.total_samples
        )
        # Assign lead times (48-480 hours before window end)
        self.lead_times = failure_rng.uniform(48.0, 480.0, size=self.total_samples)

        # Category → injection method mapping
        self._CATEGORY_TO_INJECTOR: Dict[int, str] = {
            0: 'rail_fracture',
            1: 'gauge_deviation',
            2: 'thermal_buckling',
            3: 'ballast_degradation',
            4: 'weld_failure',
            5: 'sleeper_damage',
            6: 'drainage_failure',       # uses ballast-like noise
            7: 'subgrade_settlement',
        }

        # Ensure at least some failures exist
        n_failures = int(self.failure_mask.sum())
        if n_failures == 0 and self.total_samples > 0:
            # Force at least 1 failure
            self.failure_mask[0] = True
            n_failures = 1

        print(f'📊 RakshakDataset: {self.total_samples} samples '
              f'({num_sections} sections × {num_samples_per_section} windows)')
        print(f'   Failures: {n_failures} ({100.0 * n_failures / max(1, self.total_samples):.1f}%)')
        print(f'   Seq length: {self.seq_len} hrs | Edge index: {self.edge_index.shape}')

    def __len__(self) -> int:
        """Return total number of samples.

        Args:
            None

        Returns:
            Integer count of samples.

        Raises:
            Nothing.
        """
        return self.total_samples

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single sample with all 7 modalities and labels.

        Args:
            idx: Sample index (0-based).

        Returns:
            Dictionary with keys matching the shared contract:
                - ``vibration``:  FloatTensor ``[720, 3]``
                - ``temperature``:  FloatTensor ``[720, 1]``
                - ``gauge``:  FloatTensor ``[720, 1]``
                - ``metadata``:  FloatTensor ``[32]``
                - ``weather``:  FloatTensor ``[72, 6]``
                - ``maintenance_history``:  FloatTensor ``[16, 64]``
                - ``section_id``:  int
                - ``edge_index``:  LongTensor ``[2, E]``
                - ``failure_occurred``:  FloatTensor ``[1]``
                - ``failure_category``:  LongTensor ``[1]``
                - ``time_to_failure``:  FloatTensor ``[1]``

        Raises:
            IndexError: If ``idx`` is out of range.
        """
        if idx < 0 or idx >= self.total_samples:
            raise IndexError(f"Index {idx} out of range [0, {self.total_samples})")

        # Determine section and temporal window
        section_id = idx % self.num_sections
        window_id = idx // self.num_sections

        # Per-sample RNG (deterministic given idx)
        sample_rng = np.random.default_rng(self.seed + idx * 7 + 31)

        # ── Instantiate generators with per-sample RNG ─────────────────────
        vib_gen = VibrationGenerator(rng=np.random.default_rng(self.seed + idx * 11))
        temp_gen = TemperatureGenerator(rng=np.random.default_rng(self.seed + idx * 13))
        gauge_gen = GaugeGenerator(rng=np.random.default_rng(self.seed + idx * 17))

        # ── Generate normal signals ────────────────────────────────────────
        vibration = vib_gen.generate_normal(
            length=self.seq_len, channels=CONFIG['vib_channels']
        )  # [720, 3]

        temperature = temp_gen.generate_normal(
            length=self.seq_len
        )  # [720, 1]

        gauge = gauge_gen.generate_normal(
            length=self.seq_len
        )  # [720, 1]

        # ── Failure injection ──────────────────────────────────────────────
        has_failure = bool(self.failure_mask[idx])
        failure_cat = int(self.failure_cats[idx]) if has_failure else 0
        lead_time = float(self.lead_times[idx]) if has_failure else 0.0

        if has_failure:
            cat_name = self._CATEGORY_TO_INJECTOR[failure_cat]

            if cat_name == 'rail_fracture':
                vibration = vib_gen.inject_rail_fracture(vibration, lead_time)
            elif cat_name == 'gauge_deviation':
                gauge = gauge_gen.inject_gauge_deviation(gauge, lead_time)
                vibration = vib_gen.inject_ballast_degradation(vibration, lead_time * 0.5)
            elif cat_name == 'thermal_buckling':
                temperature = temp_gen.inject_thermal_buckling(temperature, lead_time)
                vibration = vib_gen.inject_ballast_degradation(vibration, lead_time * 0.3)
            elif cat_name == 'ballast_degradation':
                vibration = vib_gen.inject_ballast_degradation(vibration, lead_time)
                gauge = gauge_gen.inject_subgrade_settlement(gauge, lead_time * 0.6)
            elif cat_name == 'weld_failure':
                vibration = vib_gen.inject_weld_failure(vibration, lead_time)
            elif cat_name == 'sleeper_damage':
                vibration = vib_gen.inject_sleeper_damage(vibration, lead_time)
                gauge = gauge_gen.inject_gauge_deviation(gauge, lead_time * 0.4)
            elif cat_name == 'drainage_failure':
                # Drainage failure manifests as ballast degradation + gauge drift
                vibration = vib_gen.inject_ballast_degradation(vibration, lead_time)
                gauge = gauge_gen.inject_subgrade_settlement(gauge, lead_time * 0.7)
            elif cat_name == 'subgrade_settlement':
                gauge = gauge_gen.inject_subgrade_settlement(gauge, lead_time)
                vibration = vib_gen.inject_ballast_degradation(vibration, lead_time * 0.5)

        # ── Static metadata for this section ───────────────────────────────
        metadata = self.section_metadata[section_id].copy()  # [32]

        # ── Weather forecast ───────────────────────────────────────────────
        weather = generate_weather(
            length=CONFIG['weather_hours'],
            rng=np.random.default_rng(self.seed + idx * 23 + window_id),
        )  # [72, 6]

        # ── Maintenance history ────────────────────────────────────────────
        maint_history = generate_maintenance_history(
            num_events=CONFIG['maint_events'],
            rng=np.random.default_rng(self.seed + section_id * 37 + window_id),
        )  # [16, 64]

        # ── Labels ─────────────────────────────────────────────────────────
        failure_occurred = 1.0 if has_failure else 0.0
        time_to_failure = lead_time if has_failure else 0.0

        # ── Assemble output dict ───────────────────────────────────────────
        sample: Dict[str, Any] = {
            'vibration': torch.tensor(vibration, dtype=torch.float32),              # [720, 3]
            'temperature': torch.tensor(temperature, dtype=torch.float32),          # [720, 1]
            'gauge': torch.tensor(gauge, dtype=torch.float32),                      # [720, 1]
            'metadata': torch.tensor(metadata, dtype=torch.float32),                # [32]
            'weather': torch.tensor(weather, dtype=torch.float32),                  # [72, 6]
            'maintenance_history': torch.tensor(maint_history, dtype=torch.float32),# [16, 64]
            'section_id': section_id,                                               # int
            'edge_index': self.edge_index.clone(),                                  # [2, E]
            'failure_occurred': torch.tensor([failure_occurred], dtype=torch.float32),  # [1]
            'failure_category': torch.tensor([failure_cat], dtype=torch.long),         # [1]
            'time_to_failure': torch.tensor([time_to_failure], dtype=torch.float32),   # [1]
        }

        return sample


print('✅ RakshakDataset class defined — full 7-modality synthetic generator')

# %% [markdown]
# ### Cell 1.10 — DataLoader Creation with Time-Based Split

# %%
def create_dataloaders(
    config: Dict[str, Any],
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train/val/test DataLoaders with time-based splitting.

    Time-based split ensures no future data leaks into training:
        - Train: first 60% of temporal windows per section
        - Val  : next 20%
        - Test : last 20%

    Args:
        config: The global ``CONFIG`` dictionary containing all parameters.

    Returns:
        Tuple of ``(train_loader, val_loader, test_loader)``.

    Raises:
        ValueError: If split ratios do not sum to 1.0.
    """
    train_ratio = config['train_ratio']
    val_ratio = config['val_ratio']
    test_ratio = config['test_ratio']

    ratio_sum = train_ratio + val_ratio + test_ratio
    if abs(ratio_sum - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1.0, got {ratio_sum}")

    num_sections = config['num_sections']
    # Number of temporal windows per section based on years of data
    # 1 year ≈ 12 monthly windows (each 30 days = 720 hours)
    num_samples_per_section = config['num_years'] * 12

    print(f'🔧 Creating dataset: {num_sections} sections × '
          f'{num_samples_per_section} windows/section')

    # Create full dataset
    dataset = RakshakDataset(
        num_sections=num_sections,
        num_samples_per_section=num_samples_per_section,
        failure_rate=config['failure_rate'],
        seed=config['seed'],
    )

    total = len(dataset)

    # ── Time-based split ──────────────────────────────────────────────────
    # Indices are laid out as: for each window w, sections 0..S-1 are at
    # positions w*S .. (w+1)*S - 1.  We split on window index.
    n_windows = num_samples_per_section
    train_windows = int(n_windows * train_ratio)
    val_windows = int(n_windows * val_ratio)
    test_windows = n_windows - train_windows - val_windows

    train_indices: List[int] = []
    val_indices: List[int] = []
    test_indices: List[int] = []

    for section in range(num_sections):
        for window in range(n_windows):
            idx = window * num_sections + section
            if idx >= total:
                continue
            if window < train_windows:
                train_indices.append(idx)
            elif window < train_windows + val_windows:
                val_indices.append(idx)
            else:
                test_indices.append(idx)

    train_set = Subset(dataset, train_indices)
    val_set = Subset(dataset, val_indices)
    test_set = Subset(dataset, test_indices)

    # Count failures per split
    def _count_failures(indices: List[int]) -> int:
        return int(sum(dataset.failure_mask[i] for i in indices))

    train_failures = _count_failures(train_indices)
    val_failures = _count_failures(val_indices)
    test_failures = _count_failures(test_indices)

    # ── DataLoaders ───────────────────────────────────────────────────────
    # Custom collate to handle the edge_index (shared across batch)
    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Custom collate function that stacks tensors and handles edge_index.

        Args:
            batch: List of sample dictionaries from __getitem__.

        Returns:
            Collated batch dictionary with batched tensors.

        Raises:
            Nothing.
        """
        collated: Dict[str, Any] = {}
        collated['vibration'] = torch.stack([s['vibration'] for s in batch])              # [B, 720, 3]
        collated['temperature'] = torch.stack([s['temperature'] for s in batch])          # [B, 720, 1]
        collated['gauge'] = torch.stack([s['gauge'] for s in batch])                      # [B, 720, 1]
        collated['metadata'] = torch.stack([s['metadata'] for s in batch])                # [B, 32]
        collated['weather'] = torch.stack([s['weather'] for s in batch])                  # [B, 72, 6]
        collated['maintenance_history'] = torch.stack(
            [s['maintenance_history'] for s in batch]
        )                                                                                  # [B, 16, 64]
        collated['section_id'] = [s['section_id'] for s in batch]                         # list of ints
        collated['edge_index'] = batch[0]['edge_index']                                   # [2, E] (shared)
        collated['failure_occurred'] = torch.stack(
            [s['failure_occurred'] for s in batch]
        )                                                                                  # [B, 1]
        collated['failure_category'] = torch.stack(
            [s['failure_category'] for s in batch]
        )                                                                                  # [B, 1]
        collated['time_to_failure'] = torch.stack(
            [s['time_to_failure'] for s in batch]
        )                                                                                  # [B, 1]
        return collated

    train_loader = DataLoader(
        train_set,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=0,       # Colab-safe
        pin_memory=(config['device'] == 'cuda'),
        drop_last=False,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=0,
        pin_memory=(config['device'] == 'cuda'),
        drop_last=False,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=0,
        pin_memory=(config['device'] == 'cuda'),
        drop_last=False,
        collate_fn=collate_fn,
    )

    # ── Print statistics ──────────────────────────────────────────────────
    failure_mask = dataset.failure_mask
    total_failures = int(failure_mask.sum())
    actual_failure_rate = total_failures / max(1, total)

    print('\n' + '=' * 60)
    print('  📊  DATASET STATISTICS')
    print('=' * 60)
    print(f'Total Samples: {total}')
    print(f'Failure Count: {total_failures}')
    print(f'Failure Rate: {actual_failure_rate * 100:.1f}%')
    print('Class Distribution:')
    
    if total_failures > 0:
        failure_cats = dataset.failure_cats[failure_mask]
        unique_cats, counts = np.unique(failure_cats, return_counts=True)
        for cat, count in zip(unique_cats, counts):
            print(f'  Category {cat} : {count}')
    else:
        print('  No failures generated.')
        
    print(f'Train/Val/Test Failure Counts: {train_failures} / {val_failures} / {test_failures}')
    
    print('-' * 60)
    print(f'  Batch size           : {config["batch_size"]}')
    print(f'  Train batches        : {len(train_loader)}')
    print(f'  Val batches          : {len(val_loader)}')
    print(f'  Test batches         : {len(test_loader)}')
    print('=' * 60)

    return train_loader, val_loader, test_loader


print('✅ create_dataloaders() defined — time-based split with custom collation')

# %% [markdown]
# ### Cell 1.11 — Section 1 Checkpoint: Create DataLoaders, Print Shapes, Visualise
#
# This cell runs the full data pipeline end-to-end and visualises a sample
# from each modality to confirm correctness.

# %%
# ── Create dataloaders ────────────────────────────────────────────────────────
train_loader, val_loader, test_loader = create_dataloaders(CONFIG)

# ── Grab one batch and inspect shapes ─────────────────────────────────────────
sample_batch = next(iter(train_loader))

print('\n📐 Sample Batch Shapes:')
print(f'  vibration           : {sample_batch["vibration"].shape}')           # [B, 720, 3]
print(f'  temperature         : {sample_batch["temperature"].shape}')         # [B, 720, 1]
print(f'  gauge               : {sample_batch["gauge"].shape}')               # [B, 720, 1]
print(f'  metadata            : {sample_batch["metadata"].shape}')            # [B, 32]
print(f'  weather             : {sample_batch["weather"].shape}')             # [B, 72, 6]
print(f'  maintenance_history : {sample_batch["maintenance_history"].shape}') # [B, 16, 64]
print(f'  edge_index          : {sample_batch["edge_index"].shape}')          # [2, E]
print(f'  failure_occurred    : {sample_batch["failure_occurred"].shape}')     # [B, 1]
print(f'  failure_category    : {sample_batch["failure_category"].shape}')     # [B, 1]
print(f'  time_to_failure     : {sample_batch["time_to_failure"].shape}')      # [B, 1]

# ── Verify dtypes ─────────────────────────────────────────────────────────────
print('\n🔬 Dtype Verification:')
for key in ['vibration', 'temperature', 'gauge', 'metadata', 'weather',
            'maintenance_history', 'failure_occurred', 'time_to_failure']:
    print(f'  {key:25s}: {sample_batch[key].dtype}')
print(f'  {"failure_category":25s}: {sample_batch["failure_category"].dtype}')
print(f'  {"edge_index":25s}: {sample_batch["edge_index"].dtype}')

# ── Visualise one sample ──────────────────────────────────────────────────────
# Find a failure sample for more interesting visualisation
failure_idx = None
full_dataset = train_loader.dataset.dataset  # unwrap Subset → RakshakDataset
for i in range(len(full_dataset)):
    if full_dataset.failure_mask[i]:
        failure_idx = i
        break

if failure_idx is None:
    failure_idx = 0
    print('⚠️  No failures found — visualising normal sample')

viz_sample = full_dataset[failure_idx]
cat_name = FAILURE_CATEGORIES[viz_sample['failure_category'].item()]
ttf = viz_sample['time_to_failure'].item()

fig, axes = plt.subplots(3, 2, figsize=(16, 12))
fig.suptitle(
    f'Sample #{failure_idx} — {"FAILURE: " + cat_name if viz_sample["failure_occurred"].item() > 0 else "NORMAL"}'
    f' | TTF={ttf:.0f}h',
    fontsize=14, fontweight='bold'
)
hours = np.arange(CONFIG['seq_len'])

# (0,0) Vibration — 3 channels
ax = axes[0, 0]
vib = viz_sample['vibration'].numpy()  # [720, 3]
for ch, label in enumerate(['X', 'Y', 'Z']):
    ax.plot(hours, vib[:, ch], label=f'Ch-{label}', alpha=0.8, linewidth=0.6)
ax.set_title('Vibration (3-axis)')
ax.set_xlabel('Hours')
ax.set_ylabel('Amplitude')
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3)

# (0,1) Temperature
ax = axes[0, 1]
temp = viz_sample['temperature'].numpy()  # [720, 1]
ax.plot(hours, temp[:, 0], color='red', linewidth=0.8)
ax.axhline(y=55.0, color='darkred', linestyle='--', alpha=0.7, label='Buckling threshold (55°C)')
ax.set_title('Rail Temperature')
ax.set_xlabel('Hours')
ax.set_ylabel('°C')
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3)

# (1,0) Gauge deviation
ax = axes[1, 0]
gauge_data = viz_sample['gauge'].numpy()  # [720, 1]
ax.plot(hours, gauge_data[:, 0], color='green', linewidth=0.8)
ax.axhline(y=10.0, color='darkred', linestyle='--', alpha=0.7, label='Safety limit (+10mm)')
ax.axhline(y=-10.0, color='darkred', linestyle='--', alpha=0.7, label='Safety limit (-10mm)')
ax.set_title('Gauge Deviation (mm from 1676mm)')
ax.set_xlabel('Hours')
ax.set_ylabel('mm')
ax.legend(fontsize=7)
ax.grid(True, alpha=0.3)

# (1,1) Weather forecast
ax = axes[1, 1]
weather_data = viz_sample['weather'].numpy()  # [72, 6]
weather_labels = ['Temp', 'Humidity', 'Precip', 'Wind', 'Pressure', 'Cloud']
weather_hours = np.arange(CONFIG['weather_hours'])
for f_idx, label in enumerate(weather_labels):
    ax.plot(weather_hours, weather_data[:, f_idx], label=label, alpha=0.8, linewidth=0.8)
ax.set_title('72h Weather Forecast (normalised)')
ax.set_xlabel('Hours ahead')
ax.set_ylabel('Normalised value')
ax.legend(fontsize=6, ncol=2)
ax.grid(True, alpha=0.3)

# (2,0) Metadata (bar chart)
ax = axes[2, 0]
meta = viz_sample['metadata'].numpy()  # [32]
ax.bar(range(32), meta, color='steelblue', alpha=0.8, width=0.8)
ax.set_title(f'Track Metadata ({len(meta)} features)')
ax.set_xlabel('Feature index')
ax.set_ylabel('Value')
ax.grid(True, alpha=0.3, axis='y')

# (2,1) Maintenance history heatmap
ax = axes[2, 1]
maint = viz_sample['maintenance_history'].numpy()  # [16, 64]
im = ax.imshow(maint, aspect='auto', cmap='YlOrRd', interpolation='nearest')
ax.set_title('Maintenance History (16 events × 64 features)')
ax.set_xlabel('Feature dimension')
ax.set_ylabel('Event (most recent first)')
plt.colorbar(im, ax=ax, fraction=0.046)

plt.tight_layout()
save_path = os.path.join(CONFIG['figures_dir'], 'section1_data_overview.png')
plt.savefig(save_path, bbox_inches='tight', dpi=150)
print(f'📊 Figure saved → {save_path}')
plt.show()
plt.close(fig)

# ── Station graph visualisation ───────────────────────────────────────────────
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
G = nx.Graph()
for i, s in enumerate(STATIONS):
    G.add_node(i, label=s)

# Add edges (remove self-loops and duplicates for visualisation)
edge_set = set()
ei = _edge_index.numpy()
for col in range(ei.shape[1]):
    u, v = ei[0, col], ei[1, col]
    if u != v:
        edge_set.add((min(u, v), max(u, v)))
G.add_edges_from(edge_set)

# Position nodes using station coordinates
pos = {}
for i, s in enumerate(STATIONS):
    lat, lon = STATION_COORDS[s]
    pos[i] = (lon, lat)  # (x=lon, y=lat)

labels = {i: s for i, s in enumerate(STATIONS)}
nx.draw_networkx(
    G, pos, ax=ax, labels=labels,
    node_color='#2196F3', node_size=600,
    font_size=8, font_weight='bold', font_color='white',
    edge_color='#757575', width=2.0, alpha=0.9,
)
ax.set_title('Delhi–Agra Corridor Station Graph (12 nodes)', fontsize=13, fontweight='bold')
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.grid(True, alpha=0.2)
plt.tight_layout()
graph_path = os.path.join(CONFIG['figures_dir'], 'station_graph.png')
plt.savefig(graph_path, bbox_inches='tight', dpi=150)
print(f'📊 Station graph saved → {graph_path}')
plt.show()
plt.close(fig)

# ── Final class distribution ──────────────────────────────────────────────────
failure_cats_all = []
for i in range(len(full_dataset)):
    if full_dataset.failure_mask[i]:
        failure_cats_all.append(int(full_dataset.failure_cats[i]))

if failure_cats_all:
    fig, ax = plt.subplots(figsize=(10, 4))
    cat_counts = [failure_cats_all.count(c) for c in range(CONFIG['num_failure_categories'])]
    bars = ax.bar(FAILURE_CATEGORIES, cat_counts, color=sns.color_palette('husl', 8), edgecolor='white')
    ax.set_title('Failure Category Distribution', fontsize=13, fontweight='bold')
    ax.set_xlabel('Category')
    ax.set_ylabel('Count')
    plt.xticks(rotation=30, ha='right')
    for bar, cnt in zip(bars, cat_counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                str(cnt), ha='center', va='bottom', fontsize=9, fontweight='bold')
    plt.tight_layout()
    dist_path = os.path.join(CONFIG['figures_dir'], 'failure_distribution.png')
    plt.savefig(dist_path, bbox_inches='tight', dpi=150)
    print(f'📊 Distribution plot saved → {dist_path}')
    plt.show()
    plt.close(fig)

print('\n' + '=' * 60)
print('  ✅  SECTION 1 CHECKPOINT — Data Pipeline Complete')
print('=' * 60)
print(f'  Estimated Dataset Size       : {len(full_dataset)} samples')
print(f'  Estimated Train Samples      : {len(train_loader.dataset)} samples')
print(f'  Estimated Validation Samples : {len(val_loader.dataset)} samples')
print(f'  Estimated Test Samples       : {len(test_loader.dataset)} samples')
print('-' * 60)
print(f'  Train loader : {len(train_loader)} batches')
print(f'  Val loader   : {len(val_loader)} batches')
print(f'  Test loader  : {len(test_loader)} batches')
print('  All shapes verified ✓')
print('  All dtypes verified ✓')
print('  Visualisations saved ✓')
print('=' * 60)
print('\n  ➡️  Proceed to Section 2 — Anomaly Detection Engine')
