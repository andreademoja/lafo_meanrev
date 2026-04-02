# State-Space Models for LAFO Mean-Reversion

This module implements state-space models from Sections 3.1-3.4 of the LAFO research paper.

## Overview

State-space models provide a flexible framework for filtering noise from price series and estimating latent fundamental levels. They are particularly useful for:

1. **Long-term memory**: Capturing persistent dependencies in financial data
2. **Regime adaptation**: Automatically adjusting to changing market conditions
3. **Noise reduction**: Separating signal from noise more effectively than simple moving averages

## Components

### 1. Kalman Filter (`state_space_models.py`)

The standard Kalman Filter implementation:

```python
from lafo import KalmanFilter

kf = KalmanFilter()
filtered_means, covariances = kf.compute_filtered_mean(observations)
```

**Key Features:**
- Optimal linear estimator for Gaussian systems
- Adaptive noise estimation
- Fast O(T) computation

### 2. Switching Kalman Filter (`state_space_models.py`)

Detects and adapts to market regime changes:

```python
from lafo import SwitchingKalmanFilter

skf = SwitchingKalmanFilter(n_regimes=2, switching_prob=0.05)
filtered_means, _ = skf.predict(observations)
```

**Use Cases:**
- Bull/bear market adaptation
- High/low volatility regimes
- Regime-specific parameter tuning

### 3. Variational Inference Filter (`state_space_models.py`)

Approximate Bayesian inference for tractable posterior estimation:

```python
from lafo import VariationalInferenceFilter

vi = VariationalInferenceFilter(signal_to_noise_ratio=0.5)
filtered = vi.compute(observations)
```

### 4. Advanced CNN Filters (`advanced_cnn.py`)

Deep learning-based filtering variants:

#### ARMABlock (Attention-based Recurrent Memory Block)

Combines convolutional filtering with attention mechanisms:

```python
from lafo import ARMABlock

filter = ARMABlock(channels=64, kernel_size=31)
output = filter(observations)
```

**Architecture:**
- Convolutional filtering (depthwise separable)
- Self-attention for long-range dependencies
- Memory gates for adaptive forgetting
- Time decay for stability

#### Dual-Path ARMA Block

Separates low-frequency (trend) and high-frequency (noise) components:

```python
from lafo import DualPathARMABlock

filter = DualPathARMABlock(
    channels=64,
    low_path_kernels=[31, 61, 101]  # Multi-scale filtering
)
output = filter(observations)
```

#### Mamba Filter

Linear time-invariant state space model (S6-inspired):

```python
from lafo import MambaFilter

filter = MambaFilter(d_state=16, d_conv=16)
output = filter(observations)
```

#### RNN-CNN Hybrid

Combines CNN and RNN components:

```python
from lafo import RNNDualPathCNN

filter = RNNDualPathCNN(channels=64, kernel_size=31)
output = filter(observations)
```

### 5. Regime Detection (`regime_detection.py`)

#### Hidden Markov Model (HMM)

Detects hidden market regimes:

```python
from lafo import HiddenMarkovModel

hmm = HiddenMarkovModel(n_states=2, emission_dims=1)
params = hmm.fit(observations)
state_sequence = hmm.get_state_sequence(observations)
```

**Parameters:**
- `n_states`: Number of regimes (2 for bull/bear)
- `emission_dims`: Observation dimensions (returns, vol, etc.)
- `max_iter`: EM algorithm iterations

#### Volatility Clustering

Detects high/low volatility regimes:

```python
from lafo import VolatilityClustering

detector = VolatilityClustering(window_size=60, threshold_std=2.0)
regimes = detector.detect_regimes(observations)
```

#### Market Regime Detection

Clustering-based regime detection:

```python
from lafo import detect_market_regime

regimes = detect_market_regime(
    returns=observations,
    window_size=50,
    num_clusters=2
)
```

### 6. Ensemble Framework (`ensemble.py`)

Combines multiple filters for robust filtering:

```python
from lafo import EnsembleEnsemble

ensemble = EnsembleEnsemble(
    filters=[
        {'type': 'kalman', 'params': {'noise_std': 0.1}},
        {'type': 'cnn', 'kernel_size': 31},
        {'type': 'arma', 'order': (1, 1, 1)}
    ],
    method='weighted'
)
output = ensemble.compute(observations)
```

**Methods:**
- `weighted`: Weighted average of filter outputs
- `stacking`: Meta-model combining predictions
- `bagging`: Bootstrap aggregation

### 7. Adaptive Filter

Regime-adaptive filtering:

```python
from lafo import AdaptiveFilter

adaptive = AdaptiveFilter(
    base_filter=kf,
    regime_detector=hmm,
    adaptation_window=20
)
filtered = adaptive.filter(observations)
```

## Usage Examples

### Complete Pipeline

```python
import numpy as np
from lafo import create_filter_pipeline, compute_lafo_signal

# Generate synthetic data
np.random.seed(42)
observations = np.cumsum(np.random.randn(1000) / np.sqrt(252))

# Apply filtering
filtered = create_filter_pipeline(observations, filter_type='ensemble')
signal = compute_lafo_signal(observations, filter_type='kalman')
```

### Hyperparameter Optimization

```python
from lafo import FilterOptimizer

optimizer = FilterOptimizer()
best_params, best_output = optimizer.grid_search(
    observations,
    objective=lambda pred, true: np.mean((pred - true) ** 2)
)
print(f"Best params: {best_params}")
```

### Cross-Validation

```python
best_params, best_output = optimizer.cross_validate(
    observations,
    folds=5
)
```

## Performance Notes

- **Kalman Filter**: O(T) complexity, suitable for real-time applications
- **CNN Filters**: GPU-accelerated with PyTorch backend
- **Ensemble**: Scales linearly with number of filters
- **HMM**: EM algorithm convergence typically < 100 iterations

## Limitations & Considerations

1. **Kalman Filter**: Assumes Gaussian noise (may need non-Gaussian variants)
2. **CNN Filters**: Require GPU for training; use pre-trained weights
3. **HMM**: May struggle with > 4 states (curse of dimensionality)
4. **Regime Detection**: Computationally expensive for long series

## References

- Kalman, R. E. (1960). "A New Approach to Linear Filtering and Prediction Problems"
- Rabiner, L. R. (1989). "A Tutorial on Hidden Markov Models and Selected Applications in Speech Recognition"
- Gu, Q. et al. (2020). "HIpPO-RNN: A New Perspective on neuronal dynamics"
- Mamba (2024). "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"

## License

MIT License - See LICENSE file for details.
