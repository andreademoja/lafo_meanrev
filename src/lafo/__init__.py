"""
LAFO Mean-Reversion Trading Package
Sections 2.1-3.4: Implementation of LAFO loss, CNN, state-space models, and ensemble methods.

Exports:
- LAFO functions: build_sliding_average_operator, compute_W, lafo_loss, lafo_loss_efficient
- CNN filter: cnn_filter (torch module)
- Penalized LAFO: penalties (ℓ₂, TV, ℓ₁)
- State-space models: Kalman filter, Variational Inference, Switching K.F.
- Advanced CNN: ARMABlock, Mamba, RNN variants
- Regime detection: HMM, volatility clustering
- Ensemble: filter averaging framework
"""
# Re-export all modules for convenience
from .simulation import *
from .lafo import build_sliding_average_operator, compute_W, lafo_loss, lafo_loss_efficient
from .cnn_filter import CNNFilter
from .penalized_lafo import LAFOPenalty, compute_penalized_lafo
from .state_space_models import (
    KalmanFilter,
    SwitchingKalmanFilter,
    VariationalInferenceFilter,
    EnsembleFilter
)
from .advanced_cnn import (
    ARMABlock,
    DualPathARMABlock,
    MambaFilter,
    RNNDualPathCNN,
    CNNFilterFactory
)
from .regime_detection import (
    HiddenMarkovModel,
    VolatilityClustering,
    AdaptiveFilter,
    detect_market_regime
)
from .ensemble import (
    EnsembleFilter as EnsembleEnsemble,
    FilterOptimizer
)
# Import simulation if it exists
try:
    from . import simulation
except ImportError:
    pass

__version__ = '1.0.0'
__all__ = [
    # LAFO functions
    'build_sliding_average_operator',
    'compute_W',
    'lafo_loss',
    'lafo_loss_efficient',
    # CNN filter
    'CNNFilter',
    # Penalized LAFO
    'LAFOPenalty',
    'compute_penalized_lafo',
    # State-space models
    'KalmanFilter',
    'SwitchingKalmanFilter',
    'VariationalInferenceFilter',
    'EnsembleFilter',
    # Advanced CNN
    'ARMABlock',
    'DualPathARMABlock',
    'MambaFilter',
    'RNNDualPathCNN',
    'CNNFilterFactory',
    # Regime detection
    'HiddenMarkovModel',
    'VolatilityClustering',
    'AdaptiveFilter',
    'detect_market_regime',
    # Ensemble
    'EnsembleEnsemble',
    'FilterOptimizer',
    # Simulation
    'generate_synthetic_data',
    'load_real_data'
]

def create_filter_pipeline(observations, filter_type='ensemble'):
    """
    Create a complete filtering pipeline.
    
    Args:
        observations: Price/return series
        filter_type: 'kalman', 'cnn', 'state_space', 'ensemble', 'variational'
    
    Returns:
        Filtered fundamental levels
    """
    # Default ensemble pipeline
    ensemble = EnsembleEnsemble()
    
    # Select filter
    if filter_type == 'kalman':
        from .state_space_models import KalmanFilter
        kf = KalmanFilter()
        kf.compute_filtered_mean(observations)
        return kf.state[0]
    
    elif filter_type == 'cnn':
        from .advanced_cnn import ARMABlock
        import torch
        filter = ARMABlock(channels=64, kernel_size=31)
        # Convert to torch tensor
        obs_torch = torch.FloatTensor(observations).unsqueeze(0).unsqueeze(1)
        output = filter(obs_torch)
        return output.squeeze().detach().numpy()
    
    elif filter_type == 'state_space':
        from .state_space_models import compute_state_space_filter
        return compute_state_space_filter(observations, filter_type='kalman')
    
    elif filter_type == 'variational':
        from .state_space_models import VariationalInferenceFilter
        vi = VariationalInferenceFilter()
        filtered = observations[0]
        for obs in observations[1:]:
            filtered = vi.update(obs, filtered)
        return np.full_like(observations, filtered)
    
    else:
        # Default: ensemble
        return ensemble.compute(observations)


def compute_lafo_signal(observations, filter_type='kalman'):
    """
    Compute LAFO mean-reversion signal.
    
    Args:
        observations: Price series
        filter_type: Filter type
    
    Returns:
        LAFO signal (fundamental level estimate)
    """
    filtered = create_filter_pipeline(observations, filter_type)
    return filtered
