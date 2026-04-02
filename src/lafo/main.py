"""
Main entry point for LAFO mean-reversion trading system
"""
import numpy as np
import sys
sys.path.insert(0, 'lafo_meanrev/src')

from . import create_filter_pipeline, compute_lafo_signal
from .lafo import (
    build_sliding_average_operator,
    compute_W,
    lafo_loss,
    lafo_loss_efficient
)
from .state_space_models import (
    KalmanFilter,
    SwitchingKalmanFilter,
    VariationalInferenceFilter,
    EnsembleFilter
)
from .cnn_filter import CNNFilter
from .advanced_cnn import ARMABlock, MambaFilter
from .penalized_lafo import LAFOPenalty, compute_penalized_lafo
from .regime_detection import HiddenMarkovModel, detect_market_regime
from .ensemble import FilterOptimizer

def generate_test_data(n=1000, regime_duration=200):
    """Generate synthetic mean-reversion data."""
    np.random.seed(42)
    trend = np.cumsum(np.random.randn(n) * 0.1 + 0.01)
    noise = np.random.randn(n) * 0.1
    observations = trend + noise
    return observations

def run_basic_test():
    """Run basic functionality test."""
    print("=" * 60)
    print("LAFO Mean-Reversion Trading System - Basic Test")
    print("=" * 60)
    
    # Generate test data
    n = 500
    observations = generate_test_data(n)
    print(f"\nGenerated synthetic data: {len(observations)} samples")
    
    # Test 1: Kalman Filter
    print("\n--- Test 1: Kalman Filter ---")
    try:
        kf = KalmanFilter()
        filtered, cov = kf.compute_filtered_mean(observations)
        print(f"✅ Kalman Filter - Filtered shape: {filtered.shape}")
        print(f"   First 5 filtered values: {filtered[:5]}")
    except Exception as e:
        print(f"❌ Kalman Filter error: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 2: Switching Kalman Filter
    print("\n--- Test 2: Switching Kalman Filter ---")
    try:
        skf = SwitchingKalmanFilter(n_regimes=2, switching_prob=0.1)
        filtered_skf, _ = skf.predict(observations)
        print(f"✅ Switching Kalman Filter - Shape: {filtered_skf.shape}")
    except Exception as e:
        print(f"❌ Switching Kalman Filter error: {e}")
    
    # Test 3: Variational Inference
    print("\n--- Test 3: Variational Inference Filter ---")
    try:
        vi = VariationalInferenceFilter(signal_to_noise_ratio=0.5)
        filtered_vi = vi.compute(observations)
        print(f"✅ Variational Inference Filter - Shape: {filtered_vi.shape}")
    except Exception as e:
        print(f"❌ Variational Inference Filter error: {e}")
    
    # Test 4: Ensemble Filter
    print("\n--- Test 4: Ensemble Filter ---")
    try:
        ensemble = EnsembleFilter()
        filtered_ens = ensemble.compute(observations)
        print(f"✅ Ensemble Filter - Shape: {filtered_ens.shape}")
    except Exception as e:
        print(f"❌ Ensemble Filter error: {e}")
    
    # Test 5: ARMABlock
    print("\n--- Test 5: ARMABlock CNN Filter ---")
    try:
        filter = ARMABlock(channels=64, kernel_size=31)
        import torch
        obs_torch = torch.FloatTensor(observations).unsqueeze(0).unsqueeze(1)
        output = filter(obs_torch).squeeze().detach().numpy()
        print(f"✅ ARMABlock - Shape: {output.shape}")
    except Exception as e:
        print(f"❌ ARMABlock error: {e}")
    
    # Test 6: Mamba Filter
    print("\n--- Test 6: Mamba Filter ---")
    try:
        filter = MambaFilter(d_state=16, d_conv=16)
        import torch
        obs_torch = torch.FloatTensor(observations).unsqueeze(0).unsqueeze(1)
        output = filter(obs_torch).squeeze().detach().numpy()
        print(f"✅ Mamba Filter - Shape: {output.shape}")
    except Exception as e:
        print(f"❌ Mamba Filter error: {e}")
    
    # Test 7: HMM Regime Detection
    print("\n--- Test 7: Hidden Markov Model ---")
    try:
        hmm = HiddenMarkovModel(n_states=2, emission_dims=1)
        params = hmm.fit(observations)
        state_sequence = hmm.get_state_sequence(observations)
        print(f"✅ HMM - High volatility periods: {state_sequence.sum()}")
        print(f"   First 10 states: {state_sequence[:10]}")
    except Exception as e:
        print(f"❌ HMM error: {e}")
    
    # Test 8: Market Regime Detection
    print("\n--- Test 8: Market Regime Detection ---")
    try:
        regimes = detect_market_regime(
            observations,
            window_size=50,
            num_clusters=2
        )
        print(f"✅ Market Regime Detection - High regime periods: {regimes.sum()}")
    except Exception as e:
        print(f"❌ Market Regime Detection error: {e}")
    
    # Test 9: Ensemble Optimizer
    print("\n--- Test 9: Filter Optimizer ---")
    try:
        optimizer = FilterOptimizer()
        best_params = optimizer.optimize_weights(observations)
        print(f"✅ Filter Optimizer - Optimized weights: {best_params}")
    except Exception as e:
        print(f"❌ Filter Optimizer error: {e}")
    
    # Test 10: Pipeline
    print("\n--- Test 10: Complete Pipeline ---")
    try:
        filtered = create_filter_pipeline(observations, filter_type='ensemble')
        signal = compute_lafo_signal(observations, filter_type='kalman')
        print(f"✅ Pipeline - Ensemble filtered: {filtered[:5]}")
        print(f"   Kalman signal: {signal[:5]}")
    except Exception as e:
        print(f"❌ Pipeline error: {e}")
    
    print("\n" + "=" * 60)
    print("✅ All tests completed!")
    print("=" * 60)
    
    return True

def main():
    """Main entry point."""
    run_basic_test()
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
