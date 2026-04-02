"""
Test script for state-space models and ensemble filtering
"""
import numpy as np
import sys
sys.path.insert(0, 'lafo_meanrev/src')

from lafo.state_space_models import (
    KalmanFilter,
    SwitchingKalmanFilter,
    VariationalInferenceFilter
)
from lafo.advanced_cnn import ARMABlock, MambaFilter
from lafo.ensemble import EnsembleFilter
from lafo.regime_detection import HiddenMarkovModel, detect_market_regime

def generate_test_data(n=1000):
    """Generate synthetic mean-reversion data."""
    np.random.seed(42)
    # Trend with mean reversion
    trend = np.cumsum(np.random.randn(n) * 0.1 + 0.01)
    noise = np.random.randn(n) * 0.1
    observations = trend + noise
    return observations

def test_kalman_filter():
    """Test Kalman filter."""
    print("Testing Kalman Filter...")
    observations = generate_test_data(500)
    kf = KalmanFilter()
    filtered, cov = kf.compute_filtered_mean(observations)
    print(f"  Filtered shape: {filtered.shape}")
    print(f"  First 5 filtered values: {filtered[:5]}")
    return True

def test_switching_kalman():
    """Test switching Kalman filter."""
    print("\nTesting Switching Kalman Filter...")
    observations = generate_test_data(500)
    skf = SwitchingKalmanFilter(n_regimes=2, switching_prob=0.1)
    filtered, _ = skf.predict(observations)
    print(f"  Filtered shape: {filtered.shape}")
    return True

def test_variational():
    """Test variational inference filter."""
    print("\nTesting Variational Inference Filter...")
    observations = generate_test_data(500)
    vi = VariationalInferenceFilter(signal_to_noise_ratio=0.5)
    filtered = vi.compute(observations)
    print(f"  Filtered shape: {filtered.shape}")
    return True

def test_armablock():
    """Test ARMABlock CNN filter."""
    print("\nTesting ARMABlock CNN Filter...")
    observations = generate_test_data(500)
    import torch
    filter = ARMABlock(channels=64, kernel_size=31)
    obs_torch = torch.FloatTensor(observations).unsqueeze(0).unsqueeze(1)
    output = filter(obs_torch).squeeze().detach().numpy()
    print(f"  Filtered shape: {output.shape}")
    return True

def test_ensemble():
    """Test ensemble filter."""
    print("\nTesting Ensemble Filter...")
    observations = generate_test_data(500)
    ensemble = EnsembleFilter()
    filtered = ensemble.compute(observations)
    print(f"  Filtered shape: {filtered.shape}")
    return True

def test_hmm():
    """Test Hidden Markov Model."""
    print("\nTesting Hidden Markov Model...")
    observations = generate_test_data(500)
    hmm = HiddenMarkovModel(n_states=2, emission_dims=1)
    params = hmm.fit(observations)
    state_sequence = hmm.get_state_sequence(observations)
    print(f"  States detected: {state_sequence[:10]}")
    print(f"  High volatility periods: {state_sequence.sum()}")
    return True

def main():
    """Run all tests."""
    print("=" * 60)
    print("LAFO State-Space Models Test Suite")
    print("=" * 60)
    
    try:
        test_kalman_filter()
        test_switching_kalman()
        test_variational()
        test_armablock()
        test_ensemble()
        test_hmm()
        
        print("\n" + "=" * 60)
        print("✅ All state-space model tests passed!")
        print("=" * 60)
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
