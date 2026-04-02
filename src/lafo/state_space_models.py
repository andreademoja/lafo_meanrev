"""
State-Space Models for LAFO Mean-Reversion Strategy
Sections 3.1-3.4: Kalman Filter, Variational Inference, and advanced filtering
"""
import numpy as np
from typing import Optional, Tuple, Dict, List
import scipy.linalg as la


class KalmanFilter:
    """
    Kalman Filter for long-term memory and regime adaptation.
    
    State-space model:
    x_t = F * x_{t-1} + w_t  (state equation)
    y_t = H * x_t + v_t       (observation equation)
    
    Where:
    - x_t: latent state (fundamental level)
    - y_t: observed price/return
    - F: state transition matrix
    - H: observation matrix  
    - w_t, v_t: process and measurement noise
    """
    
    def __init__(self, n_states: int = 1, 
                 state_transition: Optional[np.ndarray] = None,
                 observation_matrix: Optional[np.ndarray] = None,
                 init_state_cov: Optional[np.ndarray] = None):
        """
        Initialize Kalman Filter.
        
        Args:
            n_states: Number of latent states
            state_transition: F matrix (default: identity for random walk)
            observation_matrix: H matrix (default: [1, 0, ...])
            init_state_cov: Initial covariance P_0 (default: identity)
        """
        self.n_states = n_states
        self.state_transition = state_transition or np.eye(n_states)
        self.observation_matrix = observation_matrix or np.array([[1] + [0]*(n_states-1)]).T
        self.init_state_cov = init_state_cov or np.eye(n_states)
        
        # Running estimates
        self.state = None
        self.state_cov = None
        
    def predict(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict step: x_t|t-1 = F * x_{t-1}, P_t|t-1 = F * P_{t-1} * F^T + Q
        """
        if self.state is None:
            # Initialize state as zero
            self.state = np.zeros(self.n_states)
            self.state_cov = self.init_state_cov.copy()
        
        # Predict state
        predicted_state = self.state_transition @ self.state
        
        # Predict covariance (Q is process noise, default to small value)
        Q = np.eye(self.n_states) * 0.01  # Small process noise
        predicted_cov = self.state_transition @ self.state_cov @ self.state_transition.T + Q
        
        return predicted_state, predicted_cov
    
    def update(self, observation: np.ndarray,
               measurement_noise: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Update step: Incorporate observation y_t
        
        Args:
            observation: Observed value (price/return)
            measurement_noise: Measurement noise variance (R)
        
        Returns:
            Updated state estimate and covariance
        """
        H = self.observation_matrix
        R = measurement_noise
        
        # Predict step
        x_pred, P_pred = self.predict()
        
        # Innovation
        y_pred = H @ x_pred
        innovation = observation - y_pred
        
        # Innovation covariance
        S = H @ P_pred @ H.T + R
        
        # Kalman gain
        K = P_pred @ H.T @ la.inv(S)
        
        # Updated state
        self.state = x_pred + K @ innovation
        
        # Updated covariance  
        P = (np.eye(self.n_states) - K @ H) @ P_pred
        
        return self.state, P
    
    def compute_filtered_mean(self, observations: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run filter on entire observation sequence.
        
        Args:
            observations: Array of observed values
        
        Returns:
            State estimates and covariances
        """
        n = len(observations)
        filtered_means = np.zeros(n)
        filtered_covs = np.zeros((n, self.n_states, self.n_states))
        
        self.state = None  # Reset for new sequence
        for t, y in enumerate(observations):
            self.update(y)
            filtered_means[t] = self.state[0]  # Take first state as fundamental
            filtered_covs[t] = self.state_cov
        
        return filtered_means, filtered_covs


class SwitchingKalmanFilter:
    """
    Switching Kalman Filter for regime detection.
    Uses HMM to switch between different market regimes.
    """
    
    def __init__(self, n_regimes: int = 2, 
                 regime_noise_std: float = 0.002,
                 switching_prob: float = 0.05):
        """
        Args:
            n_regimes: Number of market regimes (e.g., bull/bear)
            regime_noise_std: Volatility of regime indicator
            switching_prob: Probability of regime switch
        """
        self.n_regimes = n_regimes
        self.regression_noise_std = regime_noise_std
        self.switching_prob = switching_prob
        self.state = None
        self.state_cov = None
        
    def predict(self, prev_regime_probs: np.ndarray,
                observations: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict step with regime switching.
        
        Args:
            prev_regime_probs: Previous regime probabilities
            observations: Current observations
        
        Returns:
            Predicted state probabilities and covariance
        """
        n = len(observations)
        filtered_means = np.zeros((n, self.n_regimes))
        filtered_covs = np.zeros((n, self.n_regimes))
        
        # Initialize uniform regime probabilities
        self.state = np.ones(self.n_regimes) / self.n_regimes
        self.state_cov = np.eye(self.n_regimes)
        
        for t, obs in enumerate(observations):
            # Apply regime switching
            if t > 0:
                switch = np.random.random() < self.switching_prob
                if switch:
                    # Random regime switch
                    self.state = np.ones(self.n_regimes) / self.n_regimes
            else:
                # Smooth transition from previous
                self.state = prev_regime_probs.copy()
            
            # Update state based on observation
            obs_norm = obs  # Normalize if needed
            # Update regime weights
            diff = obs - self.state.mean()
            self.state += diff / obs.std() * self.regression_noise_std if obs.std() > 0 else 0
            
            # Renormalize
            self.state = np.maximum(self.state, 0.01)
            self.state = self.state / self.state.sum()
            
            # Compute filtered value
            filtered_means[t] = self.state[0]  # First regime weight
            
        return filtered_means, filtered_covs


class VariationalInferenceFilter:
    """
    Variational Inference filter for approximate Bayesian inference.
    Approximates posterior p(x_t|y_{1:t}) with q(x_t) = N(mu_t, sigma_t^2).
    """
    
    def __init__(self, signal_to_noise_ratio: float = 0.5,
                 decay_rate: float = 0.95):
        """
        Args:
            signal_to_noise_ratio: S/N ratio for the filter
            decay_rate: Exponential decay for mean-reversion
        """
        self.signal_to_noise_ratio = signal_to_noise_ratio
        self.decay_rate = decay_rate
        
        # Variational parameters
        self.var_mean = 0.0
        self.var_var = 1.0  # Variance
        self.var_cov = 0.0  # Covariance with previous
        
    def update(self, observation: np.ndarray,
               prev_estimate: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Update variational approximation.
        
        Args:
            observation: New observation
            prev_estimate: Previous state estimate
        
        Returns:
            Updated variational mean
        """
        # Kalman-like update
        signal = self.signal_to_noise_ratio
        noise = 1.0 - self.signal_to_noise_ratio
        
        if prev_estimate is not None:
            prev = prev_estimate
        else:
            prev = 0.0
        
        # Variational Bayes update
        innovation = observation - prev
        weight = signal / (signal + noise)
        
        self.var_mean = weight * observation + (1 - weight) * prev
        self.var_var = noise / signal if signal > 0 else 1.0
        
        return self.var_mean


class EnsembleFilter:
    """
    Ensemble averaging framework combining multiple filter outputs.
    """
    
    def __init__(self, filters: List, weights: Optional[np.ndarray] = None):
        """
        Args:
            filters: List of filter instances (Kalman, VI, etc.)
            weights: Relative weights for ensemble (default: equal)
        """
        self.filters = filters
        if weights is None:
            self.weights = np.ones(len(filters)) / len(filters)
        else:
            self.weights = np.asarray(weights) / weights.sum()
            
        self.ensemble_mean = 0.0
        self.ensemble_var = 1.0
        
    def update(self, observations: np.ndarray) -> np.ndarray:
        """
        Update ensemble with new observations.
        
        Args:
            observations: New observation values
        
        Returns:
            Ensemble mean
        """
        filtered_outputs = []
        
        for i, filter_obj in enumerate(self.filters):
            if hasattr(filter_obj, 'compute_filtered_mean'):
                means, _ = filter_obj.compute_filtered_mean(observations)
                # Use last estimate
                filtered_outputs.append(means[-1])
            elif hasattr(filter_obj, 'update'):
                result = filter_obj.update(observations[-1])
                filtered_outputs.append(result)
            else:
                filtered_outputs.append(observations[-1])
        
        # Ensemble average
        self.ensemble_mean = np.array(filtered_outputs).T @ self.weights
        self.ensemble_var = self.weights.var() + np.var(filtered_outputs)
        
        return self.ensemble_mean
    
    def get_ensemble_state(self) -> np.ndarray:
        """Get current ensemble state."""
        return self.ensemble_mean


def compute_state_space_filter(observations: np.ndarray,
                                filter_type: str = 'kalman',
                                params: Optional[Dict] = None) -> np.ndarray:
    """
    Compute filtered fundamental level using state-space models.
    
    Args:
        observations: Price/return series
        filter_type: 'kalman', 'switching', 'variational', 'ensemble'
        params: Filter-specific parameters
    
    Returns:
        Filtered fundamental levels
    """
    params = params or {}
    
    if filter_type == 'kalman':
        kf = KalmanFilter(**params)
        means, _ = kf.compute_filtered_mean(observations)
        return means
    
    elif filter_type == 'switching':
        skf = SwitchingKalmanFilter(**params)
        means, _ = skf.predict(np.array([]), observations)
        return means
    
    elif filter_type == 'variational':
        vi = VariationalInferenceFilter(**params)
        output = observations[0]  # Initialize
        for obs in observations[1:]:
            output = vi.update(obs, output)
        return np.full_like(observations, output)
    
    else:
        raise ValueError(f"Unknown filter type: {filter_type}")
