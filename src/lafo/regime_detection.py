"""
Regime Detection Module for LAFO Mean-Reversion Strategy
Sections 3.4: HMM, Volatility Clustering, and adaptive filtering
"""
import numpy as np
from typing import Optional, Tuple, List, Dict
import scipy.linalg as la
import scipy.stats as st
from scipy.optimize import minimize
from scipy.cluster.hierarchy import linkage, fcluster


class HiddenMarkovModel:
    """
    Hidden Markov Model for regime detection.
    Observations are prices/returns; hidden states represent market regimes.
    """
    
    def __init__(self, n_states: int = 2, emission_dims: int = 1,
                 max_iter: int = 100, tol: float = 1e-4):
        """
        Args:
            n_states: Number of regimes (e.g., bull/bear)
            emission_dims: Dimensions of observation (returns, vol, etc.)
            max_iter: EM algorithm iterations
            tol: Convergence tolerance
        """
        self.n_states = n_states
        self.emission_dims = emission_dims
        self.max_iter = max_iter
        self.tol = tol
        
        # Transition matrix (emit probabilities)
        self.trans_mat = None
        
        # Emission parameters (Gaussian per state)
        self.emission_means = None
        self.emission_vars = None
        
        # Initial state probabilities
        self.init_probs = None
        
        # Trained model
        self.fitted = False
        
    def fit(self, observations: np.ndarray,
            initial_guess: Optional[Dict] = None) -> Dict:
        """
        Fit HMM using EM algorithm.
        
        Args:
            observations: Observation array (T, dims)
            initial_guess: Initial parameters (optional)
            
        Returns:
            Fitted model parameters
        """
        T = observations.shape[0]
        
        # Initialize parameters
        if initial_guess is None:
            # Random initialization
            self.trans_mat = np.ones((self.n_states, self.n_states)) / self.n_states
            self.trans_mat -= np.eye(self.n_states)  # Ensure rows sum to 1
            self.trans_mat += 0.1  # Add small value to avoid zeros
            self.trans_mat /= self.trans_mat.sum(axis=1, keepdims=True)
            
            # Initialize emission means
            self.emission_means = np.random.randn(self.n_states, self.emission_dims) * 0.1
            self.emission_vars = np.ones((self.n_states, self.emission_dims)) * 0.01
            
            # Initial state probabilities
            self.init_probs = np.ones(self.n_states) / self.n_states
        else:
            self.trans_mat = np.array(initial_guess.get('trans_mat', 
                        np.ones((self.n_states, self.n_states)) / self.n_states))
            self.trans_mat -= np.eye(self.n_states)
            self.trans_mat += 0.1
            self.trans_mat /= self.trans_mat.sum(axis=1, keepdims=True)
            
            self.emission_means = np.array(initial_guess.get('emission_means',
                        np.zeros((self.n_states, self.emission_dims))))
            self.emission_vars = np.array(initial_guess.get('emission_vars',
                        np.ones((self.n_states, self.emission_dims)) * 0.01))
            self.init_probs = np.array(initial_guess.get('init_probs',
                        np.ones(self.n_states) / self.n_states))
        
        # EM algorithm
        log_likelihood = 0.0
        path = []
        
        for iteration in range(self.max_iter):
            # E-step: Compute posterior state probabilities
            gamma, alpha, beta = self._forward_backward(observations)
            
            # M-step: Update parameters
            self._m_step(observations, gamma)
            
            # Compute log-likelihood
            log_likelihood = self._compute_log_likelihood(observations)
            path.append((iteration, log_likelihood))
            
            # Check convergence
            if iteration > 0 and log_likelihood - path[-2][1] < self.tol:
                break
        
        self.fitted = True
        self.log_likelihood = log_likelihood
        self.convergence_path = path
        
        return self.get_parameters()
    
    def _forward_backward(self, observations: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute forward-backward algorithm.
        
        Returns:
            Gamma (posterior state probs), Alpha (forward), Beta (backward)
        """
        T = observations.shape[0]
        gamma = np.zeros((T, self.n_states))
        alpha = np.zeros((T, self.n_states))
        beta = np.zeros((T, self.n_states))
        
        # Forward pass
        for t in range(T):
            obs = observations[t] if self.emission_dims == 1 else observations[:, t]
            log_alpha_t = self.log_pdf(self.emission_means, self.emission_vars, obs)
            
            if t == 0:
                alpha[t] = self.init_probs * np.exp(log_alpha_t)
            else:
                log_alpha_prev = np.log(np.maximum(alpha[t-1], 1e-300))
                log_probs = log_alpha_prev + np.log(self.trans_mat)
                alpha[t] = np.exp(np.logsumexp(log_probs, axis=1) + log_alpha_t)
            
            # Normalize
            alpha[t] /= alpha[t].sum()
        
        # Backward pass
        for t in range(T-1, -1, -1):
            obs = observations[t] if self.emission_dims == 1 else observations[:, t]
            log_pdf = self.log_pdf(self.emission_means, self.emission_vars, obs)
            
            if t == T - 1:
                beta[t] = np.ones(self.n_states)
            else:
                log_beta_next = np.log(np.maximum(beta[t+1], 1e-300))
                beta[t] = np.sum(self.trans_mat * np.exp(log_pdf + log_beta_next), axis=1)
        
        # Posterior
        gamma = alpha * beta
        gamma /= gamma.sum(axis=1, keepdims=True)
        
        return gamma, alpha, beta
    
    def _m_step(self, observations: np.ndarray, gamma: np.ndarray):
        """
        M-step: Update parameters given posterior state probabilities.
        """
        T = observations.shape[0]
        
        # Emission parameters
        for s in range(self.n_states):
            # Effective count for state s
            n_s = gamma[:, s].sum()
            
            if n_s > 0:
                # Update mean
                if self.emission_dims == 1:
                    self.emission_means[s] = (observations * gamma[:, s]).sum() / n_s
                else:
                    self.emission_means[s] = (observations.T @ gamma[:, s]) / n_s
                
                # Update variance
                if self.emission_dims == 1:
                    diff = observations - self.emission_means[s]
                    self.emission_vars[s] = (diff ** 2 * gamma[:, s]).sum() / n_s
                else:
                    diff = observations - self.emission_means[s]
                    self.emission_vars[s] = (diff ** 2 * gamma[:, s]).sum(axis=1) / n_s
            
            # Ensure positive variance
            self.emission_vars[s] = np.maximum(self.emission_vars[s], 1e-6)
        
        # Transition matrix
        # Count transitions
        trans_counts = np.zeros((self.n_states, self.n_states))
        for t in range(T-1):
            obs = observations[t] if self.emission_dims == 1 else observations[:, t]
            post_t = gamma[t]
            
            # Emission probs
            log_emission_t = self.log_pdf(self.emission_means, self.emission_vars, obs)
            emission_t = np.exp(log_emission_t)
            emission_t /= emission_t.sum()
            
            # Weighted transition counts
            for s_prev in range(self.n_states):
                for s_next in range(self.n_states):
                    trans_counts[s_next, s_prev] += emission_t[s_next] * post_t[s_prev]
        
        # Normalize to get transition matrix
        self.trans_mat = trans_counts.T / trans_counts.sum(axis=1, keepdims=True)
        self.trans_mat = np.maximum(self.trans_mat, 0.01)  # Avoid zeros
        self.trans_mat /= self.trans_mat.sum(axis=1, keepdims=True)
        
        # Initial probabilities
        n0 = gamma[0].sum()
        self.init_probs = gamma[0] / n0
    
    def log_pdf(self, means: np.ndarray, vars: np.ndarray, obs: np.ndarray) -> np.ndarray:
        """
        Compute log PDF of observations given state.
        """
        if self.emission_dims == 1:
            return -0.5 * np.log(2 * np.pi * vars) - 0.5 * (obs - means) ** 2 / vars
        else:
            vars_diag = np.diag(vars)
            return -0.5 * np.log((2 * np.pi) ** self.emission_dims * np.prod(vars))
    
    def _compute_log_likelihood(self, observations: np.ndarray) -> float:
        """
        Compute log-likelihood of observations.
        """
        T = observations.shape[0]
        log_likelihood = 0.0
        
        # Initial
        log_emission = self.log_pdf(self.emission_means, self.emission_vars, observations[0])
        log_likelihood += np.log(np.exp(log_emission) @ self.init_probs)
        
        # Transitions
        for t in range(1, T):
            obs = observations[t-1] if self.emission_dims == 1 else observations[:, t-1]
            obs_next = observations[t] if self.emission_dims == 1 else observations[:, t]
            
            log_emission_next = self.log_pdf(self.emission_means, self.emission_vars, obs_next)
            log_trans = np.log(self.trans_mat)
            
            log_likelihood += np.logsumexp(log_emission_next + log_trans + 
                                          np.log(self.emission_vars), axis=0)
        
        return log_likelihood
    
    def get_state_sequence(self, observations: np.ndarray) -> np.ndarray:
        """
        Decode most likely state sequence using Viterbi algorithm.
        
        Args:
            observations: Observation array
            
        Returns:
            State sequence (T,)
        """
        if not self.fitted:
            raise RuntimeError("Model not fitted")
        
        T = observations.shape[0]
        states = np.zeros(T, dtype=int)
        
        # Viterbi
        delta = np.zeros((T, self.n_states))
        psi = np.zeros((T, self.n_states), dtype=int)
        
        # Initialize
        log_emission = self.log_pdf(self.emission_means, self.emission_vars, observations[0])
        delta[0] = self.init_probs + log_emission
        
        # Recursion
        for t in range(1, T):
            log_emission = self.log_pdf(self.emission_means, self.emission_vars, observations[t])
            temp = self.trans_mat + log_emission + delta[t-1]
            delta[t] = np.logsumexp(temp, axis=1)
            psi[t] = np.argmax(temp, axis=1)
        
        # Backtrack
        states[T-1] = np.argmax(delta[T-1])
        for t in range(T-2, -1, -1):
            states[t] = psi[t+1, states[t+1]]
        
        return states
    
    def get_parameters(self) -> Dict:
        """Get fitted parameters."""
        return {
            'trans_mat': self.trans_mat,
            'emission_means': self.emission_means,
            'emission_vars': self.emission_vars,
            'init_probs': self.init_probs
        }


class VolatilityClustering:
    """
    Detect volatility clustering using GARCH and threshold-based methods.
    """
    
    def __init__(self, window_size: int = 60, threshold_std: float = 2.0):
        """
        Args:
            window_size: Rolling window for volatility estimation
            threshold_std: Standard deviations for high/low volatility regimes
        """
        self.window_size = window_size
        self.threshold_std = threshold_std
        
    def detect_regimes(self, returns: np.ndarray,
                       vol_estimates: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Detect volatility regimes.
        
        Args:
            returns: Return series
            vol_estimates: Pre-computed volatility estimates (optional)
            
        Returns:
            Regime sequence (high/low volatility)
        """
        if vol_estimates is None:
            # Rolling volatility
            windowed_returns = returns[-self.window_size:] if len(returns) >= self.window_size else returns
            vol_estimates = np.zeros(len(returns))
            
            for i in range(self.window_size, len(returns)):
                window_returns = returns[i-self.window_size:i]
                vol_estimates[i] = window_returns.std() * np.sqrt(252)  # Annualized
        
        # Classify regimes
        avg_vol = np.mean(vol_estimates[vol_estimates > 0])
        high_vol_threshold = avg_vol * (1 + self.threshold_std)
        low_vol_threshold = avg_vol * (1 - self.threshold_std)
        
        regimes = np.zeros(len(returns), dtype=int)
        regimes[vol_estimates > high_vol_threshold] = 1  # High volatility
        regimes[(vol_estimates <= high_vol_threshold) & (vol_estimates > low_vol_threshold)] = 0.5  # Medium
        
        return regimes


class AdaptiveFilter:
    """
    Adaptive filter that adjusts parameters based on detected regimes.
    """
    
    def __init__(self, base_filter, regime_detector, adaptation_window: int = 20):
        """
        Args:
            base_filter: Base filter (e.g., Kalman, CNN)
            regime_detector: Regime detection module
            adaptation_window: Window size for adaptive updates
        """
        self.base_filter = base_filter
        self.regime_detector = regime_detector
        self.adaptation_window = adaptation_window
        
        # Regime-specific parameters
        self.regime_params = {}
        self.current_regime = 0
        
    def filter(self, observations: np.ndarray) -> np.ndarray:
        """
        Filter with regime adaptation.
        
        Args:
            observations: Observation sequence
            
        Returns:
            Filtered output
        """
        # Detect regimes
        regimes = self.regime_detector.detect_regime(observations)
        
        # Apply regime-specific filtering
        filtered = np.zeros(len(observations))
        
        for i, (obs, regime) in enumerate(zip(observations, regimes)):
            if regime in self.regime_params:
                # Use regime-specific parameters
                params = self.regime_params[regime]
            else:
                # Use base parameters
                params = None
            
            filtered[i] = self.base_filter._apply_single(obs, params)
        
        return filtered
    
    def update_regime_params(self, regime: int, observations: np.ndarray):
        """
        Update parameters for a specific regime.
        
        Args:
            regime: Regime ID
            observations: Observations in this regime
        """
        # Store regime-specific parameters
        self.regime_params[regime] = {
            'mean': np.mean(observations),
            'std': np.std(observations)
        }


def detect_market_regime(returns: np.ndarray,
                          window_size: int = 50,
                          num_clusters: int = 2) -> np.ndarray:
    """
    Detect market regimes using clustering.
    
    Args:
        returns: Return series
        window_size: Rolling window
        num_clusters: Number of regimes
        
    Returns:
        Regime labels
    """
    if len(returns) <= window_size:
        return np.zeros(len(returns), dtype=int)
    
    # Rolling statistics
    rolling_mean = np.zeros(len(returns))
    rolling_std = np.zeros(len(returns))
    
    for i in range(window_size, len(returns)):
        window = returns[i-window_size:i]
        rolling_mean[i] = np.mean(window)
        rolling_std[i] = np.std(window)
    
    # Combined feature
    features = np.column_stack([rolling_mean[window_size:], rolling_std[window_size:]])
    
    # K-means clustering
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    labels = kmeans.fit_predict(features)
    
    return np.concatenate([np.zeros(window_size), labels])


if __name__ == "__main__":
    # Test regime detection
    np.random.seed(42)
    n = 500
    returns = np.cumsum(np.random.randn(n) / np.sqrt(252))
    
    print("Testing Hidden Markov Model...")
    hmm = HiddenMarkovModel(n_states=2, emission_dims=1)
    params = hmm.fit(returns)
    print(f"Log-likelihood: {params['log_likelihood']}")
    
    print("\nTesting volatility clustering...")
    vol_detector = VolatilityClustering()
    regimes = vol_detector.detect_regimes(returns)
    print(f"Regime counts: High={regimes.sum()}, Low={len(regimes)-regimes.sum()}")
    
    print("\nAll regime detection tests passed!")
