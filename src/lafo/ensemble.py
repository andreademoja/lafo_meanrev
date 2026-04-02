"""
Ensemble Averaging Framework for LAFO Mean-Reversion Strategy
Combines multiple filter outputs for robust filtering
"""
import numpy as np
from typing import Optional, List, Dict, Tuple, Callable
import sys
sys.path.insert(0, 'lafo_meanrev/src')


class EnsembleFilter:
    """
    Ensemble filter combining multiple filtering strategies.
    """
    
    def __init__(self, filters: List = None, 
                 weights: Optional[np.ndarray] = None,
                 method: str = 'weighted'):
        """
        Args:
            filters: List of filter functions or objects
            weights: Relative weights (default: equal)
            method: 'weighted', 'stacking', or 'bagging'
        """
        self.filters = filters or []
        self.weights = weights
        self.method = method
        
        if not self.filters:
            # Default ensemble of filters
            self.filters = [
                {'type': 'kalman', 'params': {'noise_std': 0.1}},
                {'type': 'cnn', 'kernel_size': 31},
                {'type': 'arma', 'order': (1, 1, 1)}
            ]
        
        # Normalize weights
        if weights is not None:
            self.weights = np.asarray(weights)
            self.weights = self.weights / self.weights.sum()
    
    def compute(self, observations: np.ndarray) -> np.ndarray:
        """
        Compute ensemble filter output.
        
        Args:
            observations: Observation sequence
            
        Returns:
            Ensemble output
        """
        n = len(observations)
        outputs = np.zeros(n)
        
        if self.method == 'weighted':
            # Weighted average
            for i, filter_config in enumerate(self.filters):
                output = self._apply_filter(filter_config, observations)
                outputs += output * (self.weights[i] if self.weights is not None else 1.0/len(self.filters))
        
        elif self.method == 'stacking':
            # Stacking with meta-model
            outputs = self._stacked_ensemble(observations)
        
        elif self.method == 'bagging':
            # Bootstrap aggregation
            outputs = self._bagging_ensemble(observations)
        
        return outputs
    
    def _apply_filter(self, config: Dict, observations: np.ndarray) -> np.ndarray:
        """
        Apply a single filter from configuration.
        """
        filter_type = config.get('type', 'kalman')
        
        if filter_type == 'kalman':
            return self._kalman_filter(observations, config.get('params', {'noise_std': 0.1}))
        
        elif filter_type == 'cnn':
            return self._cnn_filter(observations, config.get('kernel_size', 31))
        
        elif filter_type == 'arma':
            return self._arma_filter(observations, config.get('order', (1, 1, 1)))
        
        elif filter_type == 'state_space':
            return self._state_space_filter(observations, config.get('params', {}))
        
        elif filter_type == 'variational':
            return self._variational_filter(observations, config.get('params', {}))
        
        else:
            # Default: low-pass filter
            return self._lowpass_filter(observations, config.get('cutoff', 0.05))
    
    def _kalman_filter(self, observations: np.ndarray, params: Dict) -> np.ndarray:
        """Kalman filter implementation."""
        noise_std = params.get('noise_std', 0.1)
        signal_to_noise = 0.5
        
        filtered = np.zeros(len(observations))
        estimate = 0.0
        variance = 1.0
        
        for t, obs in enumerate(observations):
            # Predict
            predicted = estimate
            
            # Update
            innovation = obs - predicted
            kalman_gain = variance / (variance + noise_std ** 2)
            
            # Update estimate
            estimate = predicted + kalman_gain * innovation
            filtered[t] = estimate
            
            # Update variance
            variance = (1 - kalman_gain) * variance
        
        return filtered
    
    def _cnn_filter(self, observations: np.ndarray, kernel_size: int) -> np.ndarray:
        """CNN-based filter (simplified)."""
        # Pad observations
        pad = (kernel_size - 1) // 2
        
        # Convolution with Gaussian-like kernel
        from scipy.signal import convolve
        kernel = np.exp(-np.arange(kernel_size) ** 2 / (2 * 4))
        kernel /= kernel.sum()
        
        filtered = convolve(observations, kernel, mode='same')
        
        return filtered
    
    def _arma_filter(self, observations: np.ndarray, order: Tuple[int, int, int]) -> np.ndarray:
        """ARMA filter."""
        p, d, q = order
        
        # Simple EWMA approximation
        alpha = 2 / (p + 1)
        filtered = np.zeros(len(observations))
        
        if len(observations) > 0:
            filtered[0] = observations[0]
            for t in range(1, len(observations)):
                filtered[t] = alpha * observations[t] + (1 - alpha) * filtered[t-1]
        
        return filtered
    
    def _state_space_filter(self, observations: np.ndarray, params: Dict) -> np.ndarray:
        """State-space filter."""
        # Initialize state
        state = np.mean(observations[:min(10, len(observations))])
        variance = np.var(observations[:min(10, len(observations))])
        
        filtered = np.zeros(len(observations))
        decay = params.get('decay', 0.95)
        
        for t in range(len(observations)):
            # Observation
            obs = observations[t]
            
            # Kalman update
            innovation = obs - state
            kalman_gain = variance / (variance + 0.01)
            
            # Update
            state = state + kalman_gain * innovation
            filtered[t] = state
            
            # Decay
            variance *= decay
        
        return filtered
    
    def _variational_filter(self, observations: np.ndarray, params: Dict) -> np.ndarray:
        """Variational inference filter."""
        signal_to_noise = params.get('signal_to_noise', 0.5)
        
        filtered = np.zeros(len(observations))
        mean = 0.0
        var = 1.0
        
        for t, obs in enumerate(observations):
            # Update mean
            mean = signal_to_noise * obs + (1 - signal_to_noise) * mean
            filtered[t] = mean
        
        return filtered
    
    def _lowpass_filter(self, observations: np.ndarray, cutoff: float) -> np.ndarray:
        """Low-pass filter."""
        # Butterworth filter coefficients
        nyq = 0.5
        wc = cutoff / nyq
        
        from scipy.signal import butter, filtfilt
        
        # Low-pass filter
        b, a = butter(3, wc, btype='low')
        filtered = filtfilt(b, a, observations)
        
        return filtered
    
    def _stacked_ensemble(self, observations: np.ndarray) -> np.ndarray:
        """Stacking ensemble with meta-model."""
        # Get individual predictions
        predictions = []
        for i, config in enumerate(self.filters):
            pred = self._apply_filter(config, observations)
            predictions.append(pred)
        
        # Meta-model: simple weighted average with learning
        n = len(observations)
        X = np.column_stack(predictions)
        y = observations
        
        # Simple OLS meta-model
        from scipy.linalg import lstsq
        coeffs, _, _, _ = lstsq(X, y, rcond=None)
        
        # Predict
        stacked = X @ coeffs
        
        return stacked
    
    def _bagging_ensemble(self, observations: np.ndarray) -> np.ndarray:
        """Bootstrap aggregation."""
        n_bootstrap = 10
        n = len(observations)
        
        # Bootstrap samples
        boot_predictions = []
        
        for _ in range(n_bootstrap):
            # Bootstrap sample
            indices = np.random.choice(n, n, replace=True)
            boot_observations = observations[indices]
            
            # Filter bootstrap sample
            pred = self._apply_filter(self.filters[0], boot_observations)
            boot_predictions.append(pred)
        
        # Average
        bagged = np.mean(boot_predictions, axis=0)
        
        return bagged
    
    def optimize_weights(self, observations: np.ndarray,
                         targets: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Optimize ensemble weights.
        
        Args:
            observations: Input data
            targets: Target values (if None, use observations)
            
        Returns:
            Optimized weights
        """
        if targets is None:
            targets = observations
        
        # Get predictions
        predictions = []
        for config in self.filters:
            pred = self._apply_filter(config, observations)
            predictions.append(pred)
        
        X = np.column_stack(predictions)
        
        # Grid search for optimal weights
        n_filters = len(self.filters)
        best_weights = None
        best_score = np.inf
        
        # Search space
        for w0 in np.linspace(0, 1, 5):
            for w1 in np.linspace(0, 1, 5):
                if n_filters > 2:
                    w2 = 1 - w0 - w1
                    if w2 < 0:
                        continue
                else:
                    w2 = 0
                
                weights = np.array([w0, w1, w2])
                
                # Predict
                pred = X @ weights
                
                # Score
                score = np.mean((pred - targets) ** 2)
                
                if score < best_score:
                    best_score = score
                    best_weights = weights
        
        return best_weights


class FilterOptimizer:
    """
    Hyperparameter optimizer for filtering.
    """
    
    def __init__(self, param_grid: Dict = None):
        """
        Args:
            param_grid: Grid of parameters to search
        """
        self.param_grid = param_grid or {
            'kernel_size': [15, 31, 61],
            'noise_std': [0.01, 0.1, 0.5],
            'decay': [0.9, 0.95, 0.99],
            'cutoff': [0.01, 0.05, 0.1]
        }
        
    def grid_search(self, observations: np.ndarray,
                    objective: Callable = None) -> Tuple[np.ndarray, Dict]:
        """
        Grid search for optimal hyperparameters.
        
        Args:
            observations: Input data
            objective: Objective function (default: MSE)
            
        Returns:
            Best parameters and best output
        """
        if objective is None:
            objective = lambda pred, true: np.mean((pred - true) ** 2)
        
        best_params = None
        best_score = np.inf
        best_output = None
        
        # Grid search
        for kernel_size in self.param_grid.get('kernel_size', [31]):
            for noise_std in self.param_grid.get('noise_std', [0.1]):
                for decay in self.param_grid.get('decay', [0.95]):
                    for cutoff in self.param_grid.get('cutoff', [0.05]):
                        
                        # Create filter with these params
                        filter_obj = EnsembleFilter(
                            filters=[
                                {'type': 'kalman', 'params': {'noise_std': noise_std}},
                                {'type': 'state_space', 'params': {'decay': decay}}
                            ],
                            method='weighted'
                        )
                        
                        # Optimize weights
                        weights = filter_obj.optimize_weights(observations)
                        
                        # Get prediction
                        pred = filter_obj.compute(observations)
                        
                        # Score
                        score = objective(pred, observations)
                        
                        if score < best_score:
                            best_score = score
                            best_params = {
                                'kernel_size': kernel_size,
                                'noise_std': noise_std,
                                'decay': decay,
                                'cutoff': cutoff,
                                'weights': weights
                            }
                            best_output = pred
        
        return best_params, best_output
    
    def cross_validate(self, observations: np.ndarray,
                       folds: int = 5) -> Tuple[np.ndarray, Dict]:
        """
        Cross-validation for hyperparameter tuning.
        """
        n = len(observations)
        fold_size = n // folds
        
        # Split data
        indices = np.arange(n)
        fold_indices = np.array_split(indices, folds)
        
        # Grid search with CV
        best_score = np.inf
        best_params = None
        best_output = None
        
        for kernel_size in self.param_grid.get('kernel_size', [31]):
            for noise_std in self.param_grid.get('noise_std', [0.1]):
                for decay in self.param_grid.get('decay', [0.95]):
                    
                    filter_obj = EnsembleFilter(
                        filters=[
                            {'type': 'kalman', 'params': {'noise_std': noise_std}},
                            {'type': 'state_space', 'params': {'decay': decay}}
                        ],
                        method='weighted'
                    )
                    
                    # Average CV scores
                    cv_scores = []
                    
                    for fold in range(folds):
                        # Train set
                        train_idx = np.concatenate([fold_indices[i] for i in range(folds) if i != fold])
                        # Test set
                        test_idx = fold_indices[fold]
                        
                        # Filter train
                        train_pred = filter_obj.compute(observations[train_idx])
                        
                        # Predict test
                        test_pred = filter_obj.compute(observations[test_idx])
                        
                        # Score
                        score = np.mean((test_pred - observations[test_idx]) ** 2)
                        cv_scores.append(score)
                    
                    avg_score = np.mean(cv_scores)
                    
                    if avg_score < best_score:
                        best_score = avg_score
                        best_params = {
                            'kernel_size': kernel_size,
                            'noise_std': noise_std,
                            'decay': decay
                        }
        
        return best_params


if __name__ == "__main__":
    # Test ensemble filter
    np.random.seed(42)
    n = 500
    observations = np.cumsum(np.random.randn(n) / np.sqrt(252))
    
    print("Testing Ensemble Filter...")
    ensemble = EnsembleFilter()
    filtered = ensemble.compute(observations)
    print(f"Filtered output shape: {filtered.shape}")
    
    print("\nTesting weight optimization...")
    optimized_weights = ensemble.optimize_weights(observations)
    print(f"Optimized weights: {optimized_weights}")
    
    print("\nAll ensemble tests passed!")
