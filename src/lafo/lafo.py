"""
Core LAFO (Local Average Filtering Objective) functions
Sections 2.1-2.3: LAFO loss computation
"""
import numpy as np
import sys
sys.path.insert(0, 'lafo_meanrev/src')

def build_sliding_average_operator(K: int, T: int) -> np.ndarray:
    """
    Build the sliding average operator A (Section 2.1).
    
    Args:
        K: Sliding window size
        T: Time series length
        
    Returns:
        Matrix A where A_{t,a_t:b_t} = 1/n_t
    """
    A = np.zeros((T, T))
    for t in range(1, T + 1):
        a_t = max(1, t - K + 1)
        b_t = t
        n_t = b_t - a_t + 1
        A[a_t-1:b_t, a_t-1:b_t] = 1.0 / n_t
    return A


def compute_W(K: int, T: int) -> np.ndarray:
    """
    Compute the weight matrix W (Section 2.2).
    
    Args:
        K: Sliding window size
        T: Time series length
        
    Returns:
        Weight matrix W
    """
    W = np.zeros((T, T))
    for t in range(1, T + 1):
        a_t = max(1, t - K + 1)
        b_t = t
        n_t = b_t - a_t + 1
        W[a_t-1:b_t, a_t-1:b_t] = 1.0 / n_t
    return W


def lafo_loss(y: np.ndarray, y_hat: np.ndarray, K: int) -> float:
    """
    Compute LAFO loss for y_hat vs y with window size K.
    Section 2.3: LAFO loss function.
    
    Args:
        y: Observed series (shape: T,)
        y_hat: Filtered series (shape: T,)
        K: Sliding window size
        
    Returns:
        LAFO loss (float)
    """
    T = len(y)
    loss = 0.0
    for t in range(1, T + 1):
        a_t = max(1, t - K + 1)
        b_t = t
        n_t = b_t - a_t + 1
        residual = y[a_t-1:b_t] - y_hat[a_t-1:b_t]
        local_avg_residual = np.mean(residual)
        loss += local_avg_residual ** 2
    return loss / T


def lafo_loss_efficient(y: np.ndarray, y_hat: np.ndarray, K: int) -> float:
    """
    Efficient O(T) computation of LAFO loss.
    
    Args:
        y: Observed series
        y_hat: Filtered series
        K: Sliding window size
        
    Returns:
        LAFO loss
    """
    T = len(y)
    A = build_sliding_average_operator(K, T)
    e = y - y_hat
    local_averages = A @ e
    return np.sum(local_averages ** 2) / T
