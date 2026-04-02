import numpy as np

def build_sliding_average_operator(K: int, T: int) -> np.ndarray:
    """
    Build the causal sliding average operator matrix A (Definition 2.1).
    A[t, start:end] = 1 / n_t with proper causal boundary handling.
    """
    A = np.zeros((T, T))
    for t in range(T):
        start = max(0, t - K + 1)
        end = t + 1
        n = end - start
        A[t, start:end] = 1.0 / n
    return A


def compute_W(K: int, T: int) -> np.ndarray:
    """Compute W = A^T A (Definition 2.3)."""
    A = build_sliding_average_operator(K, T)
    return A.T @ A


def lafo_loss(y: np.ndarray, hat_y: np.ndarray, K: int) -> float:
    """
    LAFO loss - matrix version (exactly eq. 2.2)
    """
    T = len(y)
    A = build_sliding_average_operator(K, T)
    e = y - hat_y
    return np.sum((A @ e) ** 2) / T


def lafo_loss_efficient(y: np.ndarray, hat_y: np.ndarray, K: int) -> float:
    """
    Efficient LAFO loss using cumsum (O(T) time, no full matrix).
    Matches exactly Definition 2.2 and lafo_loss.
    """
    T = len(y)
    e = y - hat_y
    cum_e = np.concatenate(([0.], np.cumsum(e)))
    sq_sum = 0.0
    for t in range(T):
        start = max(0, t - K + 1)
        end = t + 1
        n = end - start
        local_sum = cum_e[end] - cum_e[start]
        local_avg = local_sum / n
        sq_sum += local_avg ** 2
    return sq_sum / T


# Test rapido (puoi rimuoverlo dopo)
if __name__ == "__main__":
    np.random.seed(42)
    T = 100
    K = 10
    y = np.random.randn(T)
    hat_y = np.random.randn(T) * 0.5
    print("LAFO matrix version    :", lafo_loss(y, hat_y, K))
    print("LAFO efficient version :", lafo_loss_efficient(y, hat_y, K))