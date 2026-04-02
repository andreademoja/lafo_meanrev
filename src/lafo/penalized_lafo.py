import numpy as np
from scipy.optimize import minimize

# Import from same package directory
from src.lafo import build_sliding_average_operator, compute_W

def lafo_l2_closed_form(y: np.ndarray, K: int, lambda_: float) -> np.ndarray:
    """
    LAFO-ℓ₂ closed-form solution (eq. 2.5 del paper).
    """
    T = len(y)
    W = compute_W(K, T)
    # First difference operator D
    D = np.eye(T) - np.eye(T, k=-1)
    D = D[1:, :]                    # shape (T-1, T)
    # Linear system: (W/T + λ D^T D) x = W y / T
    A = (W / T) + lambda_ * (D.T @ D)
    b = (W @ y) / T
    return np.linalg.solve(A, b)


def lafo_tv_solver(y: np.ndarray, K: int, lambda_: float) -> np.ndarray:
    """LAFO-TV (Total Variation ℓ1 on first differences)"""
    def objective(x):
        e = y - x
        A = build_sliding_average_operator(K, len(y))
        lafo_term = np.sum((A @ e)**2) / len(y)
        tv_term = lambda_ * np.sum(np.abs(np.diff(x)))
        return lafo_term + tv_term

    res = minimize(objective, y, method='L-BFGS-B')
    return res.x


def lafo_l1_tf_solver(y: np.ndarray, K: int, lambda_: float) -> np.ndarray:
    """LAFO-ℓ₁ Trend Filtering (ℓ1 on second differences)"""
    def objective(x):
        e = y - x
        A = build_sliding_average_operator(K, len(y))
        lafo_term = np.sum((A @ e)**2) / len(y)
        tf_term = lambda_ * np.sum(np.abs(np.diff(x, n=2)))
        return lafo_term + tf_term

    res = minimize(objective, y, method='L-BFGS-B')
    return res.x


# Test rapido
if __name__ == "__main__":
    np.random.seed(42)
    T = 200
    K = 20
    y = np.cumsum(np.random.randn(T)) * 0.1 + np.linspace(0, 10, T)
    lambda_ = 5.0

    x_l2 = lafo_l2_closed_form(y, K, lambda_)
    print("LAFO-ℓ₂ closed-form computed successfully")
    print("Shape:", x_l2.shape)