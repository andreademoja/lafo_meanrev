import numpy as np
from scipy.stats import t
import yfinance as yf
from datetime import datetime

def generate_piecewise_trendarma(
    T: int = 1200,
    R: int = 5,
    p: int = 2,
    q: int = 1,
    seed: int = 42,
    df: int = 4
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Funzione sintetica originale (per test)"""
    np.random.seed(seed)

    change_points = np.sort(np.random.choice(T, R-1, replace=False))
    regime_starts = np.concatenate(([0], change_points, [T]))
    regime_labels = np.zeros(T, dtype=int)
    for r in range(R):
        regime_labels[regime_starts[r]:regime_starts[r+1]] = r

    beta0 = np.array([0.0, 5.0, -3.0, 8.0, 2.0])
    beta1 = np.array([0.02, -0.015, 0.025, -0.018, 0.01])
    ar_coefs = np.array([[0.6, -0.3], [0.4, 0.2], [-0.5, 0.1], [0.7, -0.4], [0.3, -0.2]])
    ma_coefs = np.array([[-0.4], [-0.5], [0.3], [-0.2], [-0.6]])

    y = np.zeros(T)
    xt = np.zeros(T)
    true_fair_value = np.zeros(T)
    innovations = np.zeros(T)

    for time in range(T):
        r = regime_labels[time]
        true_fair_value[time] = beta0[r] + beta1[r] * time

        ar_term = 0.0
        for i in range(p):
            if time - i - 1 >= 0:
                ar_term += ar_coefs[r, i] * xt[time - i - 1]

        ma_term = 0.0
        for j in range(q):
            if time - j - 1 >= 0:
                ma_term += ma_coefs[r, j] * innovations[time - j - 1]

        innovation = t.rvs(df=df, scale=1.0)
        innovations[time] = innovation
        xt[time] = ar_term + ma_term + innovation
        y[time] = true_fair_value[time] + xt[time]

    return y, true_fair_value, regime_labels


def load_real_sp500(start_date: str = "2024-01-01") -> np.ndarray:
    """
    Scarica i prezzi di chiusura giornalieri dello SP500 dal 1 gennaio 2024 ad oggi.
    """
    print(f"Scaricamento dati SP500 da {start_date} ad oggi...")
    data = yf.download("^GSPC", start=start_date, end=datetime.today().strftime('%Y-%m-%d'), progress=False)
    prices = data['Close'].values.astype(float)
    print(f"Scaricati {len(prices)} giorni di dati SP500")
    return prices


# Test rapido
if __name__ == "__main__":
    prices = load_real_sp500()
    print("Prezzo primo giorno:", prices[0])
    print("Prezzo ultimo giorno:", prices[-1])