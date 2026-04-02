import numpy as np
import pandas as pd

def mean_reversion_backtest(
    y: np.ndarray,
    filtered: np.ndarray,
    entry_threshold: float = 3.0,
    tp_percent: float = 3.5,
    sl_percent: float = 1.0,
    capital: float = 3000.0,
    risk_per_trade: float = 0.5,     
    leverage: float = 500.0            # LEVA 500x REALE
) -> dict:
    """
    Versione con position sizing realistico + leva 500x REALE + Max Drawdown.
    """
    y_list = [float(x) for x in np.asarray(y).flatten()]
    filtered_list = [float(x) for x in np.asarray(filtered).flatten()]
    n = len(y_list)

    positions = [0.0] * n
    pnl = [0.0] * n
    equity = [0.0] * n
    equity[0] = float(capital)
    entry_price = [0.0] * n
    position_dollar_size = [0.0] * n   # esposizione lorda

    num_trades = 0
    num_long = 0
    num_short = 0

    peak_equity = float(capital)
    max_drawdown = 0.0

    for t in range(1, n):
        current_spread = y_list[t] - filtered_list[t]

        if positions[t-1] == 0.0:
            if current_spread < -entry_threshold or current_spread > entry_threshold:
                risk_amount = equity[t-1] * risk_per_trade
                sl_distance = sl_percent / 100.0

                # Dimensione della posizione con leva reale
                position_size = (risk_amount / sl_distance) * leverage

                if current_spread < -entry_threshold:
                    positions[t] = 1.0
                    num_long += 1
                else:
                    positions[t] = -1.0
                    num_short += 1

                entry_price[t] = y_list[t]
                position_dollar_size[t] = position_size
                num_trades += 1

        else:
            positions[t] = positions[t-1]
            entry_price[t] = entry_price[t-1]
            position_dollar_size[t] = position_dollar_size[t-1]

        # Exit
        if positions[t] != 0.0:
            price_change_pct = (y_list[t] - entry_price[t]) / entry_price[t] * 100
            if positions[t] == 1.0:
                if price_change_pct >= tp_percent or price_change_pct <= -sl_percent:
                    positions[t] = 0.0
            else:
                if price_change_pct <= -tp_percent or price_change_pct >= sl_percent:
                    positions[t] = 0.0

        if abs(current_spread) < 0.8 and positions[t] != 0.0:
            positions[t] = 0.0

        # PnL con leva REALE
        ret = (y_list[t] - y_list[t-1]) / max(abs(y_list[t-1]), 1e-8)
        trade_cost = 0.0005 * abs(positions[t] - positions[t-1])
        pnl[t] = positions[t] * ret * leverage - trade_cost

        pnl[t] = max(-0.10, min(0.10, pnl[t]))   # clipping più largo per leva alta

        equity[t] = equity[t-1] * (1.0 + pnl[t])

        # === MAX DRAWDOWN ===
        peak_equity = max(peak_equity, equity[t])
        current_dd = (equity[t] - peak_equity) / peak_equity if peak_equity > 0 else 0.0
        max_drawdown = min(max_drawdown, current_dd)

    # Metriche
    returns = pd.Series(pnl).dropna()
    sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() != 0 else 0.0
    turnover = sum(abs(positions[i] - positions[i-1]) for i in range(1, n)) / (n - 1)
    total_return = (equity[-1] / equity[0] - 1) * 100 if equity[0] != 0 else 0.0

    dollar_sizes = [position_dollar_size[t] for t in range(n) if position_dollar_size[t] > 0]

    return {
        "equity_curve": equity,
        "positions": positions,
        "pnl": pnl,
        "sharpe_ratio": sharpe,
        "turnover": turnover,
        "total_return_pct": total_return,
        "max_drawdown_pct": round(max_drawdown * 100, 2),   # ← AGGIUNTO
        "num_trades": num_trades,
        "num_long": num_long,
        "num_short": num_short,
        "avg_position_dollar": round(np.mean(dollar_sizes), 2) if dollar_sizes else 0,
        "max_position_dollar": round(np.max(dollar_sizes), 2) if dollar_sizes else 0,
        "min_position_dollar": round(np.min(dollar_sizes), 2) if dollar_sizes else 0,
        "spread": [y_list[i] - filtered_list[i] for i in range(n)]
    }