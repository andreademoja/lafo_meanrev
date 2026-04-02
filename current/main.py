import numpy as np
import matplotlib.pyplot as plt
import torch
import sys
from pathlib import Path

# Aggiungi il percorso della cartella 'src'
sys.path.append(str(Path(__file__).parent.parent / "src"))

# Importa i moduli
from simulation import load_real_sp500
from penalized_lafo import lafo_l2_closed_form
from cnn_filter import LAFOCNN
from trading_backtest import mean_reversion_backtest

def main():
    print("=== Advanced Signal Filtering for Mean Reversion Trading (SP500 2024-oggi) ===\n")

    # 1. Dati
    print("Scaricamento dati SP500...")
    y = load_real_sp500(start_date="2024-01-01")
    print(f"Dati caricati: {len(y)} giorni\n")

    # 2. Filtro LAFO-ℓ₂
    print("Filtro LAFO-ℓ₂ closed-form...")
    K = 30
    lambda_ = 5.0
    filtered_lafo = lafo_l2_closed_form(y, K, lambda_=lambda_)

    # 3. Filtro CNN
    print("Addestramento LAFOCNN...")
    model = LAFOCNN(num_channels=48, kernel_size=512)
    model.fit(y, K=K, num_epochs=80, lr=0.008)

    y_tensor = torch.from_numpy(y).float().unsqueeze(0).unsqueeze(0)
    filtered_cnn = model.forward(y_tensor).detach().cpu().numpy().squeeze()

    # 4. Backtest con position sizing realistico + leva 500x
    print("\nBacktest con position sizing realistico (1% rischio, leva 500x)...")
    res_lafo = mean_reversion_backtest(
        y, filtered_lafo,
        entry_threshold=3.0,
        tp_percent=3.5,
        sl_percent=1.0,
        risk_per_trade=0.002,
        leverage=500.0
    )
    
    res_cnn = mean_reversion_backtest(
        y, filtered_cnn,
        entry_threshold=3.0,
        tp_percent=3.5,
        sl_percent=1.0,
        risk_per_trade=0.01,
        leverage=500.0
    )

       # === RISULTATI FINALI ===
    print("\n=== RISULTATI FINALI (LAFO-ℓ₂) ===")
    print(f"LAFO → Sharpe: {res_lafo['sharpe_ratio']:.3f} | Rendimento: {res_lafo['total_return_pct']:.1f}% | "
          f"Max Drawdown: {res_lafo['max_drawdown_pct']}%")
    print(f"Trades: {res_lafo['num_trades']} (Long: {res_lafo['num_long']} | Short: {res_lafo['num_short']})")
    print(f"Turnover: {res_lafo['turnover']:.3f}")
    
    # NUOVE STATISTICHE SULLE DIMENSIONI DELLE POSIZIONI
    print(f"\nDimensioni posizioni in dollari:")
    print(f"  Media: ${res_lafo['avg_position_dollar']:,}")
    print(f"  Massima: ${res_lafo['max_position_dollar']:,}")
    print(f"  Minima: ${res_lafo['min_position_dollar']:,}")

    print(f"\nCNN → Sharpe: {res_cnn['sharpe_ratio']:.3f} | Rendimento: {res_cnn['total_return_pct']:.1f}%")

    # Grafico
    plt.figure(figsize=(15, 10))

    plt.subplot(3, 1, 1)
    plt.plot(y, label='SP500 Close', alpha=0.7)
    plt.plot(filtered_lafo, label='LAFO Filter', linestyle='--')
    plt.plot(filtered_cnn, label='CNN Filter', linestyle='-.')
    plt.title('SP500 + Filtri (LAFO-ℓ₂)')
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.plot(res_lafo['spread'], label='LAFO Spread')
    plt.axhline(0, color='black', linestyle='--', alpha=0.5)
    plt.title('Spread (Prezzo - Filtro)')
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.plot(res_lafo['equity_curve'], label='Equity LAFO')
    plt.plot(res_cnn['equity_curve'], label='Equity CNN')
    plt.title('Equity Curve - Conto Demo 3000$')
    plt.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()