"""
Enhanced Main Script with Deep Kalman Filter Support
Includes all filters including DKF for comparison
"""
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import torch

# Add src directory to path
project_root = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(project_root, 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# Import from lafo package
from lafo import (
    build_sliding_average_operator,
    compute_W,
    lafo_loss,
    lafo_loss_efficient,
    DeepKalmanFilter,
    RecurrentDeepKalmanFilter
)
from lafo.penalized_lafo import (
    lafo_l2_closed_form,
    LAFO_L2_Penalty
)
from lafo.cnn_filter import (
    LAFOCNN,
    CNNFilter
)
from lafo.trading_backtest import mean_reversion_backtest
from lafo.simulation import load_real_sp500

def main():
    print("=== Advanced Signal Filtering for Mean Reversion Trading (SP500 2024-today) ===")
    
    # 1. Load Data
    print("Downloading SP500 data...")
    y = load_real_sp500(start_date="2024-01-01")
    print(f"Loaded {len(y)} days")
    
    # 2. Filtro LAFO-l2 closed-form
    print("\nFiltro LAFO-l2 closed-form...")
    K = 30
    lambda_ = 5.0
    filtered_lafo = lafo_l2_closed_form(y, K, lambda_=lambda_)
    
    # 3. CNN Filter
    print("\nTraining LAFOCNN...")
    model = LAFOCNN(num_channels=48, kernel_size=512)
    model.fit(y, K=K, num_epochs=80, lr=0.008)
    
    y_tensor = torch.from_numpy(y).float().unsqueeze(0).unsqueeze(0)
    filtered_cnn = model.forward(y_tensor).detach().cpu().numpy().squeeze()
    
    # 4. Deep Kalman Filter (NEW - Optional)
    print("\nTraining Deep Kalman Filter...")
    try:
        dkf = DeepKalmanFilter(
            state_dim=4,
            input_dim=1,
            output_dim=1,
            hidden_dim=16,
            transition_type='neural',
            num_layers=1
        )
        
        # Quick training
        dkf.fit(y, K=K, num_epochs=50, lr=0.001)
        
        # Filter
        y_tensor = torch.from_numpy(y).float().unsqueeze(0).unsqueeze(0)
        filtered_dkf, states = dkf.forward(y_tensor)
        filtered_dkf = filtered_dkf.detach().cpu().numpy().squeeze()
        
        print("✅ Deep Kalman Filter completed")
        use_dkf = True
        res_dkf = mean_reversion_backtest(
            y, filtered_dkf,
            entry_threshold=3.0,
            tp_percent=3.5,
            sl_percent=1.0,
            risk_per_trade=0.002,
            leverage=500.0
        )
        
        # Recurrent DKF (optional)
        print("\nTraining Recurrent Deep Kalman Filter...")
        rdkf = RecurrentDeepKalmanFilter(
            state_dim=4,
            input_dim=1,
            output_dim=1,
            hidden_dim=16,
            n_layers=2
        )
        rdkf.fit(y, K=K, num_epochs=50, lr=0.001)
        
        y_tensor = torch.from_numpy(y).float().unsqueeze(0).unsqueeze(0)
        filtered_rdkf, states = rdkf.forward(y_tensor)
        filtered_rdkf = filtered_rdkf.detach().cpu().numpy().squeeze()
        
        print("✅ Recurrent Deep Kalman Filter completed")
        use_rdkf = True
        res_rdkf = mean_reversion_backtest(
            y, filtered_rdkf,
            entry_threshold=3.0,
            tp_percent=3.5,
            sl_percent=1.0,
            risk_per_trade=0.002,
            leverage=500.0
        )
        
    except Exception as e:
        print(f"⚠️  DKF training failed: {e}")
        print("Using CNN and LAFO filters only")
        use_dkf = False
        use_rdkf = False
        res_dkf = None
        res_rdkf = None
    
    # 5. Backtest with Standard Filters
    print("\nBacktest with realistic position sizing (1% risk)...")
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
    
    # === FINAL RESULTS ===
    print("\n" + "=" * 70)
    print("=== FINAL RESULTS ===")
    print("=" * 70)
    
    print("\n=== STANDARD FILTERS ===")
    print(f"LAFO -> Sharpe: {res_lafo['sharpe_ratio']:.3f} | Return: {res_lafo['total_return_pct']:.1f}% | "
          f"Max Drawdown: {res_lafo['max_drawdown_pct']}%")
    print(f"Trades: {res_lafo['num_trades']} (Long: {res_lafo['num_long']} | Short: {res_lafo['num_short']})")
    print(f"Turnover: {res_lafo['turnover']:.3f}")
    
    print(f"\nCNN -> Sharpe: {res_cnn['sharpe_ratio']:.3f} | Return: {res_cnn['total_return_pct']:.1f}%")
    
    if use_dkf:
        print("\n=== DEEP KALMAN FILTER (NEURAL) ===")
        print(f"DKF -> Sharpe: {res_dkf['sharpe_ratio']:.3f} | Return: {res_dkf['total_return_pct']:.1f}% | "
              f"Max Drawdown: {res_dkf['max_drawdown_pct']}%")
        print(f"Trades: {res_dkf['num_trades']} (Long: {res_dkf['num_long']} | Short: {res_dkf['num_short']})")
        
        if use_rdkf:
            print("\n=== RECURRENT DEEP KALMAN ===")
            print(f"RDKF -> Sharpe: {res_rdkf['sharpe_ratio']:.3f} | Return: {res_rdkf['total_return_pct']:.1f}%")
    
    # Plot
    plt.figure(figsize=(18, 12))
    
    plt.subplot(4, 1, 1)
    plt.plot(y, label='SP500 Close', alpha=0.7)
    if use_dkf:
        plt.plot(filtered_dkf, label='DKF Filter', linestyle=':', alpha=0.7)
    plt.plot(filtered_lafo, label='LAFO Filter', linestyle='--')
    plt.plot(filtered_cnn, label='CNN Filter', linestyle='-.')
    if use_rdkf:
        plt.plot(filtered_rdkf, label='RDKF Filter', linestyle='-.', alpha=0.7)
    plt.title('SP500 + All Filters')
    plt.legend()
    
    plt.subplot(4, 1, 2)
    if use_dkf:
        plt.plot(res_dkf['spread'], label='DKF Spread', linestyle=':')
    plt.plot(res_lafo['spread'], label='LAFO Spread')
    plt.axhline(0, color='black', linestyle='--', alpha=0.5)
    plt.title('Spread (Price - Filter)')
    plt.legend()
    
    plt.subplot(4, 1, 3)
    plt.plot(res_lafo['equity_curve'], label='Equity LAFO', alpha=0.7)
    if use_dkf:
        plt.plot(res_dkf['equity_curve'], label='Equity DKF', alpha=0.7)
    plt.plot(res_cnn['equity_curve'], label='Equity CNN', alpha=0.7)
    plt.title('Equity Curve - Demo Account $3000')
    plt.legend()
    
    if use_rdkf:
        plt.subplot(4, 1, 4)
        plt.plot(res_rdkf['equity_curve'], label='Equity RDKF')
        plt.title('Equity Curve - Recurrent DKF')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig('filter_comparison.png', dpi=150)
    print("\n📊 Plot saved to: filter_comparison.png")
    plt.show()

if __name__ == "__main__":
    main()
