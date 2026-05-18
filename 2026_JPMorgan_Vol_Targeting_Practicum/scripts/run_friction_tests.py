import sys
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Ensure src can be imported
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.env import Env
from src.backtest import VolTargetEngine

def run_friction_sensitivity():
    results_dir = Env.path("results")
    strategies_dir = Env.path("strategies")
    configs = list(strategies_dir.glob("*.yaml"))
    
    # Cost tiers to test (0 to 15 basis points)
    cost_tiers = [0.0, 2.0, 5.0, 10.0, 15.0]
    
    sensitivity_data = {}

    print("🚀 Starting Transaction Cost Sensitivity Test...")
    
    for config_path in configs:
        with open(config_path, "r", encoding="utf-8") as f:
            base_cfg = yaml.safe_load(f)
            
        strat_name = base_cfg['name']
        print(f"  ▶ Testing {strat_name}...")
        
        sharpes = []
        for cost in cost_tiers:
            # Dynamically override the cost parameter
            test_cfg = base_cfg.copy()
            test_cfg['cost_bps'] = cost
            
            # Run the engine
            engine = VolTargetEngine.from_config(test_cfg)
            res = engine.run(mode="all")
            
            # Calculate Sharpe
            rets = res['returns'].fillna(0)
            ann_ret = rets.mean() * 252
            ann_vol = rets.std() * np.sqrt(252)
            sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
            
            sharpes.append(sharpe)
            
        sensitivity_data[strat_name] = sharpes

    # --- Plotting the Results ---
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, ax = plt.subplots(figsize=(12, 7))
    
    markers = ['o', 's', '^', 'D', 'v', 'p', '*']
    for i, (strat_name, sharpes) in enumerate(sensitivity_data.items()):
        ax.plot(cost_tiers, sharpes, marker=markers[i % len(markers)], linewidth=2, label=strat_name)

    ax.set_title("Robustness Check: Transaction Cost Sensitivity", fontsize=15, fontweight='bold')
    ax.set_xlabel("Transaction Costs (Basis Points)", fontsize=12)
    ax.set_ylabel("Annualized Sharpe Ratio", fontsize=12)
    ax.legend(title="Strategy Models", bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    save_path = results_dir / "6_Friction_Sensitivity.png"
    plt.savefig(save_path, dpi=300)
    print(f"\n✅ Success! Friction test saved to: {save_path}")

if __name__ == "__main__":
    run_friction_sensitivity()