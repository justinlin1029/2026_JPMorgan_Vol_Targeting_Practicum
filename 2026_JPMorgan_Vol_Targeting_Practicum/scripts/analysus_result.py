import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import re
from hmmlearn.hmm import GaussianHMM
# Ensure src can be imported
root_dir = Path(__file__).resolve().parents[1]
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from src.env import Env
from src.evaluation.precision_metrics import evaluate_vol_forecast


# ============================================================
# Helpers
# ============================================================
def load_intraday_realized_vol(
    parquet_name: str = "SP500_Intraday_RealizedVol.parquet",
    vol_col: str = "realized_vol",
    min_coverage_ratio: float | None = 0.95,
) -> pd.Series:
    rv_path = Env.path("processed") / parquet_name
    if not rv_path.exists():
        raise FileNotFoundError(f"Missing intraday realized vol parquet: {rv_path}")

    rv_df = pd.read_parquet(rv_path)
    rv_df.index = pd.to_datetime(rv_df.index)
    rv_df = rv_df.sort_index()

    if vol_col not in rv_df.columns:
        raise ValueError(f"Missing '{vol_col}' in {rv_path}. Found columns: {list(rv_df.columns)}")

    rv_df[vol_col] = pd.to_numeric(rv_df[vol_col], errors="coerce")

    if "coverage" in rv_df.columns:
        rv_df["coverage"] = pd.to_numeric(rv_df["coverage"], errors="coerce")

    if min_coverage_ratio is not None and "coverage" in rv_df.columns:
        rv_df = rv_df[rv_df["coverage"] >= float(min_coverage_ratio)].copy()

    rv = rv_df[vol_col].astype(float).copy()
    rv.name = vol_col
    return rv

def pick_regime_series(df: pd.DataFrame) -> pd.Series | None:
    for col in ["regime", "regime_label", "state", "regime_state"]:
        if col in df.columns:
            s = df[col].copy()
            # normalize numeric regimes to strings if needed
            if pd.api.types.is_numeric_dtype(s):
                # keep numeric but cast to int where possible
                s = s.round().astype("Int64").astype(str)
            else:
                s = s.astype(str)
            return s
    return None

def get_vol_regime_labels(df: pd.DataFrame, index: pd.Index | None = None, col: str = "vol_regime") -> pd.Series | None:
    if col not in df.columns:
        return None

    def norm(x):
        if pd.isna(x):
            return np.nan
        s = str(x).strip().lower()
        if s == "low":
            return "Low"
        if s in ("mid", "middle", "med", "medium"):
            return "Mid"
        if s == "high":
            return "High"
        return np.nan

    reg = df[col].map(norm)
    reg = pd.Series(reg, index=df.index)
    if index is not None:
        reg = reg.reindex(index)
    return reg

def derive_regime_from_realized_vol(rv: pd.Series) -> pd.Series:
    m = rv.dropna()
    q1, q2 = m.quantile([0.33, 0.66]).values
    out = pd.Series(index=rv.index, dtype=object)
    out[rv <= q1] = "Low"
    out[(rv > q1) & (rv <= q2)] = "Mid"
    out[rv > q2] = "High"
    return out


def align_forecast_and_realized(
    df: pd.DataFrame,
    window: int = 21,
    ann_factor: float = 252.0,
    horizon: int = 0,
):
    if "vol_estimate" not in df.columns:
        return None

    rv = realized_vol_proxy(
        result_index=df.index,
        parquet_name="SP500_Intraday_RealizedVol.parquet",
        vol_col="realized_vol",
        min_coverage_ratio=0.95,
    )
    vh = pd.Series(df["vol_estimate"], index=df.index).astype(float)

    target = rv.shift(-horizon) if horizon and horizon > 0 else rv
    m = pd.concat(
        [
            vh.rename("vh"),
            rv.rename("rv"),
            target.rename("target_rv"),
        ],
        axis=1,
    ).dropna()

    if m.empty:
        return None

    m["err"] = m["vh"] - m["target_rv"]
    m["abs_err"] = m["err"].abs()
    m["loss"] = qlike_series(m["target_rv"], m["vh"])
    m["vol_jump"] = m["rv"] - m["rv"].shift(1)
    m["vol_of_vol"] = m["vol_jump"].abs()
    return m


def corr_safe(a: pd.Series, b: pd.Series) -> float:
    m = pd.concat([a, b], axis=1).dropna()
    if len(m) < 10:
        return float("nan")
    return float(m.iloc[:, 0].corr(m.iloc[:, 1]))

def estimator_key_from_name(strategy_name: str) -> str:
    return strategy_name.split("_")[0]

def reduce_to_one_per_estimator(data_map, prefer=("naive", "buy_and_hold")):
    grouped = {}
    for strat_name, df in data_map.items():
        est = estimator_key_from_name(strat_name)
        grouped.setdefault(est, []).append((strat_name, df))

    reduced = {}
    for est, items in grouped.items():
        picked = None
        for tag in prefer:
            for name, df in items:
                if tag in name.lower():
                    picked = (name, df)
                    break
            if picked:
                break
        if picked is None:
            picked = items[0]
        reduced[est] = picked[1]  
    return reduced

def get_data():
    results_dir = Env.path("results")
    files = list(results_dir.glob("*.parquet"))
    data_map = {}
    for f in files:
        try:
            data_map[f.stem] = pd.read_parquet(f)
        except Exception as e:
            print(f"⚠️ Failed to read {f.name}: {e}")
    return data_map, results_dir


def pick_strategy_returns(df: pd.DataFrame) -> pd.Series:
    """Return strategy PnL series (used for Sharpe etc.)."""
    if "strategy_returns" in df.columns:
        return df["strategy_returns"]
    if "returns" in df.columns:
        return df["returns"]
    return pd.Series(index=df.index, dtype=float)


def pick_underlying_returns(df: pd.DataFrame) -> tuple[pd.Series | None, str]:
    
    if "asset_returns" in df.columns:
        return df["asset_returns"], "asset_returns"
    return None, "missing"


def safe_mean(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    return float(np.mean(x)) if x.size > 0 else float("nan")


def safe_percentile(x: pd.Series, q: float) -> float:
    arr = x.to_numpy()
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.percentile(arr, q))

# def realized_vol_proxy(under: pd.Series, window=21, ann_factor=252.0) -> pd.Series:
    
#     r = pd.Series(under).astype(float)

#     # Sum of future squared returns
#     fwd_r2 = (r.shift(-1) ** 2).rolling(window, min_periods=window).sum().shift(-(window-1))

#     rv = np.sqrt((ann_factor / window) * fwd_r2)

#     return rv.rename("rv_expost")
def realized_vol_proxy(
    result_index: pd.Index,
    parquet_name: str = "SP500_Intraday_RealizedVol.parquet",
    vol_col: str = "realized_vol",
    min_coverage_ratio: float | None = None,
) -> pd.Series:

    rv_series = load_intraday_realized_vol(
        parquet_name=parquet_name,
        vol_col=vol_col,
        min_coverage_ratio=min_coverage_ratio,
    )

    
    rv_series.index = (
        pd.to_datetime(rv_series.index)
        .tz_localize(None)
        .normalize()
    )

    result_dates = (
        pd.to_datetime(result_index)
        .tz_localize(None)
        .normalize()
    )

    aligned_vals = rv_series.reindex(result_dates)

    return pd.Series(aligned_vals.values, index=result_index, name="rv_expost")

def qlike_series(rv: pd.Series, vol_hat: pd.Series) -> pd.Series:
    r = (rv ** 2).astype(float)
    h = (vol_hat ** 2).astype(float)
    m = pd.concat([r, h], axis=1).dropna()
    r2 = m.iloc[:, 0].clip(lower=1e-18)
    h2 = m.iloc[:, 1].clip(lower=1e-18)
    return np.log(h2) + (r2 / h2)


# ============================================================
# Plots
# ============================================================
def plot_regime_strategy_metrics_table(
    data_map,
    output_path,
    regime_col="vol_regime",
    min_days_per_regime=60,
):
    def norm_reg(x):
        if pd.isna(x):
            return None
        s = str(x).strip().lower()
        if s in ("low",):
            return "Low"
        if s in ("mid", "middle", "med", "medium"):
            return "Mid"
        if s in ("high",):
            return "High"
        return None

    def max_drawdown_from_returns(rets: pd.Series) -> float:
        r = pd.Series(rets).fillna(0.0).astype(float)
        eq = np.exp(np.cumsum(r.values))
        eq = pd.Series(eq, index=r.index)
        dd = eq / eq.cummax() - 1.0
        return float(dd.min()) if len(dd) else float("nan")

    def regime_blocks_mask(reg: pd.Series, label: str) -> list[pd.Index]:
        is_in = reg.astype(str).str.lower().eq(label.lower()).fillna(False).values
        idx = reg.index
        blocks = []
        start = None
        for i, flag in enumerate(is_in):
            if flag and start is None:
                start = i
            if (not flag) and start is not None:
                blocks.append(idx[start:i])
                start = None
        if start is not None:
            blocks.append(idx[start:])
        return blocks

    rows = []
    for name, df in data_map.items():
        if regime_col not in df.columns:
            continue

        strat = pick_strategy_returns(df).astype(float)
        if strat.dropna().empty:
            continue

        reg = df[regime_col].map(norm_reg)
        reg = pd.Series(reg, index=df.index)

        turn = None
        if "weight" in df.columns:
            turn = pd.Series(df["weight"], index=df.index).diff().abs()

        for rlabel in ["Low", "Mid", "High"]:
            idx_r = df.index[reg.eq(rlabel).fillna(False)]
            if len(idx_r) < min_days_per_regime:
                continue

            r_rets = strat.reindex(idx_r).dropna()
            if len(r_rets) < 20:
                continue

            ann_ret = safe_mean(r_rets.values) * 252
            ann_vol = float(np.nanstd(r_rets.values, ddof=0)) * np.sqrt(252)
            sharpe = ann_ret / ann_vol if np.isfinite(ann_vol) and ann_vol > 1e-12 else float("nan")

            # MaxDD computed within each contiguous regime segment; take the worst segment
            blocks = regime_blocks_mask(reg, rlabel)
            block_mdds = []
            for bidx in blocks:
                b_rets = strat.reindex(bidx).dropna()
                if len(b_rets) >= 10:
                    block_mdds.append(max_drawdown_from_returns(b_rets))
            maxdd = float(np.nanmin(block_mdds)) if len(block_mdds) else float("nan")

            if turn is not None:
                r_turn = turn.reindex(idx_r).dropna()
                avg_turn = float(np.nanmean(r_turn.values)) if len(r_turn) else float("nan")
            else:
                avg_turn = float("nan")

            rows.append({
                "strategy": name,
                "regime": rlabel,
                "n_days": int(len(idx_r)),
                "sharpe": sharpe,
                "max_drawdown": maxdd,
                "avg_turnover": avg_turn,
            })

    out = pd.DataFrame(rows)
    if out.empty:
        print("❌ No regime metrics produced (check vol_regime values and min_days_per_regime).")
        return

    out.to_csv(output_path / "Regime_Strategy_Metrics.csv", index=False)

    piv = out.pivot_table(
        index="strategy",
        columns="regime",
        values=["sharpe", "max_drawdown", "avg_turnover", "n_days"],
        aggfunc="first",
    )
    piv.columns = [f"{a}_{b}" for a, b in piv.columns]
    piv = piv.reset_index()

    disp = piv.copy()
    for c in disp.columns:
        if c == "strategy":
            continue
        if c.startswith("n_days"):
            disp[c] = disp[c].map(lambda x: f"{int(x)}" if pd.notna(x) else "0")
        elif "max_drawdown" in c:
            disp[c] = disp[c].map(lambda x: f"{x:.2%}" if pd.notna(x) and np.isfinite(x) else "nan")
        elif "avg_turnover" in c:
            disp[c] = disp[c].map(lambda x: f"{x:.3%}" if pd.notna(x) and np.isfinite(x) else "nan")
        else:
            disp[c] = disp[c].map(lambda x: f"{x:.3f}" if pd.notna(x) and np.isfinite(x) else "nan")

    fig_w = max(12, 0.75 * (disp.shape[1] + 6))
    fig_h = max(3.5, 0.45 * (disp.shape[0] + 4))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")
    ax.set_title("Strategy Metrics by vol_regime (Sharpe / MaxDD / Turnover)", fontsize=14, pad=12)

    table = ax.table(
        cellText=disp.values,
        colLabels=disp.columns.tolist(),
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.4)

    fig.tight_layout()
    fig.savefig(output_path / "Regime_Strategy_Metrics.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"✅ Saved: {output_path / 'Regime_Strategy_Metrics.csv'}")
    print(f"✅ Saved: {output_path / 'Regime_Strategy_Metrics.png'}")
def plot_dim1_returns(data_map, output_path):
    n_strat = len(data_map)
    fig_height = max(8, 4 + 0.6 * n_strat)

    fig, (ax, ax_tbl) = plt.subplots(
        2, 1,
        figsize=(15, fig_height),
        gridspec_kw={"height_ratios": [3, 1]},
        constrained_layout=True
    )
    metrics_data = []

    for name, df in data_map.items():
        rets = pick_strategy_returns(df).fillna(0)
        equity = df.get("equity_curve", None)

        if equity is None:
            # If no equity curve, build from returns
            equity = 1000.0 * np.exp(np.cumsum(rets.fillna(0).values))
            equity = pd.Series(equity, index=df.index)

        ann_ret = safe_mean(rets.values) * 252
        ann_vol = float(np.nanstd(rets.values, ddof=0)) * np.sqrt(252)
        sharpe = ann_ret / ann_vol if np.isfinite(ann_vol) and ann_vol > 0 else 0.0

        downside = rets[rets < 0].values
        downside_std = float(np.nanstd(downside, ddof=0)) * np.sqrt(252) if downside.size > 0 else float("nan")
        sortino = ann_ret / downside_std if np.isfinite(downside_std) and downside_std > 0 else 0.0

        dd = (equity / equity.cummax() - 1)
        max_dd = float(np.nanmin(dd.values)) if len(dd) else float("nan")
        calmar = ann_ret / abs(max_dd) if np.isfinite(max_dd) and max_dd != 0 else 0.0

        ax.plot(df.index, equity, label=f"{name}")
        metrics_data.append(
            [name, f"{sharpe:.2f}", f"{sortino:.2f}", f"{calmar:.2f}", f"{(equity.iloc[-1] / 1000 - 1):.2%}"]
        )

    ax.set_title("Dimension 1: Risk-Adjusted Returns (Sharpe / Sortino / Calmar)", fontsize=16)
    ax.legend()

    ax_tbl.axis("off")
    table = ax_tbl.table(
        cellText=metrics_data,
        colLabels=["Strategy Name", "Sharpe Ratio", "Sortino Ratio", "Calmar Ratio", "Total Return"],
        loc="center",
        cellLoc="center",
    )
    table.scale(1, 2)
    plt.savefig(output_path / "1_Risk_Adjusted_Returns.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_dim2_vol(data_map, output_path):
    n_strat = len(data_map)
    fig_height = max(8, 4 + 0.6 * n_strat)

    fig, (ax, ax_tbl) = plt.subplots(
        2, 1,
        figsize=(15, fig_height),
        gridspec_kw={"height_ratios": [3, 1]},
        constrained_layout=True
    )
    metrics_data = []

    for name, df in data_map.items():
        # realized vol proxy should be based on underlying returns
        rets = pick_strategy_returns(df)
        rolling_vol = rets.rolling(21).std() * np.sqrt(252)
        realized_mean = float(np.nanmean(rolling_vol.values)) if np.isfinite(rolling_vol.values).any() else float("nan")
        vol_cv = (float(np.nanstd(rolling_vol.values)) / realized_mean) if np.isfinite(realized_mean) and realized_mean > 0 else float("nan")

        if "vol_estimate" in df.columns:
            ax.scatter(df["vol_estimate"], rolling_vol, alpha=0.3, s=10, label=name)

        metrics_data.append([name, f"{realized_mean:.2%}" if np.isfinite(realized_mean) else "nan",
                             f"{vol_cv:.4f}" if np.isfinite(vol_cv) else "nan",
                             f"{(realized_mean - 0.10):.2%}" if np.isfinite(realized_mean) else "nan"])

    ax.plot([0.05, 0.25], [0.05, 0.25], "k--", alpha=0.5, label="Ideal Control")
    ax.set_title("Dimension 2: Volatility Control Ability (Vol CV)", fontsize=16)
    ax.set_xlabel("Estimated Volatility")
    ax.set_ylabel("Realized Volatility")
    ax.legend()

    ax_tbl.axis("off")
    table = ax_tbl.table(
        cellText=metrics_data,
        colLabels=["Strategy Name", "Avg Realized Vol", "Vol CV", "Target Deviation"],
        loc="center",
        cellLoc="center",
    )
    table.scale(1, 2)
    plt.savefig(output_path / "2_Vol_Control.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_dim3_risk(data_map, output_path):
    n_strat = len(data_map)
    fig_height = max(8, 4 + 0.6 * n_strat)

    fig, (ax, ax_tbl) = plt.subplots(
        2, 1,
        figsize=(15, fig_height),
        gridspec_kw={"height_ratios": [3, 1]},
        constrained_layout=True
    )
    metrics_data = []

    for name, df in data_map.items():
        equity = df.get("equity_curve", None)
        if equity is None:
            rets = pick_strategy_returns(df).fillna(0)
            equity = 1000.0 * np.exp(np.cumsum(rets.values))
            equity = pd.Series(equity, index=df.index)

        dd = (equity / equity.cummax() - 1)

        # VaR based on STRATEGY returns (extreme loss of strategy)
        strat_rets = pick_strategy_returns(df).fillna(0)
        var_95 = safe_percentile(strat_rets, 5)

        # Ulcer index
        ui = np.sqrt(safe_mean(np.square(dd.values)))

        stress_days = int((dd < -0.05).sum())

        ax.fill_between(df.index, dd, 0, alpha=0.3, label=name)

        metrics_data.append([
            name,
            f"{float(np.nanmin(dd.values)):.2%}" if np.isfinite(dd.values).any() else "nan",
            f"{ui:.4f}" if np.isfinite(ui) else "nan",
            f"{var_95:.2%}" if np.isfinite(var_95) else "nan",
            f"{stress_days} Days"
        ])

    ax.set_title("Dimension 3: Extreme Risk Metrics (MaxDD / Ulcer / VaR)", fontsize=16)
    ax.set_ylabel("Drawdown")
    ax.legend()

    ax_tbl.axis("off")
    table = ax_tbl.table(
        cellText=metrics_data,
        colLabels=["Strategy Name", "Max Drawdown", "Ulcer Index", "VaR (95%)", "Days with DD > 5%"],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    plt.savefig(output_path / "3_Extreme_Risk.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_dim4_costs(data_map, output_path):
    n_strat = len(data_map)

    fig_height = max(8, 4 + 0.6 * n_strat)

    fig, (ax, ax_tbl) = plt.subplots(
        2, 1,
        figsize=(15, fig_height),
        gridspec_kw={"height_ratios": [3, 1]},
        constrained_layout=True
    )

    metrics_data = []
    names = []
    turnovers = []

    for name, df in data_map.items():
        if "weight" not in df.columns:
            continue

        turnover = float(np.nanmean(df["weight"].diff().abs().values))
        strat_rets = pick_strategy_returns(df)

        pos_rets = strat_rets[strat_rets > 0]
        neg_rets = strat_rets[strat_rets < 0]

        if len(pos_rets) > 0 and len(neg_rets) > 0:
            win_loss = float(pos_rets.mean() / abs(neg_rets.mean()))
        else:
            win_loss = float("nan")

        win_rate = float(len(pos_rets) / len(strat_rets)) if len(strat_rets) > 0 else float("nan")

        names.append(name)
        turnovers.append(turnover)

        metrics_data.append([
            name,
            f"{turnover:.2%}" if np.isfinite(turnover) else "nan",
            f"{win_loss:.2f}" if np.isfinite(win_loss) else "nan",
            f"{win_rate:.2%}" if np.isfinite(win_rate) else "nan"
        ])

    # Bar plot
    x = np.arange(len(names))
    ax.bar(x, turnovers, alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_title("Dimension 4: Trading Efficiency (Turnover / Win-Loss Ratio)", fontsize=16)
    ax.set_ylabel("Avg Daily Turnover")

    # Add small margin at top
    ax.margins(y=0.15)

 
    ax_tbl.axis("off")
    table = ax_tbl.table(
        cellText=metrics_data,
        colLabels=["Strategy Name", "Turnover", "Win-Loss Ratio", "Win Rate"],
        loc="center",
        cellLoc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)

    fig.savefig(output_path / "4_Trading_Costs.png", dpi=300, bbox_inches="tight")
    plt.close(fig)




def plot_precision_table(data_map, output_path, ann_factor=252.0):
    rows = []
    warnings = []

    for name, df in data_map.items():
        if "vol_estimate" not in df.columns:
            warnings.append(f"{name}: missing vol_estimate (skip precision metrics)")
            continue

        under, src = pick_underlying_returns(df)
        if under is None:
            warnings.append(f"{name}: missing underlying returns series (skip)")
            continue

        # If we had to fallback to "returns", warn because it might be strategy PnL in old files
        if src == "returns" and "asset_returns" not in df.columns and "returns_clean" not in df.columns:
            warnings.append(f"{name}: using df['returns'] fallback (verify this is UNDERLYING returns)")

        # evaluate_vol_forecast expects finite pairs
        m = evaluate_vol_forecast(
            returns=under,
            vol_hat=df["vol_estimate"],
            ann_factor=ann_factor,
        )

        # Guard: if metrics are empty/invalid, skip instead of crashing later
        if m is None or (isinstance(m, dict) and m.get("n", 0) == 0):
            warnings.append(f"{name}: precision metrics empty (n=0), skipped")
            continue

        m["strategy"] = name
        rows.append(m)

    out = pd.DataFrame(rows)
    if out.empty:
        print("❌ No valid strategies for precision table.")
        if warnings:
            print("\n⚠️ Warnings:")
            for w in warnings:
                print(" -", w)
        return

    out = out.set_index("strategy")

    cols = [
        "n",
        "qlike",
        "mse_var",
        "mae_vol",
        "oos_r2_var_vs_const",
        "mz_alpha",
        "mz_beta",
        "mz_r2",
    ]
    cols = [c for c in cols if c in out.columns]

    # sort (qlike lower better) if exists
    if "qlike" in out.columns:
        out = out[cols].sort_values("qlike", ascending=True)
    else:
        out = out[cols]

    # save CSV
    out.to_csv(output_path / "precision_metrics.csv")

    # display formatting
    disp = out.copy()
    if "n" in disp.columns:
        disp["n"] = disp["n"].fillna(0).astype(int)

    for c in disp.columns:
        if c != "n":
            disp[c] = disp[c].map(lambda x: f"{x:.4f}" if pd.notna(x) and np.isfinite(x) else "nan")

    nrows, ncols = disp.shape
    fig_w = max(10, 1.2 * (ncols + 3))
    fig_h = max(2.5, 0.6 * (nrows + 3))

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")
    ax.set_title("Volatility Estimation Precision Metrics (Forecast vs Realized Proxy)", fontsize=14, pad=12)

    table = ax.table(
        cellText=disp.values,
        rowLabels=disp.index.tolist(),
        colLabels=disp.columns.tolist(),
        loc="center",
        cellLoc="center",
        rowLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.6)

    fig.tight_layout()
    fig.savefig(output_path / "table.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"✅ Saved precision table CSV: {output_path / 'precision_metrics.csv'}")
    print(f"✅ Saved precision table PNG : {output_path / 'table.png'}")

    if warnings:
        print("\n⚠️ Warnings:")
        for w in warnings:
            print(" -", w)


def plot_precision_timeseries(data_map, output_path, window=21, ann_factor=252.0, top_k=5):
    rows = []
    series = {}

    for name, df in data_map.items():
        if "vol_estimate" not in df.columns:
            continue
        rv = realized_vol_proxy(
            result_index=df.index,
            parquet_name="SP500_Intraday_RealizedVol.parquet",
            vol_col="realized_vol",
            min_coverage_ratio=0.95,
        )
        vh = pd.Series(df["vol_estimate"], index=df.index)
        loss = qlike_series(rv, vh)
        score = float(np.nanmean(loss.values)) if len(loss) else np.nan
        rows.append((name, score))
        series[name] = (rv, vh)

    if not rows:
        return

    rows = sorted(rows, key=lambda x: x[1])
    keep = [n for n, _ in rows[:top_k]]

    fig, ax = plt.subplots(figsize=(14, 6))
    # plot realized once (common axis)
    rv0 = series[keep[0]][0]
    ax.plot(rv0.index, rv0.values, linewidth=2, label=f"Realized vol proxy")

    for name in keep:
        rv, vh = series[name]
        ax.plot(vh.index, vh.values, alpha=0.8, label=name)

    ax.set_title("Forecast Vol vs Realized Vol Proxy")
    ax.set_ylabel("Annualized Vol")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path / "P_TS_Forecast_vs_Realized.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

def plot_precision_by_regime(data_map, output_path, window=21, ann_factor=252.0, regime_col="vol_regime"):
    rows = []
    regimes = ["Low", "Mid", "High"]

    for name, df in data_map.items():
        if "vol_estimate" not in df.columns:
            continue
        rv = realized_vol_proxy(
            result_index=df.index,
            parquet_name="SP500_Intraday_RealizedVol.parquet",
            vol_col="realized_vol",
            min_coverage_ratio=0.95,
        )
        vh = pd.Series(df["vol_estimate"], index=df.index)

        loss = qlike_series(rv, vh)
        m = pd.concat([rv.rename("rv"), loss.rename("loss")], axis=1).dropna()
        if len(m) < 100:
            continue

        reg = get_vol_regime_labels(df, index=m.index, col=regime_col)
        if reg is None:
            continue

        m = m.copy()
        m["regime"] = reg
        m = m.dropna(subset=["regime"])

        low = m.loc[m["regime"].eq("Low"), "loss"].mean()
        mid = m.loc[m["regime"].eq("Mid"), "loss"].mean()
        high = m.loc[m["regime"].eq("High"), "loss"].mean()

        rows.append([name, low, mid, high])

    if not rows:
        return

    out = pd.DataFrame(rows, columns=["strategy"] + regimes).set_index("strategy")
    out = out.sort_values("High")  # focus on high-vol regime

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(out.index))
    width = 0.25
    ax.bar(x - width, out["Low"].values, width, label="Low")
    ax.bar(x,         out["Mid"].values, width, label="Mid")
    ax.bar(x + width, out["High"].values, width, label="High")
    ax.set_xticks(x)
    ax.set_xticklabels(out.index, rotation=45, ha="right")
    ax.set_title("Regime Precision: Average QLIKE Loss by vol_regime")
    ax.set_ylabel("Avg QLIKE loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path / "P_Regime_QLIKE.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

def plot_regime_explainer_table(
    data_map,
    output_path,
    window=21,
    ann_factor=252.0,
    horizon=0,
    min_n=80,
    regime_col="vol_regime",
):
    rows = []
    for name, df in data_map.items():
        m = align_forecast_and_realized(df, 1, ann_factor=ann_factor, horizon=horizon)
        if m is None or len(m) < min_n:
            continue

        reg = get_vol_regime_labels(df, index=m.index, col=regime_col)
        if reg is None:
            continue

        m = m.copy()
        m["regime"] = reg
        m = m.dropna(subset=["regime"])

        for r in ["Low", "Mid", "High"]:
            mr = m[m["regime"].eq(r)]
            if len(mr) < 20:
                continue
            rows.append({
                "strategy": name,
                "regime": r,
                "n": int(len(mr)),
                "mean_abs_err": float(mr["abs_err"].mean()),
                "bias_err": float(mr["err"].mean()),
                "corr_vh_vs_target": corr_safe(mr["vh"], mr["target_rv"]),
                "mean_qlike": float(mr["loss"].mean()),
                "mean_vol_of_vol": float(mr["vol_of_vol"].mean()),
            })

    out = pd.DataFrame(rows)
    if out.empty:
        print("❌ Regime explainer table empty (missing vol_regime or insufficient data).")
        return

    out = out.sort_values(["strategy", "regime"])
    out.to_csv(output_path / "Regime_Explainer_Table.csv", index=False)

    piv = out.pivot_table(
        index="strategy",
        columns="regime",
        values=["mean_abs_err", "bias_err", "corr_vh_vs_target", "mean_qlike", "mean_vol_of_vol"],
        aggfunc="first",
    )
    piv.columns = [f"{a}_{b}" for a, b in piv.columns]
    piv = piv.reset_index()

    disp = piv.copy()
    for c in disp.columns:
        if c == "strategy":
            continue
        disp[c] = disp[c].map(lambda x: f"{x:.4f}" if pd.notna(x) and np.isfinite(x) else "nan")

    fig_w = max(12, 0.8 * (disp.shape[1] + 4))
    fig_h = max(3, 0.5 * (disp.shape[0] + 3))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")
    ax.set_title(f"Regime Explainer Table by vol_regime (horizon={horizon})", fontsize=14, pad=12)

    table = ax.table(
        cellText=disp.values,
        colLabels=disp.columns.tolist(),
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.5)

    fig.tight_layout()
    fig.savefig(output_path / "Regime_Explainer_Table.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"✅ Saved: {output_path / 'Regime_Explainer_Table.csv'}")
    print(f"✅ Saved: {output_path / 'Regime_Explainer_Table.png'}")

def plot_abs_error_by_regime(
    data_map,
    output_path,
    window=21,
    ann_factor=252.0,
    horizon=0,
    top_k=7,
    min_n=120,
    regime_col="vol_regime",
):
    scores = []
    aligned = {}

    for name, df in data_map.items():
        m = align_forecast_and_realized(df, 1, ann_factor=ann_factor, horizon=horizon)
        if m is None or len(m) < min_n:
            continue
        scores.append((name, float(np.nanmean(m["loss"].values))))
        aligned[name] = (df, m)

    if not scores:
        return

    scores = sorted(scores, key=lambda x: x[1])
    keep = [n for n, _ in scores[:top_k]]

    fig, ax = plt.subplots(figsize=(14, 6))
    positions, labels, data = [], [], []
    pos = 1

    for name in keep:
        df, m = aligned[name]

        reg = get_vol_regime_labels(df, index=m.index, col=regime_col)
        if reg is None:
            continue

        tmp = m.copy()
        tmp["regime"] = reg
        tmp = tmp.dropna(subset=["regime"])

        for r in ["Low", "Mid", "High"]:
            arr = tmp.loc[tmp["regime"].eq(r), "abs_err"].dropna().values
            if arr.size < 20:
                continue
            data.append(arr)
            positions.append(pos)
            labels.append(f"{name}\n{r}")
            pos += 1

        pos += 1

    if len(data) == 0:
        return

    ax.boxplot(data, positions=positions, showfliers=False)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_title(f"Abs Forecast Error by vol_regime (horizon={horizon})")
    ax.set_ylabel("|vol_forecast - vol_realized|")
    ax.grid(True, linestyle="--", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path / "R_AbsError_ByRegime.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

def plot_vol_of_vol_by_regime(
    data_map,
    output_path,
    window=21,
    ann_factor=252.0,
    horizon=0,
    top_k=7,
    min_n=120,
    regime_col="vol_regime",
):
    scores = []
    aligned = {}

    for name, df in data_map.items():
        m = align_forecast_and_realized(df, 1, ann_factor=ann_factor, horizon=horizon)
        if m is None or len(m) < min_n:
            continue
        scores.append((name, float(np.nanmean(m["loss"].values))))
        aligned[name] = (df, m)

    if not scores:
        return

    scores = sorted(scores, key=lambda x: x[1])
    keep = [n for n, _ in scores[:top_k]]

    fig, ax = plt.subplots(figsize=(14, 6))
    positions, labels, data = [], [], []
    pos = 1

    for name in keep:
        df, m = aligned[name]

        reg = get_vol_regime_labels(df, index=m.index, col=regime_col)
        if reg is None:
            continue

        tmp = m.copy()
        tmp["regime"] = reg
        tmp = tmp.dropna(subset=["regime"])

        for r in ["Low", "Mid", "High"]:
            arr = tmp.loc[tmp["regime"].eq(r), "vol_of_vol"].dropna().values
            if arr.size < 20:
                continue
            data.append(arr)
            positions.append(pos)
            labels.append(f"{name}\n{r}")
            pos += 1

        pos += 1

    if len(data) == 0:
        return

    ax.boxplot(data, positions=positions, showfliers=False)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_title("Vol-of-Vol (|Δ realized vol|) by vol_regime")
    ax.set_ylabel("|Δ realized vol|")
    ax.grid(True, linestyle="--", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path / "R_VolOfVol_ByRegime.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_pred_vol_vs_realized_from_parquet(
    parquet_name: str,
    output_path,
    window: int = 21,
    ann_factor: float = 252.0,
    pred_col: str = "vol_estimate",
):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from pathlib import Path
    from src.env import Env

    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    pq_path = Path(parquet_name)
    if not pq_path.exists():
        pq_path = Env.path("results") / parquet_name
    if not pq_path.exists():
        raise FileNotFoundError(f"Missing strategy parquet: {pq_path}")

    df = pd.read_parquet(pq_path)
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    if pred_col not in df.columns:
        raise ValueError(f"Parquet missing '{pred_col}' column: {pq_path}")

    pred = pd.Series(df[pred_col].astype(float), index=df.index).replace([np.inf, -np.inf], np.nan)

    rv = realized_vol_proxy(
        result_index=df.index,
        parquet_name="SP500_Intraday_RealizedVol.parquet",
        vol_col="realized_vol",
        min_coverage_ratio=0.95,
    )

    # ---- align + dropna ----
    m = pd.concat([rv.rename("rv"), pred.rename("pred")], axis=1).dropna()
    if len(m) < 5:
        print(f" Not enough overlapping data after dropna: {len(m)} rows for {pq_path.name}")
        return

    # ---- plot ----
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(m.index, m["rv"].values, linewidth=2, label=f"Realized Vol Proxy (window={window})")
    ax.plot(m.index, m["pred"].values, alpha=0.9, label=f"Predicted Vol ({pq_path.name}:{pred_col})")

    ax.set_title("Predicted Volatility vs Realized Volatility Proxy")
    ax.set_ylabel("Annualized Vol (decimal)")
    ax.legend()
    fig.tight_layout()

    out_file = output_path / f"P_TS_Pred_vs_Realized__{pq_path.stem}.png"
    fig.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ Saved: {out_file}")



























































import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM


def compute_hmm_regime_from_series(
    vol_series: pd.Series,
    n_components: int = 3,
    random_state: int = 42,
) -> tuple[pd.Series, pd.Series, GaussianHMM | None]:
    """
    Use the user's HMM logic on a volatility series and return:
      - regime_state: numeric hidden state
      - regime_label: Low / Mid / High
      - fitted model
    """
    vol = pd.Series(vol_series).astype(float)

    mask = np.isfinite(vol.values) & (vol.values > 0)
    regime_state = pd.Series(np.nan, index=vol.index, dtype="float")
    regime_label = pd.Series(pd.NA, index=vol.index, dtype="object")

    if mask.sum() < 30:
        return regime_state, regime_label, None

    log_vol = np.log(vol.loc[mask]).values.reshape(-1, 1)

    try:
        model = GaussianHMM(
            n_components=n_components,
            covariance_type="diag",
            n_iter=1000,
            random_state=random_state,
            init_params="mc",
            params="stmc",
        )

        model.startprob_ = np.array([0.33, 0.33, 0.34])
        model.transmat_ = np.array([
            [0.96, 0.03, 0.01],
            [0.03, 0.94, 0.03],
            [0.01, 0.03, 0.96],
        ])

        model.fit(log_vol)

        p_diag = np.diag(model.transmat_)
        expected_duration = 1 / (1 - p_diag)
        print("HMM expected duration:", expected_duration)

        states = model.predict(log_vol)

        state_means = model.means_.reshape(-1)
        order = np.argsort(state_means)
        ordered_labels = np.array(["Low", "Mid", "High"])
        state_to_label = {int(order[k]): ordered_labels[k] for k in range(n_components)}

        regime_state.loc[mask] = states
        regime_label.loc[mask] = pd.Series(states, index=vol.loc[mask].index).map(state_to_label).values

        return regime_state, regime_label, model

    except Exception as e:
        print(f"⚠️ HMM regime labeling failed: {e}")
        return regime_state, regime_label, None


def build_realized_vol_benchmark_regime(
    df: pd.DataFrame,
    window: int = 21,
    ann_factor: float = 252.0,
) -> pd.DataFrame | None:
    """
    Build realized-vol benchmark and label it with the user's HMM logic.
    """
    rv = realized_vol_proxy(
        result_index=df.index,
        parquet_name="SP500_Intraday_RealizedVol.parquet",
        vol_col="realized_vol",
        min_coverage_ratio=0.95,
    )
    regime_state, regime_label, model = compute_hmm_regime_from_series(rv)

    out = pd.DataFrame(index=df.index)
    out["benchmark_rv"] = rv.reindex(df.index)
    out["benchmark_regime_state"] = regime_state.reindex(df.index)
    out["benchmark_regime"] = regime_label.reindex(df.index)
    return out


def qlike_series(actual: pd.Series, forecast: pd.Series) -> pd.Series:
    """
    QLIKE loss using volatility inputs.
    Formula applied on variance scale:
        log(h) + rv / h
    where h = forecast variance, rv = realized variance
    """
    a = pd.Series(actual).astype(float)
    f = pd.Series(forecast).astype(float)

    df = pd.concat([a.rename("actual"), f.rename("forecast")], axis=1).dropna()
    if df.empty:
        return pd.Series(index=a.index, dtype=float)

    eps = 1e-12
    actual_var = np.maximum(df["actual"].values ** 2, eps)
    forecast_var = np.maximum(df["forecast"].values ** 2, eps)

    loss = np.log(forecast_var) + actual_var / forecast_var
    out = pd.Series(np.nan, index=a.index, dtype=float)
    out.loc[df.index] = loss
    return out

def annualized_vol_from_returns(ret: pd.Series, ann_factor: float = 252.0) -> float:
    """
    Annualized volatility from simple returns.
    """
    r = pd.Series(ret).astype(float).dropna()
    if len(r) < 2:
        return float("nan")
    sigma = r.std(ddof=1)
    if not np.isfinite(sigma):
        return float("nan")
    return float(sigma * np.sqrt(ann_factor))

def max_drawdown_from_returns(ret: pd.Series) -> float:
    """
    Max drawdown computed on the cumulative wealth path of the supplied return series.
    """
    r = pd.Series(ret).astype(float).dropna()
    if r.empty:
        return float("nan")

    wealth = (1.0 + r).cumprod()
    peak = wealth.cummax()
    dd = wealth / peak - 1.0
    return float(dd.min())


def ulcer_index_from_returns(ret: pd.Series) -> float:
    """
    Ulcer Index on the cumulative wealth path of the supplied return series.
    """
    r = pd.Series(ret).astype(float).dropna()
    if r.empty:
        return float("nan")

    wealth = (1.0 + r).cumprod()
    peak = wealth.cummax()
    dd_pct = 100.0 * (wealth / peak - 1.0)
    return float(np.sqrt(np.mean(dd_pct**2)))


def var_95_from_returns(ret: pd.Series) -> float:
    """
    Historical daily VaR at 95%.
    Returned as a positive loss number.
    """
    r = pd.Series(ret).astype(float).dropna()
    if r.empty:
        return float("nan")
    return float(-np.percentile(r, 5))


def annualized_return_from_returns(ret: pd.Series, ann_factor: float = 252.0) -> float:
    """
    Geometric annualized return from simple returns.
    """
    r = pd.Series(ret).astype(float).dropna()
    if r.empty:
        return float("nan")

    wealth = (1.0 + r).prod()
    n = len(r)
    if n == 0 or wealth <= 0:
        return float("nan")

    return float(wealth ** (ann_factor / n) - 1.0)


def sharpe_ratio_from_returns(ret: pd.Series, ann_factor: float = 252.0) -> float:
    """
    Annualized Sharpe ratio from simple returns, rf assumed zero.
    """
    r = pd.Series(ret).astype(float).dropna()
    if len(r) < 2:
        return float("nan")

    mu = r.mean()
    sigma = r.std(ddof=1)
    if not np.isfinite(sigma) or sigma <= 0:
        return float("nan")

    return float(np.sqrt(ann_factor) * mu / sigma)


def oos_r2_from_forecast(actual_vol: pd.Series, forecast_vol: pd.Series) -> float:
    """
    Out-of-sample R^2 relative to a constant-mean benchmark forecast.
    Computed on volatility level, matching the existing MSE setup.
    """
    df = pd.concat(
        [
            pd.Series(actual_vol).astype(float).rename("actual"),
            pd.Series(forecast_vol).astype(float).rename("forecast"),
        ],
        axis=1,
    ).dropna()

    if len(df) < 2:
        return float("nan")

    sse_model = np.sum((df["forecast"] - df["actual"]) ** 2)
    mean_benchmark = df["actual"].mean()
    sse_bench = np.sum((df["actual"] - mean_benchmark) ** 2)

    if not np.isfinite(sse_bench) or sse_bench <= 0:
        return float("nan")

    return float(1.0 - sse_model / sse_bench)


def annualized_vol_from_returns(ret: pd.Series, ann_factor: float = 252.0) -> float:
    """
    Annualized volatility from simple returns.
    """
    r = pd.Series(ret).astype(float).dropna()
    if len(r) < 2:
        return float("nan")
    sigma = r.std(ddof=1)
    if not np.isfinite(sigma):
        return float("nan")
    return float(sigma * np.sqrt(ann_factor))


def evaluate_one_strategy_by_benchmark_regime(
    df: pd.DataFrame,
    benchmark_regime_df: pd.DataFrame,
    window: int = 21,
    ann_factor: float = 252.0,
    horizon: int = 0,
) -> pd.DataFrame:
    """
    Evaluate one result file by benchmark volatility regime.

    Output is one row per regime:
      - regime
      - n_obs
      - AnnualizedReturn
      - AnnualizedVol
      - OOS_R2
      - Sharpe
      - VaR_95
      - MaxDD
      - UlcerIndex
      - QLIKE
    """
    aligned = align_forecast_and_realized(
        df=df,
        window=window,
        ann_factor=ann_factor,
        horizon=horizon,
    )

    if aligned is None or aligned.empty:
        return pd.DataFrame()

    strat_ret = pick_strategy_returns(df).astype(float).reindex(aligned.index)
    bench_reg = benchmark_regime_df["benchmark_regime"].reindex(aligned.index)

    tmp = aligned.copy()
    tmp["strategy_returns"] = strat_ret
    tmp["benchmark_regime"] = bench_reg
    tmp = tmp.dropna(subset=["benchmark_regime"])

    if tmp.empty:
        return pd.DataFrame()

    tmp["qlike"] = qlike_series(tmp["target_rv"], tmp["vh"]).reindex(tmp.index)

    rows = []
    regime_order = ["Low", "Mid", "High"]

    for reg in regime_order:
        g = tmp[tmp["benchmark_regime"] == reg].copy()
        if g.empty:
            continue

        r = g["strategy_returns"].dropna()
        if r.empty:
            continue

        rows.append({
            "regime": reg,
            "n_obs": int(len(g)),
            "AnnualizedReturn": annualized_return_from_returns(r, ann_factor=ann_factor),
            "AnnualizedVol": annualized_vol_from_returns(r, ann_factor=ann_factor),
            "OOS_R2": oos_r2_from_forecast(g["target_rv"], g["vh"]),
            "Sharpe": sharpe_ratio_from_returns(r, ann_factor=ann_factor),
            "VaR_95": var_95_from_returns(r),
            "MaxDD": max_drawdown_from_returns(r),
            "UlcerIndex": ulcer_index_from_returns(r),
            "QLIKE": float(g["qlike"].mean()) if g["qlike"].notna().any() else float("nan"),
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    out["regime"] = pd.Categorical(out["regime"], categories=regime_order, ordered=True)
    out = out.sort_values("regime").reset_index(drop=True)
    return out

def plot_rf_vs_no_rf_difference_summary(data_map: dict[str, pd.DataFrame], output_path):
    """
    One figure:
      - top: bar chart of annualized return difference
             = ann_return(with_rf) - ann_return(no_rf)
      - bottom: table of metric differences for each strategy

    Required columns in each result parquet:
      - returns_with_rf
      - returns_no_rf
      - weight

    Notes
    -----
    Difference columns are defined as:
      metric_diff = metric_with_rf - metric_no_rf

    So:
      - positive ann_return_diff: RF helped annualized return
      - positive sharpe_diff: RF improved Sharpe
      - positive turnover_diff: with-RF had higher turnover
      - positive maxdd_diff: with-RF max drawdown is less severe only if you
        interpret drawdown numerically with negatives carefully.
        Since MaxDD is negative, I also include abs_maxdd_diff for clarity.
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from pathlib import Path

    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    def ann_return_from_log_returns(r: pd.Series, ann_factor: float = 252.0) -> float:
        r = pd.Series(r).astype(float).dropna()
        if len(r) == 0:
            return float("nan")
        return float(r.mean() * ann_factor)

    def sharpe_from_log_returns(r: pd.Series, ann_factor: float = 252.0) -> float:
        r = pd.Series(r).astype(float).dropna()
        if len(r) < 2:
            return float("nan")
        mu = float(r.mean())
        sigma = float(r.std(ddof=1))
        if not np.isfinite(sigma) or sigma <= 0:
            return float("nan")
        return float(np.sqrt(ann_factor) * mu / sigma)

    def max_drawdown_from_log_returns(r: pd.Series) -> float:
        r = pd.Series(r).astype(float).fillna(0.0)
        if len(r) == 0:
            return float("nan")
        eq = np.exp(np.cumsum(r.values))
        eq = pd.Series(eq, index=r.index)
        dd = eq / eq.cummax() - 1.0
        return float(dd.min())

    rows = []

    for name, df in data_map.items():
        required = ["returns_with_rf", "returns_no_rf"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            print(f"⚠️ Skip {name}: missing columns {missing}")
            continue

        r_with = pd.Series(df["returns_with_rf"], index=df.index).astype(float).fillna(0.0)
        r_no = pd.Series(df["returns_no_rf"], index=df.index).astype(float).fillna(0.0)

        ann_ret_with = ann_return_from_log_returns(r_with)
        ann_ret_no = ann_return_from_log_returns(r_no)

        sharpe_with = sharpe_from_log_returns(r_with)
        sharpe_no = sharpe_from_log_returns(r_no)

        maxdd_with = max_drawdown_from_log_returns(r_with)
        maxdd_no = max_drawdown_from_log_returns(r_no)

        # Same weight path in your current engine, so this difference will usually be 0
        if "weight" in df.columns:
            turnover = pd.Series(df["weight"], index=df.index).astype(float).diff().abs()
            turnover_with = float(np.nanmean(turnover.values))
            turnover_no = float(np.nanmean(turnover.values))
        else:
            turnover_with = float("nan")
            turnover_no = float("nan")

        rows.append({
            "strategy": name,
            "ann_return_with_rf": ann_ret_with,
            "ann_return_no_rf": ann_ret_no,
            "ann_return_diff": ann_ret_with - ann_ret_no,
            "sharpe_with_rf": sharpe_with,
            "sharpe_no_rf": sharpe_no,
            "sharpe_diff": sharpe_with - sharpe_no,
            "turnover_with_rf": turnover_with,
            "turnover_no_rf": turnover_no,
            "turnover_diff": turnover_with - turnover_no,
            "maxdd_with_rf": maxdd_with,
            "maxdd_no_rf": maxdd_no,
            "maxdd_diff": maxdd_with - maxdd_no,
            "abs_maxdd_diff": abs(maxdd_with) - abs(maxdd_no),
        })

    out = pd.DataFrame(rows)
    if out.empty:
        print("❌ No valid strategies found for RF vs No-RF difference plot.")
        return

    out = out.sort_values("ann_return_diff", ascending=False).reset_index(drop=True)

    # Display table
    disp = out[[
        "strategy",
        "ann_return_diff",
        "sharpe_diff",
        "turnover_diff",
        "maxdd_diff",
        "abs_maxdd_diff",
    ]].copy()

    disp["ann_return_diff"] = disp["ann_return_diff"].map(
        lambda x: f"{x:.2%}" if pd.notna(x) and np.isfinite(x) else "nan"
    )
    disp["sharpe_diff"] = disp["sharpe_diff"].map(
        lambda x: f"{x:.3f}" if pd.notna(x) and np.isfinite(x) else "nan"
    )
    disp["turnover_diff"] = disp["turnover_diff"].map(
        lambda x: f"{x:.3%}" if pd.notna(x) and np.isfinite(x) else "nan"
    )
    disp["maxdd_diff"] = disp["maxdd_diff"].map(
        lambda x: f"{x:.2%}" if pd.notna(x) and np.isfinite(x) else "nan"
    )
    disp["abs_maxdd_diff"] = disp["abs_maxdd_diff"].map(
        lambda x: f"{x:.2%}" if pd.notna(x) and np.isfinite(x) else "nan"
    )

    fig_h = max(8, 5 + 0.35 * len(out))
    fig, (ax, ax_tbl) = plt.subplots(
        2, 1,
        figsize=(16, fig_h),
        gridspec_kw={"height_ratios": [3, 1.8]},
        constrained_layout=True,
    )

    # Top chart: annualized return difference
    x = np.arange(len(out))
    vals = out["ann_return_diff"].values
    ax.bar(x, vals)
    ax.axhline(0.0, linestyle="--", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(out["strategy"], rotation=45, ha="right")
    ax.set_ylabel("Annualized Return Difference")
    ax.set_title("With RF vs No RF: Annualized Return Difference by Strategy")
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)

    # Bottom table
    ax_tbl.axis("off")
    table = ax_tbl.table(
        cellText=disp.values,
        colLabels=[
            "Strategy",
            "Δ Annualized Return",
            "Δ Sharpe",
            "Δ Turnover",
            "Δ MaxDD",
            "Δ |MaxDD|",
        ],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.6)

    out_file = output_path / "RF_vs_NoRF_Difference_Summary.png"
    fig.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close(fig)

    csv_file = output_path / "RF_vs_NoRF_Difference_Summary.csv"
    out.to_csv(csv_file, index=False)

    print(f"✅ Saved: {out_file}")
    print(f"✅ Saved: {csv_file}")
def build_regime_metric_table(
    data_map: dict[str, pd.DataFrame],
    benchmark_strategy: str | None = None,
    window: int = 21,
    ann_factor: float = 252.0,
    horizon: int = 0,
    one_per_estimator: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """
    Build a regime-conditioned metric table for all strategies.

    Parameters
    ----------
    benchmark_strategy : str | None
        Which strategy file to use to construct the realized-vol benchmark regime.
        If None, use the first available df that contains underlying returns.
    one_per_estimator : bool
        If True, reduce to one strategy per estimator before evaluation.

    Returns
    -------
    summary_table : pd.DataFrame
    benchmark_regime_df : pd.DataFrame | None
    """
    if one_per_estimator:
        data_map = reduce_to_one_per_estimator(data_map)

    if not data_map:
        return pd.DataFrame(), None

    benchmark_df = None
    benchmark_name = None

    if benchmark_strategy is not None and benchmark_strategy in data_map:
        benchmark_name = benchmark_strategy
        benchmark_df = data_map[benchmark_strategy]
    else:
        for name, df in data_map.items():
            under, _ = pick_underlying_returns(df)
            if under is not None:
                benchmark_name = name
                benchmark_df = df
                break

    if benchmark_df is None:
        print("⚠️ Could not find a strategy file with underlying returns for benchmark regime.")
        return pd.DataFrame(), None

    print(f"Using benchmark regime source: {benchmark_name}")

    benchmark_regime_df = build_realized_vol_benchmark_regime(
        benchmark_df,
        window=window,
        ann_factor=ann_factor,
    )

    if benchmark_regime_df is None or benchmark_regime_df.empty:
        return pd.DataFrame(), None

    all_rows = []
    for strat_name, df in data_map.items():
        try:
            metric_df = evaluate_one_strategy_by_benchmark_regime(
                df=df,
                benchmark_regime_df=benchmark_regime_df,
                window=window,
                ann_factor=ann_factor,
                horizon=horizon,
            )

            if metric_df.empty:
                continue

            metric_df.insert(0, "strategy", strat_name)
            metric_df.insert(1, "estimator", estimator_key_from_name(strat_name))
            all_rows.append(metric_df)

        except Exception as e:
            print(f"⚠️ Failed on {strat_name}: {e}")

    if not all_rows:
        return pd.DataFrame(), benchmark_regime_df

    summary = pd.concat(all_rows, ignore_index=True)

    regime_order = ["Low", "Mid", "High"]
    summary["regime"] = pd.Categorical(summary["regime"], categories=regime_order, ordered=True)

    wanted_cols = [
        "strategy",
        "estimator",
        "regime",
        "n_obs",
        "AnnualizedReturn",
        "AnnualizedVol",
        "OOS_R2",
        "Sharpe",
        "VaR_95",
        "MaxDD",
        "UlcerIndex",
        "QLIKE",
    ]

    summary = summary[
        [c for c in wanted_cols if c in summary.columns]
    ].sort_values(["estimator", "strategy", "regime"]).reset_index(drop=True)

    return summary, benchmark_regime_df
def plot_rf_vs_no_rf_difference_summary_by_regime(
    data_map: dict[str, pd.DataFrame],
    output_path,
    benchmark_strategy: str | None = None,
    ann_factor: float = 252.0,
):
    """
    One figure:
      - top: grouped bar chart of annualized return difference
             = ann_return(with_rf) - ann_return(no_rf)
             for every strategy, split by HMM regime (Low / Mid / High)
      - bottom: one table showing regime-conditioned differences in:
             annualized return, Sharpe, max drawdown

    Regimes are built from the EXISTING HMM benchmark regime logic:
        build_realized_vol_benchmark_regime(...)

    Required in each RF-enabled result parquet:
      - returns_with_rf
      - returns_no_rf

    Notes
    -----
    Difference columns are always:
        metric_diff = metric_with_rf - metric_no_rf
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from pathlib import Path

    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------
    # 1) choose benchmark df for HMM regime labeling
    # --------------------------------------------------
    benchmark_df = None
    benchmark_name = None

    if benchmark_strategy is not None and benchmark_strategy in data_map:
        benchmark_name = benchmark_strategy
        benchmark_df = data_map[benchmark_strategy]
    else:
        for name, df in data_map.items():
            if "asset_returns" in df.columns:
                benchmark_name = name
                benchmark_df = df
                break

    if benchmark_df is None:
        raise ValueError(
            "Could not find a benchmark strategy with underlying returns "
            "to build HMM benchmark regimes."
        )

    benchmark_regime_df = build_realized_vol_benchmark_regime(
        benchmark_df,
        window=21,
        ann_factor=ann_factor,
    )

    if benchmark_regime_df is None or benchmark_regime_df.empty:
        raise ValueError("HMM benchmark regime construction failed.")

    # --------------------------------------------------
    # 2) helper metrics on LOG returns
    #    (consistent with your RF engine output)
    # --------------------------------------------------
    def ann_return_from_log_returns(r: pd.Series, ann_factor: float = 252.0) -> float:
        r = pd.Series(r).astype(float).dropna()
        if len(r) == 0:
            return float("nan")
        return float(r.mean() * ann_factor)

    def sharpe_from_log_returns(r: pd.Series, ann_factor: float = 252.0) -> float:
        r = pd.Series(r).astype(float).dropna()
        if len(r) < 2:
            return float("nan")
        mu = float(r.mean())
        sigma = float(r.std(ddof=1))
        if not np.isfinite(sigma) or sigma <= 0:
            return float("nan")
        return float(np.sqrt(ann_factor) * mu / sigma)

    def max_drawdown_from_log_returns(r: pd.Series) -> float:
        r = pd.Series(r).astype(float).fillna(0.0)
        if len(r) == 0:
            return float("nan")
        eq = np.exp(np.cumsum(r.values))
        eq = pd.Series(eq, index=r.index)
        dd = eq / eq.cummax() - 1.0
        return float(dd.min())

    # --------------------------------------------------
    # 3) compute regime-conditioned diffs for each strategy
    # --------------------------------------------------
    regime_order = ["Low", "Mid", "High"]
    rows = []

    for name, df in data_map.items():
        required = ["returns_with_rf", "returns_no_rf"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            print(f"⚠️ Skip {name}: missing columns {missing}")
            continue

        r_with = pd.Series(df["returns_with_rf"], index=df.index).astype(float)
        r_no   = pd.Series(df["returns_no_rf"], index=df.index).astype(float)
        reg    = benchmark_regime_df["benchmark_regime"].reindex(df.index)

        tmp = pd.concat(
            [
                r_with.rename("with_rf"),
                r_no.rename("no_rf"),
                reg.rename("regime"),
            ],
            axis=1,
        ).dropna(subset=["regime"])

        if tmp.empty:
            print(f"⚠️ Skip {name}: no overlap with benchmark HMM regime.")
            continue

        for regime in regime_order:
            g = tmp[tmp["regime"] == regime].copy()
            if g.empty:
                continue

            rw = g["with_rf"].dropna()
            rn = g["no_rf"].dropna()

            if len(rw) < 2 or len(rn) < 2:
                continue

            ann_ret_with = ann_return_from_log_returns(rw, ann_factor=ann_factor)
            ann_ret_no   = ann_return_from_log_returns(rn, ann_factor=ann_factor)

            sharpe_with = sharpe_from_log_returns(rw, ann_factor=ann_factor)
            sharpe_no   = sharpe_from_log_returns(rn, ann_factor=ann_factor)

            maxdd_with = max_drawdown_from_log_returns(rw)
            maxdd_no   = max_drawdown_from_log_returns(rn)

            rows.append({
                "strategy": name,
                "regime": regime,
                "n_obs": int(len(g)),
                "ann_return_diff": ann_ret_with - ann_ret_no,
                "sharpe_diff": sharpe_with - sharpe_no,
                "maxdd_diff": maxdd_with - maxdd_no,
                "abs_maxdd_diff": abs(maxdd_with) - abs(maxdd_no),
            })

    out = pd.DataFrame(rows)
    if out.empty:
        print("❌ No valid strategies found for RF vs No-RF regime summary.")
        return

    out["regime"] = pd.Categorical(out["regime"], categories=regime_order, ordered=True)
    out = out.sort_values(["strategy", "regime"]).reset_index(drop=True)

    # --------------------------------------------------
    # 4) chart data: grouped bars for Δ annualized return
    # --------------------------------------------------
    pivot_ret = out.pivot(index="strategy", columns="regime", values="ann_return_diff")
    pivot_ret = pivot_ret.reindex(columns=regime_order)

    strategies = pivot_ret.index.tolist()
    x = np.arange(len(strategies))
    width = 0.25

    # --------------------------------------------------
    # 5) table data: one row per strategy,
    #    columns split by regime and metric
    # --------------------------------------------------
    table_df = out.pivot_table(
        index="strategy",
        columns="regime",
        values=["ann_return_diff", "sharpe_diff", "maxdd_diff"],
        aggfunc="first",
    )

    # flatten columns in regime-major order
    ordered_cols = []
    for reg in regime_order:
        for metric in ["ann_return_diff", "sharpe_diff", "maxdd_diff"]:
            if (metric, reg) in table_df.columns:
                ordered_cols.append((metric, reg))

    table_df = table_df[ordered_cols].copy()
    table_df.columns = [
        f"{reg}_ΔRet" if metric == "ann_return_diff" else
        f"{reg}_ΔSharpe" if metric == "sharpe_diff" else
        f"{reg}_ΔMaxDD"
        for metric, reg in table_df.columns
    ]
    table_df = table_df.reset_index()

    disp = table_df.copy()
    for c in disp.columns:
        if c == "strategy":
            continue
        if "ΔRet" in c or "ΔMaxDD" in c:
            disp[c] = disp[c].map(
                lambda x: f"{x:.2%}" if pd.notna(x) and np.isfinite(x) else "nan"
            )
        else:
            disp[c] = disp[c].map(
                lambda x: f"{x:.3f}" if pd.notna(x) and np.isfinite(x) else "nan"
            )

    # --------------------------------------------------
    # 6) plot
    # --------------------------------------------------
    fig_h = max(9, 5 + 0.45 * len(strategies))
    fig_w = max(18, 1.2 * (len(disp.columns) + 4))

    fig, (ax, ax_tbl) = plt.subplots(
        2, 1,
        figsize=(fig_w, fig_h),
        gridspec_kw={"height_ratios": [3, 2.1]},
        constrained_layout=True,
    )

    ax.bar(x - width, pivot_ret["Low"].values if "Low" in pivot_ret.columns else np.nan, width, label="Low")
    ax.bar(x,         pivot_ret["Mid"].values if "Mid" in pivot_ret.columns else np.nan, width, label="Mid")
    ax.bar(x + width, pivot_ret["High"].values if "High" in pivot_ret.columns else np.nan, width, label="High")

    ax.axhline(0.0, linestyle="--", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(strategies, rotation=45, ha="right")
    ax.set_ylabel("Δ Annualized Return")
    ax.set_title(f"With RF vs No RF: Annualized Return Difference by Strategy and HMM Regime\nBenchmark regime source: {benchmark_name}")
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    ax.legend()

    ax_tbl.axis("off")
    table = ax_tbl.table(
        cellText=disp.values,
        colLabels=disp.columns.tolist(),
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.5)

    out_file = output_path / "RF_vs_NoRF_Difference_Summary_By_HMM_Regime.png"
    csv_file = output_path / "RF_vs_NoRF_Difference_Summary_By_HMM_Regime.csv"

    fig.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close(fig)

    out.to_csv(csv_file, index=False)

    print(f"✅ Saved: {out_file}")
    print(f"✅ Saved: {csv_file}")
    return out, benchmark_regime_df
if __name__ == "__main__":
    data, path = get_data()
    plot_rf_vs_no_rf_difference_summary_by_regime(
        data_map=data,
        output_path=path,
        benchmark_strategy=None,   # or set a specific strategy name
        ann_factor=252.0,
    )
    plot_dim1_returns(data, path)
    plot_dim2_vol(data, path)
    plot_dim3_risk(data, path)
    plot_dim4_costs(data, path)
    data_map_est = reduce_to_one_per_estimator(data)
    plot_precision_table(data_map_est, path)
    plot_precision_timeseries(data_map_est, path)
    plot_precision_by_regime(data_map_est, path)
    print(f"Analysis plots generated successfully at: {path}")
    horizon = 0
    plot_rf_vs_no_rf_difference_summary(data, path)
    plot_regime_explainer_table(data_map_est, path, horizon=horizon)
    plot_abs_error_by_regime(data_map_est, path, horizon=horizon, top_k=6)
    plot_vol_of_vol_by_regime(data_map_est, path, horizon=horizon, top_k=6)
    plot_regime_strategy_metrics_table(data, path)
    plot_pred_vol_vs_realized_from_parquet(
        parquet_name="gjr_garch_regime.parquet",
        output_path=Env.path("results"),
        window=21,
        ann_factor=252.0,
        pred_col="vol_estimate",
    )

    summary_table, benchmark_regime_df = build_regime_metric_table(
        data_map=data,
        benchmark_strategy=None,   # or specify one file name
        window=21,
        ann_factor=252.0,
        horizon=0,
        one_per_estimator=False,
    )

    summary_table.to_csv(path / "regime_metrics_long.csv", index=False)
        


#def forward_filter_probs(model, obs: np.ndarray) -> np.ndarray:
#     """
#     Filtered probabilities P(S_t | y_1,...,y_t).
#     obs shape: (T, 1) or (T,)
#     """
#     obs = np.asarray(obs, dtype=float)
#     if obs.ndim == 1:
#         obs = obs.reshape(-1, 1)

#     T = obs.shape[0]
#     K = model.n_components

#     startprob = np.asarray(model.startprob_, dtype=float)
#     transmat = np.asarray(model.transmat_, dtype=float)

#     framelogprob = model._compute_log_likelihood(obs)
#     emission = np.exp(framelogprob)

#     alpha = np.zeros((T, K), dtype=float)

#     alpha[0] = startprob * emission[0]
#     s = alpha[0].sum()
#     if s <= 0 or not np.isfinite(s):
#         alpha[0] = np.ones(K) / K
#     else:
#         alpha[0] /= s

#     for t in range(1, T):
#         alpha[t] = (alpha[t - 1] @ transmat) * emission[t]
#         s = alpha[t].sum()
#         if s <= 0 or not np.isfinite(s):
#             alpha[t] = np.ones(K) / K
#         else:
#             alpha[t] /= s

#     return alpha
# from hmmlearn.hmm import GaussianHMM


# def hmm_smooth_vol_series(
#     vol_series: pd.Series,
#     min_hmm_obs: int = 126,
#     refit_every: int = 5,
#     smooth_span: int = 5,
# ) -> pd.Series:
#     """
#     HMM-smooth a positive volatility series.

#     Steps:
#     - optional EWM smoothing on log vol
#     - fit 3-state Gaussian HMM on historical log vol
#     - compute filtered state probabilities at each date
#     - convert filtered probs into smoothed log vol via p_t @ state_means
#     - return exp(smoothed_log_vol)

#     Output is aligned to the same date as the input series.
#     """
#     idx = vol_series.index
#     vol = pd.Series(vol_series, index=idx, dtype=float).copy()

#     # mild pre-smoothing
#     valid0 = np.isfinite(vol.values) & (vol.values > 0)
#     lv = pd.Series(np.nan, index=idx, dtype=float)
#     lv.loc[valid0] = np.log(vol.loc[valid0])
#     lv = lv.ewm(span=smooth_span, adjust=False).mean()
#     vol_used = np.exp(lv)

#     out = pd.Series(np.nan, index=idx, dtype=float)

#     valid = np.isfinite(vol_used.values) & (vol_used.values > 0)
#     valid_idx = idx[valid]
#     valid_log_vol = np.log(vol_used.loc[valid_idx]).to_numpy(dtype=float).reshape(-1, 1)

#     if len(valid_idx) < min_hmm_obs:
#         return out

#     last_model = None
#     last_state_means = None

#     for j in range(min_hmm_obs - 1, len(valid_idx)):
#         obs_hist = valid_log_vol[: j + 1]

#         should_refit = (last_model is None) or ((j - (min_hmm_obs - 1)) % refit_every == 0)

#         try:
#             if should_refit:
#                 model = GaussianHMM(
#                     n_components=3,
#                     covariance_type="diag",
#                     n_iter=1000,
#                     random_state=42,
#                     init_params="mc",
#                     params="stmc",
#                 )
#                 model.startprob_ = np.array([0.33, 0.33, 0.34], dtype=float)
#                 model.transmat_ = np.array(
#                     [
#                         [0.96, 0.03, 0.01],
#                         [0.03, 0.94, 0.03],
#                         [0.01, 0.03, 0.96],
#                     ],
#                     dtype=float,
#                 )
#                 model.fit(obs_hist)
#                 state_means = model.means_.reshape(-1)

#                 last_model = model
#                 last_state_means = state_means
#             else:
#                 model = last_model
#                 state_means = last_state_means

#             filtered = forward_filter_probs(model, obs_hist)
#             p_t = filtered[-1]

#             smoothed_log_vol_t = float(p_t @ state_means)
#             smoothed_vol_t = float(np.exp(smoothed_log_vol_t))

#             out.loc[valid_idx[j]] = smoothed_vol_t

#         except Exception:
#             continue

#     return out
# def classify_series_by_expanding_terciles(
#     x: pd.Series,
#     min_obs: int = 126,
#     labels: tuple[str, str, str] = ("Low", "Mid", "High"),
# ) -> pd.Series:
#     """
#     Classify a volatility series into Low/Mid/High using expanding historical terciles.

#     For date t:
#       - thresholds are computed from x[:t-1]
#       - x[t] is classified using those thresholds

#     This avoids look-ahead bias and keeps the benchmark construction
#     consistent with the prediction-side regime mapping.
#     """
#     x = pd.Series(x, index=x.index, dtype=float)
#     out = pd.Series(np.nan, index=x.index, dtype="object")

#     vals = x.to_numpy(dtype=float)

#     for i in range(len(x)):
#         xi = vals[i]
#         hist = vals[:i]
#         hist = hist[np.isfinite(hist) & (hist > 0)]

#         if (not np.isfinite(xi)) or (xi <= 0) or (len(hist) < min_obs):
#             continue

#         q1, q2 = np.quantile(hist, [1/3, 2/3])

#         if xi <= q1:
#             out.iloc[i] = labels[0]
#         elif xi <= q2:
#             out.iloc[i] = labels[1]
#         else:
#             out.iloc[i] = labels[2]

#     return out




# def get_predicted_regime_labels(
#     df: pd.DataFrame,
#     index: pd.Index | None = None,
#     col: str = "pred_next_regime",
# ) -> pd.Series | None:
#     """
#     Return standardized predicted regime labels (Low/Mid/High).
#     Accepts values like: low/middle/high (case-insensitive).
#     If index is provided, reindex to that index.
#     """
#     if col not in df.columns:
#         return None

#     def norm(x):
#         if pd.isna(x):
#             return np.nan
#         s = str(x).strip().lower()
#         if s == "low":
#             return "Low"
#         if s in ("mid", "middle", "med", "medium"):
#             return "Mid"
#         if s == "high":
#             return "High"
#         return np.nan

#     reg = df[col].map(norm)
#     reg = pd.Series(reg, index=df.index)
#     if index is not None:
#         reg = reg.reindex(index)
#     return reg


# def regime_to_num(s: pd.Series) -> pd.Series:
#     """
#     Map regime labels to numeric values for plotting.
#     Low -> 0, Mid -> 1, High -> 2
#     """
#     mp = {"Low": 0.0, "Mid": 1.0, "High": 2.0}
#     return s.map(mp)


# def build_realized_regime_benchmark(
#     data_map: dict[str, pd.DataFrame],
#     window: int = 21,
#     ann_factor: float = 252.0,
#     min_obs: int = 126,
#     hmm_min_obs: int = 126,
#     hmm_refit_every: int = 5,
#     hmm_smooth_span: int = 5,
# ) -> pd.Series | None:
#     """
#     Build benchmark regime series from realized volatility proxy of the underlying,
#     using HMM-smoothed realized volatility before regime classification.
#     """
#     for name, df in data_map.items():
#         under, src = pick_underlying_returns(df)
#         if under is None:
#             continue

#         rv = realized_vol_proxy(under, window=window, ann_factor=ann_factor)
#         rv = pd.Series(rv, index=rv.index, dtype=float)

#         rv_hmm = hmm_smooth_vol_series(
#             rv,
#             min_hmm_obs=hmm_min_obs,
#             refit_every=hmm_refit_every,
#             smooth_span=hmm_smooth_span,
#         )

#         bench = classify_series_by_expanding_terciles(rv_hmm, min_obs=min_obs)
#         bench.name = "benchmark_regime"
#         return bench

#     return None


# def regime_accuracy(pred: pd.Series, truth: pd.Series) -> float:
#     m = pd.concat([pred.rename("pred"), truth.rename("truth")], axis=1).dropna()
#     if len(m) == 0:
#         return float("nan")
#     return float((m["pred"] == m["truth"]).mean())

# def plot_predicted_regimes_vs_benchmark(
#     data_map: dict[str, pd.DataFrame],
#     output_path: Path,
#     window: int = 21,
#     ann_factor: float = 252.0,
#     use_one_per_estimator: bool = True,
#     min_obs: int = 126,
#     hmm_min_obs: int = 126,
#     hmm_refit_every: int = 5,
#     hmm_smooth_span: int = 5,
# ):
#     """
#     Compare each estimator's predicted regime column against a benchmark regime
#     derived from the realized volatility proxy.

#     Expected prediction column:
#         pred_next_regime
#     which should already be aligned to the actual target date.
#     """
#     if use_one_per_estimator:
#         data_map = reduce_to_one_per_estimator(data_map)

#     benchmark = build_realized_regime_benchmark(
#         data_map,
#         window=window,
#         ann_factor=ann_factor,
#         min_obs=min_obs,
#         hmm_min_obs=hmm_min_obs,
#         hmm_refit_every=hmm_refit_every,
#         hmm_smooth_span=hmm_smooth_span,
#     )
#     if benchmark is None:
#         print("⚠️ Could not build realized-vol benchmark.")
#         return

#     plotted = []

#     # collect aligned predicted regimes
#     for strat_name, df in data_map.items():
#         pred = get_predicted_regime_labels(df, index=benchmark.index, col="pred_next_regime")
#         if pred is None:
#             continue

#         m = pd.concat(
#             [
#                 pred.rename("pred"),
#                 benchmark.rename("benchmark"),
#             ],
#             axis=1,
#         ).dropna()

#         if len(m) == 0:
#             continue

#         acc = regime_accuracy(m["pred"], m["benchmark"])
#         plotted.append((strat_name, m, acc))

#     if not plotted:
#         print("⚠️ No strategy files contain usable pred_next_regime series.")
#         return

#     n = len(plotted)
#     fig_h = max(3.0 * n, 4.0)
#     fig, axes = plt.subplots(n, 1, figsize=(16, fig_h), sharex=True)

#     if n == 1:
#         axes = [axes]

#     for ax, (name, m, acc) in zip(axes, plotted):
#         y_bench = regime_to_num(m["benchmark"])
#         y_pred = regime_to_num(m["pred"])

#         ax.step(m.index, y_bench, where="post", linewidth=2, label="Benchmark (realized vol terciles)")
#         ax.step(m.index, y_pred, where="post", linewidth=1.5, alpha=0.9, label=f"{name} predicted")

#         ax.set_yticks([0, 1, 2])
#         ax.set_yticklabels(["Low", "Mid", "High"])
#         ax.set_ylim(-0.3, 2.3)
#         ax.grid(True, alpha=0.3)
#         ax.legend(loc="upper left")
#         ax.set_title(f"{name} vs benchmark   |   accuracy = {acc:.2%}")

#     axes[-1].set_xlabel("Date")
#     fig.suptitle("Predicted Regime vs Realized-Vol Benchmark", y=0.995)
#     fig.tight_layout()

#     output_path.parent.mkdir(parents=True, exist_ok=True)
#     fig.savefig(output_path, dpi=200, bbox_inches="tight")
#     plt.close(fig)

#     print(f"✅ Saved regime comparison plot to: {output_path}")