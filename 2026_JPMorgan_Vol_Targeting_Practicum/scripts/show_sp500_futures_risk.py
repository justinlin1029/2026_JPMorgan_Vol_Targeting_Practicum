"""Show drawdown and CVaR for the underlying S&P futures return series.

The script reads ``asset_returns`` from any backtest result parquet produced by
the engine, treats that column as the raw underlying futures log-return series,
converts it to simple returns, and reports:

  - Max drawdown
  - Monthly average drawdown
  - CVaR / Expected Shortfall at a configurable confidence level

It also saves a figure with the cumulative equity path, drawdown series, and
tail-return histogram.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".mplconfig"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.env import Env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute drawdown and CVaR for raw S&P futures returns from an engine result parquet."
    )
    parser.add_argument(
        "--result",
        default="",
        help="Path to one result parquet file. If omitted, the first parquet containing asset_returns is used.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.95,
        help="Confidence level for CVaR / Expected Shortfall. Default: 0.95",
    )
    parser.add_argument(
        "--output-dir",
        default=str(Env.path("results") / "sp500_futures_risk"),
        help="Directory for the output plot and summary CSV.",
    )
    return parser.parse_args()


def pick_result_file(path_arg: str) -> Path:
    if path_arg:
        path = Path(path_arg)
        if not path.exists():
            raise FileNotFoundError(f"Result parquet not found: {path}")
        return path

    for path in sorted(Env.path("results").glob("*.parquet")):
        try:
            df = pd.read_parquet(path, columns=["asset_returns"])
            if "asset_returns" in df.columns:
                return path
        except Exception:
            continue
    raise FileNotFoundError("Could not find any result parquet with an 'asset_returns' column.")


def max_drawdown_from_simple_returns(ret: pd.Series) -> tuple[float, pd.Series, pd.Series]:
    r = pd.Series(ret).astype(float).dropna()
    wealth = (1.0 + r).cumprod()
    peak = wealth.cummax()
    dd = wealth / peak - 1.0
    return float(dd.min()), wealth, dd


def monthly_average_drawdown_from_simple_returns(ret: pd.Series) -> float:
    r = pd.Series(ret).astype(float).dropna()
    if r.empty:
        return float("nan")
    vals: list[float] = []
    for _, block in r.groupby(r.index.to_period("M")):
        if block.empty:
            continue
        mdd, _, _ = max_drawdown_from_simple_returns(block)
        vals.append(mdd)
    return float(np.mean(vals)) if vals else float("nan")


def cvar_from_simple_returns(ret: pd.Series, alpha: float = 0.95) -> tuple[float, float]:
    r = pd.Series(ret).astype(float).dropna()
    if r.empty:
        return float("nan"), float("nan")
    cutoff = float(np.percentile(r, (1.0 - alpha) * 100.0))
    tail = r[r <= cutoff]
    cvar = float(-tail.mean()) if not tail.empty else float("nan")
    return cvar, cutoff


def annualized_volatility(ret: pd.Series, ann_factor: float = 252.0) -> float:
    r = pd.Series(ret).astype(float).dropna()
    if len(r) < 2:
        return float("nan")
    return float(r.std(ddof=1) * np.sqrt(ann_factor))


def rolling_annualized_volatility(ret: pd.Series, window: int = 21, ann_factor: float = 252.0) -> pd.Series:
    r = pd.Series(ret).astype(float)
    return (r.rolling(window, min_periods=window).std(ddof=1) * np.sqrt(ann_factor)).rename("rolling_ann_vol")


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
    if rv_df.index.tz is not None:
        rv_df.index = rv_df.index.tz_localize(None)
    rv_df.index = rv_df.index.normalize()
    rv_df = rv_df.sort_index()

    if vol_col not in rv_df.columns:
        raise ValueError(f"Missing '{vol_col}' in {rv_path}. Found columns: {list(rv_df.columns)}")

    rv_df[vol_col] = pd.to_numeric(rv_df[vol_col], errors="coerce")
    if "coverage" in rv_df.columns:
        rv_df["coverage"] = pd.to_numeric(rv_df["coverage"], errors="coerce")
    if min_coverage_ratio is not None and "coverage" in rv_df.columns:
        rv_df = rv_df[rv_df["coverage"] >= float(min_coverage_ratio)].copy()
    return rv_df[vol_col].astype(float).rename(vol_col)


def compute_hmm_regime_from_series(
    vol_series: pd.Series,
    n_components: int = 3,
    random_state: int = 42,
) -> pd.Series:
    vol = pd.Series(vol_series).astype(float)
    mask = np.isfinite(vol.values) & (vol.values > 0)
    regime_label = pd.Series(pd.NA, index=vol.index, dtype="object")
    if mask.sum() < 30:
        return regime_label

    log_vol = np.log(vol.loc[mask]).values.reshape(-1, 1)
    model = GaussianHMM(
        n_components=n_components,
        covariance_type="diag",
        n_iter=1000,
        random_state=random_state,
        init_params="mc",
        params="stmc",
    )
    model.startprob_ = np.array([0.33, 0.33, 0.34])
    model.transmat_ = np.array(
        [
            [0.96, 0.03, 0.01],
            [0.03, 0.94, 0.03],
            [0.01, 0.03, 0.96],
        ]
    )
    model.fit(log_vol)
    states = model.predict(log_vol)
    state_means = model.means_.reshape(-1)
    order = np.argsort(state_means)
    ordered_labels = np.array(["Low", "Mid", "High"])
    state_to_label = {int(order[k]): ordered_labels[k] for k in range(n_components)}
    regime_label.loc[mask] = pd.Series(states, index=vol.loc[mask].index).map(state_to_label).values
    return regime_label


def build_regime_frame(index: pd.Index) -> pd.DataFrame:
    idx = pd.to_datetime(index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    idx = idx.normalize()

    rv = load_intraday_realized_vol()
    regime = compute_hmm_regime_from_series(rv)
    out = pd.DataFrame(index=idx)
    out["benchmark_rv"] = rv.reindex(idx)
    out["regime"] = regime.reindex(idx)
    return out


def load_underlying_simple_returns(result_path: Path) -> pd.Series:
    df = pd.read_parquet(result_path)
    if "asset_returns" not in df.columns:
        raise ValueError(f"{result_path} does not contain 'asset_returns'.")
    idx = pd.to_datetime(df.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    s = pd.Series(df["asset_returns"], index=idx).astype(float)
    # Engine stores log returns for the underlying risky asset.
    return np.expm1(s).rename("asset_simple_returns")


def drawdown_duration_series_from_simple_returns(ret: pd.Series) -> pd.Series:
    r = pd.Series(ret).astype(float).dropna()
    wealth = (1.0 + r).cumprod()
    peak = wealth.cummax()
    dd = wealth / peak - 1.0

    durations = []
    current = 0
    for value in dd:
        if value < 0:
            current += 1
        else:
            current = 0
        durations.append(current)
    return pd.Series(durations, index=dd.index, name="drawdown_duration")


def summarize_regime_drawdown_metrics(simple_ret: pd.Series, regime_df: pd.DataFrame) -> pd.DataFrame:
    tmp = pd.DataFrame(index=simple_ret.index)
    tmp["ret"] = pd.Series(simple_ret, index=simple_ret.index).astype(float)
    tmp["regime"] = regime_df["regime"].reindex(tmp.index)
    tmp = tmp.dropna(subset=["regime", "ret"])
    if tmp.empty:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    for regime in ["Low", "Mid", "High"]:
        g = tmp[tmp["regime"] == regime].copy()
        if g.empty:
            continue
        max_dd, _, _ = max_drawdown_from_simple_returns(g["ret"])
        duration = drawdown_duration_series_from_simple_returns(g["ret"])
        rows.append(
            {
                "regime": regime,
                "n_obs": int(len(g)),
                "max_drawdown": max_dd,
                "max_drawdown_duration_days": int(duration.max()) if not duration.empty else 0,
            }
        )
    return pd.DataFrame(rows)


def plot_regime_bar_chart(
    regime_summary: pd.DataFrame,
    value_col: str,
    title: str,
    ylabel: str,
    percent: bool,
    output_path: Path,
) -> None:
    if regime_summary.empty:
        return

    plot_df = regime_summary.copy()
    order = ["Low", "Mid", "High"]
    plot_df["regime"] = pd.Categorical(plot_df["regime"], categories=order, ordered=True)
    plot_df = plot_df.sort_values("regime")

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#4C78A8", "#F58518", "#54A24B"]
    vals = plot_df[value_col].to_numpy(dtype=float)
    ax.bar(plot_df["regime"].astype(str), vals, color=colors)
    ax.set_title(title, fontsize=15, pad=10)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Regime")
    ax.grid(axis="y", linestyle="--", alpha=0.25)

    for i, value in enumerate(vals):
        label = f"{value:.2%}" if percent else f"{int(value)}"
        ax.text(i, value, label, ha="center", va="bottom", fontsize=10)

    fig.savefig(output_path, dpi=260, bbox_inches="tight")
    plt.close(fig)


def plot_risk_figure(
    simple_ret: pd.Series,
    wealth: pd.Series,
    drawdown: pd.Series,
    rolling_ann_vol: pd.Series,
    cutoff: float,
    alpha: float,
    summary: pd.DataFrame,
    output_path: Path,
) -> None:
    fig = plt.figure(figsize=(16, 13))
    gs = fig.add_gridspec(4, 1, height_ratios=[2.0, 1.5, 1.5, 1.6], hspace=0.24)

    ax1 = fig.add_subplot(gs[0])
    ax1.plot(wealth.index, wealth.values, color="#1f77b4", linewidth=1.6)
    ax1.set_title("Underlying S&P Futures Equity Path", fontsize=15, pad=10)
    ax1.set_ylabel("Wealth Index")
    ax1.grid(alpha=0.25, linestyle="--")

    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax2.fill_between(drawdown.index, drawdown.values, 0.0, color="#d62728", alpha=0.75)
    ax2.set_title("Drawdown", fontsize=14, pad=8)
    ax2.set_ylabel("Drawdown")
    ax2.grid(alpha=0.25, linestyle="--")

    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    ax3.plot(rolling_ann_vol.index, rolling_ann_vol.values, color="#9467bd", linewidth=1.4)
    ax3.set_title("21-Day Rolling Annualized Volatility", fontsize=14, pad=8)
    ax3.set_ylabel("Ann. Vol")
    ax3.grid(alpha=0.25, linestyle="--")

    ax4 = fig.add_subplot(gs[3])
    vals = simple_ret.dropna().to_numpy(dtype=float)
    ax4.hist(vals, bins=60, color="#2ca02c", alpha=0.75, edgecolor="white")
    ax4.axvline(cutoff, color="#d62728", linestyle="--", linewidth=1.5, label=f"{int(alpha*100)}% VaR cutoff")
    ax4.set_title("Return Distribution and CVaR Tail", fontsize=14, pad=8)
    ax4.set_xlabel("Simple Daily Return")
    ax4.set_ylabel("Frequency")
    ax4.legend()
    ax4.grid(alpha=0.2, linestyle="--")

    summary_text = "\n".join(f"{row.metric}: {row.value}" for row in summary.itertuples(index=False))
    fig.text(
        0.79,
        0.35,
        summary_text,
        ha="left",
        va="top",
        fontsize=10,
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "#cccccc", "boxstyle": "round,pad=0.5"},
    )

    fig.savefig(output_path, dpi=260, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    result_path = pick_result_file(args.result)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    simple_ret = load_underlying_simple_returns(result_path)
    max_dd, wealth, drawdown = max_drawdown_from_simple_returns(simple_ret)
    monthly_avg_dd = monthly_average_drawdown_from_simple_returns(simple_ret)
    cvar, cutoff = cvar_from_simple_returns(simple_ret, alpha=args.alpha)
    ann_vol = annualized_volatility(simple_ret)
    rolling_ann_vol = rolling_annualized_volatility(simple_ret)
    regime_df = build_regime_frame(simple_ret.index)
    regime_summary = summarize_regime_drawdown_metrics(simple_ret, regime_df)

    summary = pd.DataFrame(
        [
            {"metric": "Source Result", "value": result_path.name},
            {"metric": "Observations", "value": str(int(simple_ret.dropna().shape[0]))},
            {"metric": "Annualized Volatility", "value": f"{ann_vol:.2%}"},
            {"metric": "Max Drawdown", "value": f"{max_dd:.2%}"},
            {"metric": "Monthly Avg Drawdown", "value": f"{monthly_avg_dd:.2%}"},
            {"metric": f"CVaR {int(args.alpha * 100)}%", "value": f"{cvar:.2%}"},
            {"metric": f"VaR cutoff {int(args.alpha * 100)}%", "value": f"{cutoff:.2%}"},
        ]
    )

    stem = result_path.stem
    csv_path = output_dir / f"{stem}__sp500_futures_risk_summary.csv"
    png_path = output_dir / f"{stem}__sp500_futures_risk.png"
    regime_csv_path = output_dir / f"{stem}__sp500_futures_regime_drawdown_summary.csv"
    regime_mdd_png = output_dir / f"{stem}__sp500_futures_max_drawdown_by_regime.png"
    regime_duration_png = output_dir / f"{stem}__sp500_futures_drawdown_duration_by_regime.png"

    summary.to_csv(csv_path, index=False)
    regime_summary.to_csv(regime_csv_path, index=False)
    plot_risk_figure(simple_ret, wealth, drawdown, rolling_ann_vol, cutoff, args.alpha, summary, png_path)
    plot_regime_bar_chart(
        regime_summary,
        value_col="max_drawdown",
        title="Max Drawdown by HMM Volatility Regime",
        ylabel="Max Drawdown",
        percent=True,
        output_path=regime_mdd_png,
    )
    plot_regime_bar_chart(
        regime_summary,
        value_col="max_drawdown_duration_days",
        title="Max Drawdown Duration by HMM Volatility Regime",
        ylabel="Days",
        percent=False,
        output_path=regime_duration_png,
    )

    print(f"Source result: {result_path}")
    print(summary.to_string(index=False))
    if not regime_summary.empty:
        print("\nBy regime")
        print(regime_summary.to_string(index=False))
    print(f"Saved summary CSV to: {csv_path}")
    print(f"Saved figure to: {png_path}")
    print(f"Saved regime summary CSV to: {regime_csv_path}")
    print(f"Saved max drawdown by regime figure to: {regime_mdd_png}")
    print(f"Saved drawdown duration by regime figure to: {regime_duration_png}")


if __name__ == "__main__":
    main()
