"""Build picture-based summaries from estimator_controller_metrics_by_regime.csv."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import textwrap

os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".mplconfig"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from hmmlearn.hmm import GaussianHMM


REGIME_ORDER = ["Low", "Mid", "High"]
REGIME_COLORS = {
    "Low": "#2ecc71",
    "Mid": "#f39c12",
    "High": "#e74c3c",
}
REGIME_LABELS = {
    "Low": "Low Vol",
    "Mid": "Mid Vol",
    "High": "High Vol",
}
DEFAULT_EXCLUDED_ESTIMATORS = ["buy_and_hold", "xgb_vix", "ar1"]
DEFAULT_EXCLUDED_CONTROLLERS = ["c_va_res_targeting"]
STRATEGY_METRICS = {
    "Sharpe": False,
    "MonthlyAverageDrawdown": False,
    "CVaR": True,
    "Turnover": True,
    "WinRate": False,
    "AnnualizedReturn": False,
}
METRIC_LABELS = {
    "Sharpe": "Sharpe",
    "MaxDrawdown": "Max Drawdown",
    "DrawdownDuration": "Drawdown Duration",
    "MonthlyAverageDrawdown": "Monthly Avg Drawdown",
    "CVaR": "CVaR",
    "Turnover": "Turnover",
    "WinRate": "Win Rate",
    "AnnualizedReturn": "Annualized Return",
}
REGIME_ONLY_METRICS = {
    "MaxDrawdown": False,
    "DrawdownDuration": True,
}
TABLE_SHOW_OVERALL_VOL_METRICS = {
    "Sharpe",
    "MaxDrawdown",
    "MonthlyAverageDrawdown",
    "CVaR",
    "Turnover",
}
ESTIMATOR_METRICS = [
    "QLIKE",
    "OOS_R2",
    "MSE",
    "MincerAlpha",
    "MincerBeta",
    "MincerR2",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Use results/estimator_controller_metrics_by_regime.csv to generate "
            "picture-based overall and regime summary reports."
        )
    )
    parser.add_argument(
        "--input",
        default="results/estimator_controller_metrics_by_regime.csv",
        help="Input CSV exported by scripts/export_regime_metrics_csv.py.",
    )
    parser.add_argument(
        "--output-dir",
        default="results/metric_summary_pictures",
        help="Directory where summary PNGs and CSVs will be written.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=5,
        help="Number of top strategies to keep per metric.",
    )
    parser.add_argument(
        "--exclude-estimators",
        nargs="*",
        default=DEFAULT_EXCLUDED_ESTIMATORS,
        help="Optional estimator names to exclude.",
    )
    parser.add_argument(
        "--exclude-controllers",
        nargs="*",
        default=DEFAULT_EXCLUDED_CONTROLLERS,
        help="Optional controller names to exclude.",
    )
    return parser.parse_args()


def weighted_mean(series: pd.Series, weights: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce")
    w = pd.to_numeric(weights, errors="coerce")
    mask = values.notna() & w.notna()
    if not mask.any():
        return float("nan")
    total_weight = float(w.loc[mask].sum())
    if total_weight <= 0:
        return float("nan")
    return float(np.average(values.loc[mask], weights=w.loc[mask]))


def format_metric_value(value: float, metric: str) -> str:
    if pd.isna(value):
        return ""
    if metric == "DrawdownDuration":
        return f"{int(round(value))}d"
    if metric in {"WinRate", "AnnualizedReturn", "MonthlyAverageDrawdown", "CVaR"}:
        return f"{value:.2%}"
    return f"{value:.4f}"


def load_input(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {
        "strategy",
        "estimator",
        "controller",
        "regime",
        "n_obs",
        *STRATEGY_METRICS.keys(),
        *ESTIMATOR_METRICS,
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns in {path}: {missing}")

    df["regime"] = pd.Categorical(df["regime"], categories=REGIME_ORDER, ordered=True)
    return df.sort_values(["estimator", "controller", "regime"]).reset_index(drop=True)


def exclude_estimators(df: pd.DataFrame, excluded_estimators: list[str]) -> pd.DataFrame:
    if not excluded_estimators:
        return df.copy()
    excluded = set(excluded_estimators)
    return df.loc[~df["estimator"].isin(excluded)].copy().reset_index(drop=True)


def exclude_controllers(df: pd.DataFrame, excluded_controllers: list[str]) -> pd.DataFrame:
    if not excluded_controllers:
        return df.copy()
    excluded = set(excluded_controllers)
    return df.loc[~df["controller"].isin(excluded)].copy().reset_index(drop=True)


def aggregate_strategy_overall(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (strategy, estimator, controller), group in df.groupby(
        ["strategy", "estimator", "controller"], sort=False
    ):
        row: dict[str, object] = {
            "strategy": strategy,
            "estimator": estimator,
            "controller": controller,
            "n_obs": int(pd.to_numeric(group["n_obs"], errors="coerce").sum()),
        }
        for metric in STRATEGY_METRICS:
            row[metric] = weighted_mean(group[metric], group["n_obs"])
        if "AnnualizedVol" in group.columns:
            row["AnnualizedVol"] = weighted_mean(group["AnnualizedVol"], group["n_obs"])
        rows.append(row)
    return pd.DataFrame(rows)


def collapse_estimator_metrics(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (estimator, regime), group in df.groupby(["estimator", "regime"], sort=False, observed=True):
        row: dict[str, object] = {
            "estimator": estimator,
            "regime": regime,
            "n_obs": int(pd.to_numeric(group["n_obs"], errors="coerce").max()),
        }
        if "ForecastSourceStrategy" in group.columns:
            non_na_sources = group["ForecastSourceStrategy"].dropna()
            row["ForecastSourceStrategy"] = non_na_sources.iloc[0] if not non_na_sources.empty else ""
        for metric in ESTIMATOR_METRICS:
            row[metric] = weighted_mean(group[metric], group["n_obs"])
        rows.append(row)

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["regime"] = pd.Categorical(out["regime"], categories=REGIME_ORDER, ordered=True)
    return out.sort_values(["regime", "QLIKE", "estimator"]).reset_index(drop=True)


def aggregate_estimator_overall(estimator_regime_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for estimator, group in estimator_regime_df.groupby("estimator", sort=False):
        row: dict[str, object] = {
            "estimator": estimator,
            "n_obs": int(pd.to_numeric(group["n_obs"], errors="coerce").sum()),
        }
        if "ForecastSourceStrategy" in group.columns:
            non_na_sources = group["ForecastSourceStrategy"].dropna()
            row["ForecastSourceStrategy"] = non_na_sources.iloc[0] if not non_na_sources.empty else ""
        for metric in ESTIMATOR_METRICS:
            row[metric] = weighted_mean(group[metric], group["n_obs"])
        rows.append(row)

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = out.sort_values(["QLIKE", "estimator"], ascending=[True, True]).reset_index(drop=True)
    out.insert(0, "rank_by_QLIKE", np.arange(1, len(out) + 1))
    return out


def attach_overall_annualized_vol(
    df: pd.DataFrame,
    strategy_overall: pd.DataFrame,
    column_name: str = "OverallAnnualizedVol",
) -> pd.DataFrame:
    if "AnnualizedVol" not in strategy_overall.columns:
        return df.copy()

    vol_map = strategy_overall.loc[:, ["strategy", "AnnualizedVol"]].rename(
        columns={"AnnualizedVol": column_name}
    )
    out = df.merge(vol_map, on="strategy", how="left")
    return out


def top_n_overall_for_metric(strategy_overall: pd.DataFrame, metric: str, top_n: int) -> pd.DataFrame:
    ascending = STRATEGY_METRICS[metric]
    columns = ["strategy", "estimator", "controller", "n_obs", metric]
    if "AnnualizedVol" in strategy_overall.columns:
        columns.append("AnnualizedVol")
    out = (
        strategy_overall.loc[:, columns]
        .dropna(subset=[metric])
        .sort_values(metric, ascending=ascending)
        .head(top_n)
        .reset_index(drop=True)
    )
    out.insert(0, "rank", np.arange(1, len(out) + 1))
    return out


def top_n_by_regime_for_metric(metrics_df: pd.DataFrame, metric: str, top_n: int) -> pd.DataFrame:
    ascending = STRATEGY_METRICS[metric]
    frames: list[pd.DataFrame] = []
    for regime in REGIME_ORDER:
        part = metrics_df.loc[metrics_df["regime"] == regime].copy()
        columns = ["strategy", "estimator", "controller", "n_obs", metric]
        if "AnnualizedVol" in part.columns:
            columns.append("AnnualizedVol")
        if "OverallAnnualizedVol" in part.columns:
            columns.append("OverallAnnualizedVol")
        ranked = (
            part.loc[:, columns]
            .dropna(subset=[metric])
            .sort_values(metric, ascending=ascending)
            .head(top_n)
            .reset_index(drop=True)
        )
        ranked.insert(0, "rank", np.arange(1, len(ranked) + 1))
        ranked.insert(0, "regime", regime)
        frames.append(ranked)
    return pd.concat(frames, ignore_index=True)


def top_n_by_regime_custom_metric(
    metrics_df: pd.DataFrame,
    metric: str,
    top_n: int,
    ascending: bool,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for regime in REGIME_ORDER:
        part = metrics_df.loc[metrics_df["regime"] == regime].copy()
        columns = ["strategy", "estimator", "controller", "n_obs", metric]
        if "AnnualizedVol" in part.columns:
            columns.append("AnnualizedVol")
        if "OverallAnnualizedVol" in part.columns:
            columns.append("OverallAnnualizedVol")
        ranked = (
            part.loc[:, columns]
            .dropna(subset=[metric])
            .sort_values(metric, ascending=ascending)
            .head(top_n)
            .reset_index(drop=True)
        )
        ranked.insert(0, "rank", np.arange(1, len(ranked) + 1))
        ranked.insert(0, "regime", regime)
        frames.append(ranked)
    return pd.concat(frames, ignore_index=True)


def sanitize_name(text: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in text).strip("_")


def wrap_text(value: object, width: int = 26) -> str:
    text = str(value)
    return "\n".join(textwrap.wrap(text, width=width)) if len(text) > width else text


def compact_name(value: object, max_len: int = 42) -> str:
    text = str(value)
    if len(text) <= max_len:
        return text
    return f"{text[:max_len-3]}..."


def plot_top5_metric_overall(top_df: pd.DataFrame, metric: str, output_path: Path) -> None:
    fig = plt.figure(figsize=(16, 10.5))
    gs = fig.add_gridspec(2, 1, height_ratios=[2.1, 2.2], hspace=0.08)

    ax = fig.add_subplot(gs[0])
    vals = top_df[metric].to_numpy(dtype=float)
    colors = plt.cm.Blues(np.linspace(0.45, 0.9, len(top_df)))
    y_labels = top_df["strategy"].map(lambda x: wrap_text(x, width=28))
    ax.barh(y_labels, vals, color=colors)
    ax.invert_yaxis()
    ax.set_title(f"Top {len(top_df)} Strategies Overall by {METRIC_LABELS[metric]}", fontsize=16, pad=12)
    ax.set_xlabel(METRIC_LABELS[metric])
    ax.set_ylabel("Strategy")
    ax.grid(axis="x", linestyle="--", alpha=0.25)
    ax.tick_params(axis="y", labelsize=9)

    for i, value in enumerate(vals):
        ax.text(value, i, f" {format_metric_value(value, metric)}", va="center", ha="left", fontsize=9)

    ax_tbl = fig.add_subplot(gs[1])
    ax_tbl.axis("off")
    table_df = top_df.copy()
    table_df["strategy"] = table_df["strategy"].map(lambda x: compact_name(x, max_len=36))
    table_df["controller"] = table_df["controller"].map(lambda x: compact_name(x, max_len=24))
    table_df[metric] = table_df[metric].map(lambda x: format_metric_value(x, metric))
    columns = ["rank", "strategy", "estimator", "controller", metric]
    if metric in TABLE_SHOW_OVERALL_VOL_METRICS and "AnnualizedVol" in table_df.columns:
        table_df["Overall Annualized Vol"] = table_df["AnnualizedVol"].map(
            lambda x: f"{x:.2%}" if pd.notna(x) else ""
        )
        columns.append("Overall Annualized Vol")
    table_df = table_df[columns]
    table = ax_tbl.table(
        cellText=table_df.values,
        colLabels=table_df.columns,
        loc="center",
        cellLoc="left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9.5)
    table.scale(1.08, 2.05)

    fig.savefig(output_path, dpi=260, bbox_inches="tight")
    plt.close(fig)


def plot_top5_metric_by_regime(top_df: pd.DataFrame, metric: str, output_path: Path) -> None:
    fig = plt.figure(figsize=(22, 15))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.8, 2.4], hspace=0.12, wspace=0.28)
    palette = ["#4C78A8", "#F58518", "#54A24B"]

    for i, regime in enumerate(REGIME_ORDER):
        ax = fig.add_subplot(gs[0, i])
        part = top_df.loc[top_df["regime"] == regime].copy()
        vals = part[metric].to_numpy(dtype=float)
        y_labels = [
            wrap_text(f"{int(rank)}. {compact_name(strategy, max_len=30)}", width=18)
            for rank, strategy in part[["rank", "strategy"]].itertuples(index=False, name=None)
        ]
        ax.barh(y_labels, vals, color=palette[i])
        ax.invert_yaxis()
        ax.set_title(f"{regime} Regime")
        ax.set_xlabel(METRIC_LABELS[metric])
        if i == 0:
            ax.set_ylabel("Strategy")
        ax.grid(axis="x", linestyle="--", alpha=0.25)
        ax.tick_params(axis="y", labelsize=7.5, pad=4)
        ax.margins(y=0.14)

        if len(vals):
            xmin, xmax = ax.get_xlim()
            extra = max((xmax - xmin) * 0.14, 0.001)
            ax.set_xlim(xmin, xmax + extra)

        for j, value in enumerate(vals):
            ax.text(value, j, f" {format_metric_value(value, metric)}", va="center", ha="left", fontsize=7.5)

    ax_tbl = fig.add_subplot(gs[1, :])
    ax_tbl.axis("off")
    rows: list[list[str]] = []
    for regime in REGIME_ORDER:
        part = top_df.loc[top_df["regime"] == regime].copy()
        for idx, (rank, strategy, value) in enumerate(
            part[["rank", "strategy", metric]].itertuples(index=False, name=None)
        ):
            overall_vol = part.iloc[idx]["OverallAnnualizedVol"] if "OverallAnnualizedVol" in part.columns else np.nan
            regime_vol = part.iloc[idx]["AnnualizedVol"] if "AnnualizedVol" in part.columns else np.nan
            row = [
                regime if idx == 0 else "",
                int(rank),
                compact_name(strategy, max_len=60),
                format_metric_value(value, metric),
            ]
            if metric in TABLE_SHOW_OVERALL_VOL_METRICS:
                row.append(f"{overall_vol:.2%}" if pd.notna(overall_vol) else "")
                row.append(f"{regime_vol:.2%}" if pd.notna(regime_vol) else "")
            rows.append(row)
    columns = ["Regime", "Rank", "Strategy", METRIC_LABELS[metric]]
    if metric in TABLE_SHOW_OVERALL_VOL_METRICS:
        columns.append("Overall Annualized Vol")
        columns.append("Regime Annualized Vol")
    table_df = pd.DataFrame(rows, columns=columns)
    table = ax_tbl.table(
        cellText=table_df.values,
        colLabels=table_df.columns,
        loc="center",
        cellLoc="left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.02, 1.75)

    fig.suptitle(f"Top Strategies by {METRIC_LABELS[metric]} Across Regimes", fontsize=16, y=0.98)
    fig.savefig(output_path, dpi=260, bbox_inches="tight")
    plt.close(fig)


def build_estimator_table_display(df: pd.DataFrame) -> pd.DataFrame:
    table = df.copy()
    for col in ESTIMATOR_METRICS:
        table[col] = table[col].map(lambda x: f"{x:.4f}" if pd.notna(x) else "")
    keep = ["rank_by_QLIKE", "estimator", "QLIKE", "OOS_R2", "MSE", "MincerAlpha", "MincerBeta", "MincerR2"]
    if "ForecastSourceStrategy" in table.columns:
        keep.append("ForecastSourceStrategy")
    return table[keep]


def choose_dynamic_precision_result(results_dir: Path) -> Path | None:
    preferred = [
        results_dir / "dynamic_precision_ensemble__trend_filter.parquet",
        results_dir / "dynamic_precision_ensemble__naive_scaling.parquet",
        results_dir / "dynamic_precision_ensemble__vol_target_clip.parquet",
    ]
    for path in preferred:
        if path.exists():
            return path

    candidates = sorted(results_dir.glob("dynamic_precision_ensemble__*.parquet"))
    for path in candidates:
        if "c_va_res_targeting" not in path.name:
            return path
    return candidates[0] if candidates else None


def load_intraday_proxy_series(path: Path) -> pd.Series:
    df = pd.read_parquet(path)
    if "realized_vol" not in df.columns:
        raise ValueError(f"Missing 'realized_vol' column in {path}")
    series = pd.Series(df["realized_vol"], index=pd.to_datetime(df.index)).astype(float)
    if getattr(series.index, "tz", None) is not None:
        series.index = series.index.tz_convert(None)
    series.index = pd.to_datetime(series.index).normalize()
    return series.dropna().sort_index()


def make_sticky_proxy_hmm(random_state: int = 42) -> GaussianHMM:
    stay_prob = 0.94
    transition_prob = (1.0 - stay_prob) / 2.0
    transmat = np.array(
        [
            [stay_prob, 1.0 - stay_prob, 0.0],
            [transition_prob, stay_prob, transition_prob],
            [0.0, 1.0 - stay_prob, stay_prob],
        ]
    )
    model = GaussianHMM(
        n_components=3,
        covariance_type="diag",
        n_iter=1000,
        random_state=random_state,
        transmat_prior=transmat * 1000.0 + 1.0,
        init_params="mc",
        params="stmc",
    )
    model.startprob_ = np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])
    model.transmat_ = transmat
    return model


def label_proxy_regimes_with_model(
    model: GaussianHMM,
    values: np.ndarray,
    index: pd.Index,
) -> pd.Series:
    states = model.predict(values)
    order = np.argsort(model.means_.flatten())
    mapping = {int(order[i]): REGIME_ORDER[i] for i in range(3)}
    return pd.Series(states, index=index).map(mapping)


def fit_proxy_regimes(proxy_vol: pd.Series) -> tuple[pd.Series, GaussianHMM | None]:
    log_proxy = np.log(proxy_vol.clip(lower=1e-8)).dropna().sort_index()
    values = log_proxy.to_numpy().reshape(-1, 1)
    min_obs = 252
    refit_every = 252
    regime = pd.Series(index=log_proxy.index, dtype=object)
    last_model: GaussianHMM | None = None

    if len(log_proxy) < min_obs:
        model = make_sticky_proxy_hmm()
        model.fit(values)
        regime.loc[log_proxy.index] = label_proxy_regimes_with_model(model, values, log_proxy.index)
        last_model = model
    else:
        for fit_end in range(min_obs, len(log_proxy), refit_every):
            segment_start = 0 if fit_end == min_obs else fit_end
            segment_end = min(fit_end + refit_every, len(log_proxy))
            model = make_sticky_proxy_hmm()
            model.fit(values[:fit_end])
            segment_values = values[segment_start:segment_end]
            segment_index = log_proxy.index[segment_start:segment_end]
            regime.loc[segment_index] = label_proxy_regimes_with_model(
                model,
                segment_values,
                segment_index,
            )
            last_model = model

        if regime.isna().any():
            fill_start = int(np.flatnonzero(regime.isna().to_numpy())[0])
            model = make_sticky_proxy_hmm()
            model.fit(values)
            segment_values = values[fill_start:]
            segment_index = log_proxy.index[fill_start:]
            regime.loc[segment_index] = label_proxy_regimes_with_model(
                model,
                segment_values,
                segment_index,
            )
            last_model = model

    regime = pd.Categorical(regime, categories=REGIME_ORDER, ordered=True)
    return pd.Series(regime, index=log_proxy.index, name="regime"), last_model


def shade_regimes(ax: plt.Axes, index: pd.DatetimeIndex, regime: pd.Series, alpha: float = 0.18) -> None:
    if len(index) == 0:
        return
    prev_reg = None
    block_start = index[0]
    for date, reg in zip(index, regime):
        if reg != prev_reg:
            if prev_reg is not None and prev_reg in REGIME_COLORS:
                ax.axvspan(block_start, date, color=REGIME_COLORS[prev_reg], alpha=alpha, linewidth=0, zorder=1)
            prev_reg = reg
            block_start = date
    if prev_reg is not None and prev_reg in REGIME_COLORS:
        ax.axvspan(block_start, index[-1], color=REGIME_COLORS[prev_reg], alpha=alpha, linewidth=0, zorder=1)


def draw_regime_strip(ax: plt.Axes, index: pd.DatetimeIndex, regime: pd.Series) -> None:
    if len(index) == 0:
        return
    prev_reg = None
    block_start = index[0]
    for date, reg in zip(index, regime):
        if reg != prev_reg:
            if prev_reg is not None and prev_reg in REGIME_COLORS:
                ax.axvspan(block_start, date, color=REGIME_COLORS[prev_reg], alpha=1.0, linewidth=0)
            prev_reg = reg
            block_start = date
    if prev_reg is not None and prev_reg in REGIME_COLORS:
        ax.axvspan(block_start, index[-1], color=REGIME_COLORS[prev_reg], alpha=1.0, linewidth=0)


def regime_legend_handles() -> list[mpatches.Patch]:
    return [
        mpatches.Patch(
            facecolor=REGIME_COLORS[r],
            alpha=0.55,
            edgecolor="grey",
            linewidth=0.4,
            label=REGIME_LABELS[r],
        )
        for r in REGIME_ORDER
    ]


def annotate_regime_pcts(ax: plt.Axes, regime: pd.Series) -> None:
    total = regime.dropna().shape[0]
    if total == 0:
        return
    lines = [f"{REGIME_LABELS[r]}: {(regime == r).sum() / total:.1%}" for r in REGIME_ORDER]
    ax.text(
        0.995,
        0.975,
        "\n".join(lines),
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8.5,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.88, edgecolor="grey"),
    )


def table_return_series(df: pd.DataFrame) -> pd.Series:
    for col in ["returns_with_rf", "returns", "returns_no_rf"]:
        if col in df.columns:
            return np.expm1(pd.Series(df[col], index=df.index).astype(float))
    return pd.Series(index=df.index, dtype=float)


def add_vol_proxy_regime_stats_table(fig: plt.Figure, merged: pd.DataFrame) -> None:
    rows: list[list[str]] = []
    for regime_name in REGIME_ORDER:
        mask = merged["regime"] == regime_name
        n = int(mask.sum())
        if n == 0:
            continue
        rets = merged.loc[mask, "strategy_simple_return"].fillna(0.0)
        ann_ret = float(rets.mean() * 252)
        ann_vol = float(rets.std() * np.sqrt(252))
        sharpe = ann_ret / ann_vol if ann_vol > 0 else np.nan
        avg_proxy = float(merged.loc[mask, "intraday_proxy_vol"].mean())
        avg_forecast = float(merged.loc[mask, "vol_estimate"].mean())
        rows.append([
            REGIME_LABELS[regime_name],
            f"{n:,}",
            f"{avg_proxy:.2%}",
            f"{avg_forecast:.2%}",
            f"{ann_ret:+.2%}",
            f"{ann_vol:.2%}",
            f"{sharpe:.2f}" if np.isfinite(sharpe) else "—",
        ])

    if not rows:
        return

    table_ax = fig.add_axes([0.06, 0.0, 0.90, 0.058])
    table_ax.axis("off")
    tbl = table_ax.table(
        cellText=rows,
        colLabels=["Regime", "Days", "Avg Proxy Vol", "Avg Forecast", "Ann. Return", "Ann. Vol", "Sharpe"],
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    tbl.scale(1, 1.45)

    for (row_i, col_i), cell in tbl.get_celld().items():
        if row_i == 0:
            cell.set_facecolor("#2c3e50")
            cell.set_text_props(color="white", fontweight="bold")
        elif col_i == 0 and row_i > 0:
            label_text = rows[row_i - 1][0]
            for regime_key, regime_label in REGIME_LABELS.items():
                if regime_label == label_text:
                    cell.set_facecolor(REGIME_COLORS[regime_key])
                    cell.set_text_props(color="white", fontweight="bold")
                    break
        else:
            cell.set_facecolor("#f5f6fa" if row_i % 2 == 0 else "white")


def plot_dynamic_precision_vs_intraday_proxy(output_path: Path) -> None:
    results_dir = Path("results")
    proxy_path = Path("data/processed/SP500_Intraday_RealizedVol.parquet")
    result_path = choose_dynamic_precision_result(results_dir)
    if result_path is None or not proxy_path.exists():
        return

    df = pd.read_parquet(result_path)
    df.index = pd.to_datetime(df.index)
    if getattr(df.index, "tz", None) is not None:
        df.index = df.index.tz_convert(None)
    df.index = df.index.normalize()
    df = df.sort_index()
    if "vol_estimate" not in df.columns:
        return

    proxy_vol = load_intraday_proxy_series(proxy_path)
    regime, _ = fit_proxy_regimes(proxy_vol)

    merged = df.join(proxy_vol.rename("intraday_proxy_vol"), how="inner")
    merged = merged.join(regime.rename("regime"), how="inner")
    merged = merged.dropna(subset=["vol_estimate", "intraday_proxy_vol", "regime"])
    if merged.empty:
        return

    merged["strategy_simple_return"] = table_return_series(merged)
    if "returns_with_rf" in merged.columns:
        strat_log = pd.Series(merged["returns_with_rf"], index=merged.index).astype(float).fillna(0.0)
    elif "returns" in merged.columns:
        strat_log = pd.Series(merged["returns"], index=merged.index).astype(float).fillna(0.0)
    else:
        strat_log = pd.Series(0.0, index=merged.index)
    merged["strategy_equity"] = 1000.0 * np.exp(strat_log.cumsum())

    if "asset_returns" in merged.columns:
        asset_log = pd.Series(merged["asset_returns"], index=merged.index).astype(float).fillna(0.0)
        merged["buy_hold_equity"] = 1000.0 * np.exp(asset_log.cumsum())
    else:
        merged["buy_hold_equity"] = np.nan

    plt.style.use("seaborn-v0_8-darkgrid")
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 1, figure=fig, height_ratios=[5, 0.7, 2.8], hspace=0.07)
    ax_vol = fig.add_subplot(gs[0])
    ax_strip = fig.add_subplot(gs[1], sharex=ax_vol)
    ax_eq = fig.add_subplot(gs[2], sharex=ax_vol)

    fig.suptitle(
        "Dynamic Precision Ensemble vs Intraday Vol Proxy\n"
        "Regimes defined by an annual-refit 3-state HMM fitted on log intraday vol",
        fontsize=13,
        fontweight="bold",
        y=0.99,
    )

    shade_regimes(ax_vol, merged.index, merged["regime"], alpha=0.20)
    ax_vol.plot(
        merged.index,
        merged["intraday_proxy_vol"],
        color="#7f8c8d",
        linewidth=1.1,
        alpha=0.8,
        label="Intraday realized-vol proxy",
        zorder=2,
    )
    ax_vol.plot(
        merged.index,
        merged["vol_estimate"],
        color="#2c3e50",
        linewidth=1.5,
        alpha=0.95,
        label="Dynamic Precision Ensemble forecast",
        zorder=3,
    )
    ax_vol.axhline(0.10, color="#3498db", linestyle="--", linewidth=1.2, alpha=0.7, label="Target Vol = 10%")
    ax_vol.set_ylabel("Annualized Volatility", fontsize=11)
    ax_vol.set_ylim(bottom=0)
    ax_vol.tick_params(labelbottom=False)
    ax_vol.legend(handles=regime_legend_handles() + ax_vol.get_legend_handles_labels()[0], loc="upper left", fontsize=9, framealpha=0.88)
    annotate_regime_pcts(ax_vol, merged["regime"])

    draw_regime_strip(ax_strip, merged.index, merged["regime"])
    ax_strip.set_yticks([0.5])
    ax_strip.set_yticklabels(["Regime"], fontsize=8)
    ax_strip.tick_params(labelbottom=False, left=False)
    ax_strip.set_ylim(0, 1)

    shade_regimes(ax_eq, merged.index, merged["regime"], alpha=0.20)
    ax_eq.plot(merged.index, merged["strategy_equity"], color="#2c3e50", linewidth=1.5, zorder=3, label="Strategy Equity")
    if merged["buy_hold_equity"].notna().any():
        ax_eq.plot(
            merged.index,
            merged["buy_hold_equity"],
            color="#7f8c8d",
            linewidth=1.0,
            alpha=0.65,
            linestyle="--",
            zorder=2,
            label="Buy & Hold",
        )
    ax_eq.set_ylabel("Portfolio Value", fontsize=11)
    ax_eq.set_xlabel("Date", fontsize=11)
    ax_eq.legend(loc="upper left", fontsize=9, framealpha=0.88)
    ax_eq.tick_params(axis="x", rotation=20)

    add_vol_proxy_regime_stats_table(fig, merged)
    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_estimator_ranking(estimator_overall: pd.DataFrame, output_path: Path) -> None:
    plot_df = estimator_overall.copy()
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.9, 2.9], hspace=0.12)

    ax = fig.add_subplot(gs[0])
    colors = plt.cm.viridis(np.linspace(0.2, 0.85, len(plot_df)))
    ax.barh(plot_df["estimator"].astype(str).map(lambda x: wrap_text(x, width=18)), plot_df["QLIKE"], color=colors)
    ax.invert_yaxis()
    ax.set_title("Estimator Ranking Overall (sorted by QLIKE)", fontsize=15, pad=12)
    ax.set_xlabel("Weighted Average QLIKE")
    ax.set_ylabel("Estimator")
    ax.grid(axis="x", linestyle="--", alpha=0.25)
    ax.tick_params(axis="y", labelsize=9)

    for i, value in enumerate(plot_df["QLIKE"]):
        ax.text(value, i, f" {value:.3f}", va="center", ha="left", fontsize=9)

    ax_table = fig.add_subplot(gs[1])
    ax_table.axis("off")
    table_df = build_estimator_table_display(plot_df)
    table = ax_table.table(
        cellText=table_df.values,
        colLabels=table_df.columns,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9.5)
    table.scale(1.02, 1.9)

    fig.savefig(output_path, dpi=260, bbox_inches="tight")
    plt.close(fig)


def plot_estimator_ranking_by_regime(estimator_regime_df: pd.DataFrame, output_path: Path) -> None:
    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.9, 2.1], hspace=0.16, wspace=0.28)
    palette = ["#4C78A8", "#F58518", "#54A24B"]

    table_rows: list[list[str]] = []
    for i, regime in enumerate(REGIME_ORDER):
        ax = fig.add_subplot(gs[0, i])
        part = estimator_regime_df.loc[estimator_regime_df["regime"] == regime].copy()
        part = part.sort_values(["QLIKE", "estimator"], ascending=[True, True]).reset_index(drop=True)
        part["rank_by_QLIKE"] = np.arange(1, len(part) + 1)
        ax.barh(part["estimator"].astype(str).map(lambda x: wrap_text(x, width=16)), part["QLIKE"], color=palette[i])
        ax.invert_yaxis()
        ax.set_title(f"{regime} Regime")
        ax.set_xlabel("QLIKE")
        if i == 0:
            ax.set_ylabel("Estimator")
        ax.grid(axis="x", linestyle="--", alpha=0.25)
        ax.tick_params(axis="y", labelsize=8)
        for j, value in enumerate(part["QLIKE"]):
            ax.text(value, j, f" {value:.3f}", va="center", ha="left", fontsize=8)

        best = part.head(5)
        summary = "\n".join(
            f"{int(r)}. {e} | QLIKE {q:.3f} | OOSR2 {o:.3f} | MSE {m:.3f}"
            for r, e, q, o, m in best[
                ["rank_by_QLIKE", "estimator", "QLIKE", "OOS_R2", "MSE"]
            ].itertuples(index=False, name=None)
        )
        table_rows.append([regime, summary])

    ax_tbl = fig.add_subplot(gs[1, :])
    ax_tbl.axis("off")
    table_df = pd.DataFrame(table_rows, columns=["Regime", "Top 5 Estimators by QLIKE"])
    table = ax_tbl.table(
        cellText=table_df.values,
        colLabels=table_df.columns,
        loc="center",
        cellLoc="left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10.5)
    table.scale(1.08, 2.8)

    fig.suptitle("Estimator Ranking by Regime (sorted by QLIKE)", fontsize=16, y=0.98)
    fig.savefig(output_path, dpi=260, bbox_inches="tight")
    plt.close(fig)


def save_top_tables(
    strategy_overall: pd.DataFrame,
    metrics_df: pd.DataFrame,
    output_dir: Path,
    top_n: int,
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    overall_rankings: dict[str, pd.DataFrame] = {}
    regime_rankings: dict[str, pd.DataFrame] = {}
    for metric in STRATEGY_METRICS:
        overall_rankings[metric] = top_n_overall_for_metric(strategy_overall, metric, top_n)
        regime_rankings[metric] = top_n_by_regime_for_metric(metrics_df, metric, top_n)

        overall_rankings[metric].to_csv(
            output_dir / f"top_{top_n}_{sanitize_name(metric.lower())}_overall.csv",
            index=False,
        )
        regime_rankings[metric].to_csv(
            output_dir / f"top_{top_n}_{sanitize_name(metric.lower())}_by_regime.csv",
            index=False,
        )
    return overall_rankings, regime_rankings


def save_regime_only_tables(metrics_df: pd.DataFrame, output_dir: Path, top_n: int) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for metric, ascending in REGIME_ONLY_METRICS.items():
        if metric not in metrics_df.columns:
            continue
        ranked = top_n_by_regime_custom_metric(metrics_df, metric, top_n=top_n, ascending=ascending)
        ranked.to_csv(
            output_dir / f"top_{top_n}_{sanitize_name(metric.lower())}_by_regime.csv",
            index=False,
        )
        out[metric] = ranked
    return out


def top_n_closest_to_target_vol(strategy_overall: pd.DataFrame, target_vol: float, top_n: int) -> pd.DataFrame:
    if "AnnualizedVol" not in strategy_overall.columns:
        return pd.DataFrame()
    out = strategy_overall.loc[:, ["strategy", "estimator", "controller", "n_obs", "AnnualizedVol"]].copy()
    out = out.dropna(subset=["AnnualizedVol"])
    out["DistanceToTargetVol"] = (out["AnnualizedVol"] - float(target_vol)).abs()
    out = out.sort_values(["DistanceToTargetVol", "AnnualizedVol", "strategy"], ascending=[True, True, True]).head(top_n)
    out = out.reset_index(drop=True)
    out.insert(0, "rank", np.arange(1, len(out) + 1))
    return out


def plot_top10_closest_to_target_vol(top_df: pd.DataFrame, target_vol: float, output_path: Path) -> None:
    if top_df.empty:
        return

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 1, height_ratios=[2.2, 1.8], hspace=0.1)

    ax = fig.add_subplot(gs[0])
    y_labels = [
        wrap_text(f"{int(rank)}. {compact_name(strategy, max_len=34)}", width=22)
        for rank, strategy in top_df[["rank", "strategy"]].itertuples(index=False, name=None)
    ]
    vals = top_df["AnnualizedVol"].to_numpy(dtype=float)
    colors = plt.cm.PuBuGn(np.linspace(0.35, 0.9, len(top_df)))
    ax.barh(y_labels, vals, color=colors)
    ax.axvline(float(target_vol), color="#d62728", linestyle="--", linewidth=1.5, label=f"Target {target_vol:.0%}")
    ax.invert_yaxis()
    ax.set_title(f"Top {len(top_df)} Strategies Closest to {target_vol:.0%} Annualized Volatility", fontsize=16, pad=12)
    ax.set_xlabel("Annualized Volatility")
    ax.set_ylabel("Strategy")
    ax.grid(axis="x", linestyle="--", alpha=0.25)
    ax.legend(loc="lower right")

    xmin, xmax = ax.get_xlim()
    extra = max((xmax - xmin) * 0.12, 0.01)
    ax.set_xlim(xmin, xmax + extra)

    for i, value in enumerate(vals):
        ax.text(value, i, f" {value:.2%}", va="center", ha="left", fontsize=9)

    ax_tbl = fig.add_subplot(gs[1])
    ax_tbl.axis("off")
    table_df = top_df.copy()
    table_df["strategy"] = table_df["strategy"].map(lambda x: compact_name(x, max_len=38))
    table_df["controller"] = table_df["controller"].map(lambda x: compact_name(x, max_len=24))
    table_df["AnnualizedVol"] = table_df["AnnualizedVol"].map(lambda x: f"{x:.2%}")
    table_df["DistanceToTargetVol"] = table_df["DistanceToTargetVol"].map(lambda x: f"{x:.2%}")
    table_df = table_df[
        ["rank", "strategy", "estimator", "controller", "AnnualizedVol", "DistanceToTargetVol"]
    ]
    table = ax_tbl.table(
        cellText=table_df.values,
        colLabels=table_df.columns,
        loc="center",
        cellLoc="left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.7)

    fig.savefig(output_path, dpi=260, bbox_inches="tight")
    plt.close(fig)


def print_top_summary(overall_rankings: dict[str, pd.DataFrame]) -> None:
    for metric, top_df in overall_rankings.items():
        print(f"\nTop strategies overall by {METRIC_LABELS[metric]}")
        cols = ["rank", "strategy", "estimator", "controller", metric]
        print(top_df[cols].to_string(index=False))


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_df = load_input(input_path)
    metrics_df = exclude_estimators(metrics_df, args.exclude_estimators)
    metrics_df = exclude_controllers(metrics_df, args.exclude_controllers)
    strategy_overall = aggregate_strategy_overall(metrics_df)
    metrics_df = attach_overall_annualized_vol(metrics_df, strategy_overall)
    estimator_regime = collapse_estimator_metrics(metrics_df)
    estimator_overall = aggregate_estimator_overall(estimator_regime)

    estimator_by_regime = estimator_regime.sort_values(
        ["regime", "QLIKE", "estimator"], ascending=[True, True, True]
    ).reset_index(drop=True)
    estimator_by_regime["rank_by_QLIKE"] = (
        estimator_by_regime.groupby("regime", observed=True).cumcount() + 1
    )

    strategy_overall.to_csv(output_dir / "strategy_overall_weighted_summary.csv", index=False)
    estimator_overall.to_csv(output_dir / "estimator_ranking_overall.csv", index=False)
    estimator_by_regime.to_csv(output_dir / "estimator_ranking_by_regime.csv", index=False)

    overall_rankings, regime_rankings = save_top_tables(
        strategy_overall=strategy_overall,
        metrics_df=metrics_df,
        output_dir=output_dir,
        top_n=args.top_n,
    )
    regime_only_rankings = save_regime_only_tables(
        metrics_df=metrics_df,
        output_dir=output_dir,
        top_n=args.top_n,
    )

    closest_target_vol = top_n_closest_to_target_vol(strategy_overall, target_vol=0.10, top_n=10)
    if not closest_target_vol.empty:
        closest_target_vol.to_csv(output_dir / "top_10_closest_to_10pct_annualized_vol.csv", index=False)

    for metric in STRATEGY_METRICS:
        plot_top5_metric_overall(
            overall_rankings[metric],
            metric,
            output_dir / f"Top_{args.top_n}_{sanitize_name(metric)}_Overall.png",
        )
        plot_top5_metric_by_regime(
            regime_rankings[metric],
            metric,
            output_dir / f"Top_{args.top_n}_{sanitize_name(metric)}_By_Regime.png",
        )

    for metric, ranked in regime_only_rankings.items():
        plot_top5_metric_by_regime(
            ranked,
            metric,
            output_dir / f"Top_{args.top_n}_{sanitize_name(metric)}_By_Regime.png",
        )

    plot_estimator_ranking(estimator_overall, output_dir / "Estimator_Ranking_Overall.png")
    plot_estimator_ranking_by_regime(estimator_by_regime, output_dir / "Estimator_Ranking_By_Regime.png")
    plot_top10_closest_to_target_vol(
        closest_target_vol,
        target_vol=0.10,
        output_path=output_dir / "Top_10_Closest_To_10pct_AnnualizedVol_Overall.png",
    )
    plot_dynamic_precision_vs_intraday_proxy(
        output_dir / "DynamicPrecisionEnsemble_Vs_IntradayProxy_RegimeTS.png"
    )

    print(f"Saved picture summaries to {output_dir.resolve()}")
    if args.exclude_estimators:
        print(f"Excluded estimators: {', '.join(args.exclude_estimators)}")
    if args.exclude_controllers:
        print(f"Excluded controllers: {', '.join(args.exclude_controllers)}")
    print_top_summary(overall_rankings)
    print("\nEstimator ranking overall (sorted by QLIKE)")
    show_cols = ["rank_by_QLIKE", "estimator", "QLIKE", "OOS_R2", "MSE", "MincerAlpha", "MincerBeta", "MincerR2"]
    print(estimator_overall.loc[:, show_cols].to_string(index=False))


if __name__ == "__main__":
    main()
