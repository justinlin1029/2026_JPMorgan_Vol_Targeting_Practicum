"""Export estimator-controller metrics by HMM volatility regime to CSV."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from hmmlearn.hmm import GaussianHMM
except ImportError as exc:  # pragma: no cover - user environment dependency
    raise SystemExit(
        "Missing dependency 'hmmlearn'. Install it first, for example with:\n"
        "  pip install hmmlearn"
    ) from exc

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.env import Env


REGIME_ORDER = ["Low", "Mid", "High"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export estimator/controller metrics by HMM benchmark regime."
    )
    parser.add_argument(
        "--results-dir",
        default=str(Env.path("results")),
        help="Directory containing backtest result parquet files.",
    )
    parser.add_argument(
        "--output",
        default=str(Env.path("results") / "estimator_controller_metrics_by_regime.csv"),
        help="Output CSV path.",
    )
    parser.add_argument(
        "--ann-factor",
        type=float,
        default=252.0,
        help="Annualization factor for return/vol metrics.",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=0,
        help="Forecast horizon to pass through to alignment helper.",
    )
    parser.add_argument(
        "--min-coverage-ratio",
        type=float,
        default=0.95,
        help="Minimum intraday RV coverage ratio for the realized-vol benchmark.",
    )
    return parser.parse_args()


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

    rv = rv_df[vol_col].astype(float).copy()
    rv.name = vol_col
    return rv


def realized_vol_proxy(
    result_index: pd.Index,
    parquet_name: str = "SP500_Intraday_RealizedVol.parquet",
    vol_col: str = "realized_vol",
    min_coverage_ratio: float | None = 0.95,
) -> pd.Series:
    rv_series = load_intraday_realized_vol(
        parquet_name=parquet_name,
        vol_col=vol_col,
        min_coverage_ratio=min_coverage_ratio,
    )
    idx = pd.to_datetime(result_index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    idx = idx.normalize()

    out = rv_series.reindex(idx)
    out.index = idx
    return out.astype(float)


def compute_hmm_regime_from_series(
    vol_series: pd.Series,
    n_components: int = 3,
    random_state: int = 42,
) -> tuple[pd.Series, pd.Series, GaussianHMM | None]:
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
        ordered_labels = np.array(REGIME_ORDER)
        state_to_label = {int(order[k]): ordered_labels[k] for k in range(n_components)}

        regime_state.loc[mask] = states
        regime_label.loc[mask] = pd.Series(states, index=vol.loc[mask].index).map(state_to_label).values
        return regime_state, regime_label, model
    except Exception as exc:
        print(f"⚠️ HMM regime labeling failed: {exc}")
        return regime_state, regime_label, None


def build_realized_vol_benchmark_regime(
    index: pd.Index,
    min_coverage_ratio: float = 0.95,
) -> pd.DataFrame:
    rv = realized_vol_proxy(
        result_index=index,
        parquet_name="SP500_Intraday_RealizedVol.parquet",
        vol_col="realized_vol",
        min_coverage_ratio=min_coverage_ratio,
    )
    regime_state, regime_label, _ = compute_hmm_regime_from_series(rv)

    out = pd.DataFrame(index=pd.to_datetime(index))
    out["benchmark_rv"] = rv.reindex(out.index)
    out["benchmark_regime_state"] = regime_state.reindex(out.index)
    out["benchmark_regime"] = regime_label.reindex(out.index)
    return out


def qlike_series(actual: pd.Series, forecast: pd.Series) -> pd.Series:
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


def align_forecast_and_realized(
    df: pd.DataFrame,
    horizon: int = 0,
    min_coverage_ratio: float = 0.95,
) -> pd.DataFrame | None:
    if "vol_estimate" not in df.columns:
        return None

    rv = realized_vol_proxy(
        result_index=df.index,
        parquet_name="SP500_Intraday_RealizedVol.parquet",
        vol_col="realized_vol",
        min_coverage_ratio=min_coverage_ratio,
    )
    vh = pd.Series(df["vol_estimate"], index=df.index).astype(float)
    target = rv.shift(-horizon) if horizon > 0 else rv

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
    return m


def annualized_vol_from_returns(ret: pd.Series, ann_factor: float = 252.0) -> float:
    r = pd.Series(ret).astype(float).dropna()
    if len(r) < 2:
        return float("nan")
    sigma = r.std(ddof=1)
    if not np.isfinite(sigma):
        return float("nan")
    return float(sigma * np.sqrt(ann_factor))


def annualized_return_from_returns(ret: pd.Series, ann_factor: float = 252.0) -> float:
    r = pd.Series(ret).astype(float).dropna()
    if r.empty:
        return float("nan")
    wealth = (1.0 + r).prod()
    n = len(r)
    if n == 0 or wealth <= 0:
        return float("nan")
    return float(wealth ** (ann_factor / n) - 1.0)


def sharpe_ratio_from_returns(ret: pd.Series, ann_factor: float = 252.0) -> float:
    r = pd.Series(ret).astype(float).dropna()
    if len(r) < 2:
        return float("nan")
    mu = r.mean()
    sigma = r.std(ddof=1)
    if not np.isfinite(sigma) or sigma <= 0:
        return float("nan")
    return float(np.sqrt(ann_factor) * mu / sigma)


def max_drawdown_from_returns(ret: pd.Series) -> float:
    r = pd.Series(ret).astype(float).dropna()
    if r.empty:
        return float("nan")
    wealth = (1.0 + r).cumprod()
    peak = wealth.cummax()
    dd = wealth / peak - 1.0
    return float(dd.min())


def drawdown_duration_from_returns(ret: pd.Series) -> float:
    """
    Maximum drawdown duration in trading days on the cumulative wealth path.
    """
    r = pd.Series(ret).astype(float).dropna()
    if r.empty:
        return float("nan")

    wealth = (1.0 + r).cumprod()
    peak = wealth.cummax()
    underwater = (wealth / peak) < (1.0 - 1e-12)

    max_duration = 0
    current = 0
    for flag in underwater.to_numpy(dtype=bool):
        if flag:
            current += 1
            if current > max_duration:
                max_duration = current
        else:
            current = 0
    return float(max_duration)


def drawdown_duration_by_regime_blocks(ret: pd.Series, regime_labels: pd.Series, regime: str) -> float:
    """
    Compute max drawdown duration using contiguous blocks of a given regime
    on the original calendar path, instead of stitching all same-regime days
    together across time.
    """
    r = pd.Series(ret).astype(float)
    reg = pd.Series(regime_labels).reindex(r.index)
    if r.dropna().empty or reg.dropna().empty:
        return float("nan")

    mask = reg.eq(regime)
    if not mask.any():
        return float("nan")

    block_ids = mask.ne(mask.shift(fill_value=False)).cumsum()
    durations: list[float] = []
    for _, block in r[mask].groupby(block_ids[mask]):
        block = block.dropna()
        if block.empty:
            continue
        durations.append(drawdown_duration_from_returns(block))

    if not durations:
        return float("nan")
    return float(max(durations))


def strategy_return_series(df: pd.DataFrame) -> pd.Series:
    """
    Backtest result files store log returns, so convert to simple returns.
    """
    for col in ["returns_with_rf", "strategy_returns", "returns", "returns_no_rf"]:
        if col in df.columns:
            log_ret = pd.Series(df[col], index=df.index).astype(float)
            return np.expm1(log_ret)
    return pd.Series(index=df.index, dtype=float)


def turnover_series(df: pd.DataFrame) -> pd.Series:
    if "weight" not in df.columns:
        return pd.Series(index=df.index, dtype=float)
    w = pd.Series(df["weight"], index=df.index).astype(float)
    return w.diff().abs()


def monthly_average_drawdown_from_returns(ret: pd.Series) -> float:
    r = pd.Series(ret).astype(float).dropna()
    if r.empty:
        return float("nan")

    monthly_mdds: list[float] = []
    for _, block in r.groupby(r.index.to_period("M")):
        if block.empty:
            continue
        monthly_mdds.append(max_drawdown_from_returns(block))

    if not monthly_mdds:
        return float("nan")
    return float(np.mean(monthly_mdds))


def win_rate_from_returns(ret: pd.Series) -> float:
    r = pd.Series(ret).astype(float).dropna()
    if r.empty:
        return float("nan")
    return float((r > 0).mean())


def cvar_from_returns(ret: pd.Series, alpha: float = 0.95) -> float:
    """
    Historical CVaR / Expected Shortfall returned as a positive loss number.
    At alpha=0.95, this is the average of the worst 5% daily returns.
    """
    r = pd.Series(ret).astype(float).dropna()
    if r.empty:
        return float("nan")

    cutoff = float(np.percentile(r, (1.0 - alpha) * 100.0))
    tail = r[r <= cutoff]
    if tail.empty:
        return float("nan")
    return float(-tail.mean())


def mse_from_forecast(actual_vol: pd.Series, forecast_vol: pd.Series) -> float:
    m = pd.concat(
        [
            pd.Series(actual_vol).astype(float).rename("actual"),
            pd.Series(forecast_vol).astype(float).rename("forecast"),
        ],
        axis=1,
    ).dropna()
    if m.empty:
        return float("nan")
    return float(np.mean((m["forecast"] - m["actual"]) ** 2))


def oos_r2_from_forecast(actual_vol: pd.Series, forecast_vol: pd.Series) -> float:
    m = pd.concat(
        [
            pd.Series(actual_vol).astype(float).rename("actual"),
            pd.Series(forecast_vol).astype(float).rename("forecast"),
        ],
        axis=1,
    ).dropna()
    if len(m) < 2:
        return float("nan")

    sse_model = float(np.sum((m["forecast"] - m["actual"]) ** 2))
    mean_benchmark = float(m["actual"].mean())
    sse_bench = float(np.sum((m["actual"] - mean_benchmark) ** 2))
    if not np.isfinite(sse_bench) or sse_bench <= 0:
        return float("nan")
    return float(1.0 - sse_model / sse_bench)


def mincer_regression_stats(actual_vol: pd.Series, forecast_vol: pd.Series) -> tuple[float, float, float]:
    m = pd.concat(
        [
            pd.Series(actual_vol).astype(float).rename("actual"),
            pd.Series(forecast_vol).astype(float).rename("forecast"),
        ],
        axis=1,
    ).dropna()
    if len(m) < 3:
        return float("nan"), float("nan"), float("nan")

    x = m["forecast"].to_numpy(dtype=float)
    y = m["actual"].to_numpy(dtype=float)
    X = np.column_stack([np.ones(len(m)), x])

    try:
        coef, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        y_hat = X @ coef
        ss_res = float(np.sum((y - y_hat) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
        return float(coef[0]), float(coef[1]), r2
    except Exception:
        return float("nan"), float("nan"), float("nan")


def split_strategy_name(name: str) -> tuple[str, str]:
    if "__" in name:
        estimator, controller = name.split("__", 1)
        return estimator, controller
    return name, ""


def load_result_files(results_dir: Path) -> dict[str, pd.DataFrame]:
    data_map: dict[str, pd.DataFrame] = {}
    for path in sorted(results_dir.glob("*.parquet")):
        if "__" not in path.stem:
            continue
        try:
            df = pd.read_parquet(path)
            df.index = pd.to_datetime(df.index)
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            df.index = df.index.normalize()
            df = df.sort_index()
            data_map[path.stem] = df
        except Exception as exc:
            print(f"⚠️ Skip {path.name}: {exc}")
    return data_map


def estimate_forecast_signature(df: pd.DataFrame) -> tuple:
    """
    Compact fingerprint of the forecast series so we can spot mixed reruns.
    """
    if "vol_estimate" not in df.columns:
        return tuple()
    s = pd.Series(df["vol_estimate"], index=df.index).astype(float)
    s = s.dropna()
    if s.empty:
        return tuple()
    head_vals = tuple(np.round(s.head(5).to_numpy(dtype=float), 12))
    tail_vals = tuple(np.round(s.tail(5).to_numpy(dtype=float), 12))
    return (len(s), head_vals, tail_vals)


def evaluate_strategy(
    strategy_name: str,
    df: pd.DataFrame,
    benchmark_regime_df: pd.DataFrame,
    ann_factor: float,
    horizon: int,
    min_coverage_ratio: float,
) -> pd.DataFrame:
    aligned = align_forecast_and_realized(
        df=df,
        horizon=horizon,
        min_coverage_ratio=min_coverage_ratio,
    )
    if aligned is None or aligned.empty:
        return pd.DataFrame()

    tmp = aligned.copy()
    tmp["benchmark_regime"] = benchmark_regime_df["benchmark_regime"].reindex(tmp.index)
    tmp["strategy_returns"] = strategy_return_series(df).reindex(tmp.index)
    tmp["turnover"] = turnover_series(df).reindex(tmp.index)
    tmp["qlike"] = qlike_series(tmp["target_rv"], tmp["vh"]).reindex(tmp.index)
    tmp = tmp.dropna(subset=["benchmark_regime"])
    if tmp.empty:
        return pd.DataFrame()

    estimator, controller = split_strategy_name(strategy_name)
    rows: list[dict] = []

    for regime in REGIME_ORDER:
        g = tmp[tmp["benchmark_regime"] == regime].copy()
        if g.empty:
            continue

        strategy_ret = g["strategy_returns"].dropna()
        if strategy_ret.empty:
            continue

        mincer_alpha, mincer_beta, mincer_r2 = mincer_regression_stats(g["target_rv"], g["vh"])
        rows.append(
            {
                "strategy": strategy_name,
                "estimator": estimator,
                "controller": controller,
                "regime": regime,
                "n_obs": int(len(g)),
                "QLIKE": float(g["qlike"].mean()) if g["qlike"].notna().any() else float("nan"),
                "OOS_R2": oos_r2_from_forecast(g["target_rv"], g["vh"]),
                "MSE": mse_from_forecast(g["target_rv"], g["vh"]),
                "MincerAlpha": mincer_alpha,
                "MincerBeta": mincer_beta,
                "MincerR2": mincer_r2,
                "Sharpe": sharpe_ratio_from_returns(strategy_ret, ann_factor=ann_factor),
                "MaxDrawdown": max_drawdown_from_returns(strategy_ret),
                "DrawdownDuration": drawdown_duration_by_regime_blocks(
                    ret=tmp["strategy_returns"],
                    regime_labels=tmp["benchmark_regime"],
                    regime=regime,
                ),
                "MonthlyAverageDrawdown": monthly_average_drawdown_from_returns(strategy_ret),
                "Turnover": float(g["turnover"].mean()) if g["turnover"].notna().any() else float("nan"),
                "CVaR": cvar_from_returns(strategy_ret),
                "AnnualizedVol": annualized_vol_from_returns(strategy_ret, ann_factor=ann_factor),
                "AnnualizedReturn": annualized_return_from_returns(strategy_ret, ann_factor=ann_factor),
                "WinRate": win_rate_from_returns(strategy_ret),
            }
        )

    return pd.DataFrame(rows)


def compute_estimator_forecast_metrics(
    estimator: str,
    representative_name: str,
    representative_df: pd.DataFrame,
    benchmark_regime_df: pd.DataFrame,
    horizon: int,
    min_coverage_ratio: float,
) -> pd.DataFrame:
    aligned = align_forecast_and_realized(
        df=representative_df,
        horizon=horizon,
        min_coverage_ratio=min_coverage_ratio,
    )
    if aligned is None or aligned.empty:
        return pd.DataFrame()

    tmp = aligned.copy()
    tmp["benchmark_regime"] = benchmark_regime_df["benchmark_regime"].reindex(tmp.index)
    tmp["qlike"] = qlike_series(tmp["target_rv"], tmp["vh"]).reindex(tmp.index)
    tmp = tmp.dropna(subset=["benchmark_regime"])
    if tmp.empty:
        return pd.DataFrame()

    rows: list[dict] = []
    for regime in REGIME_ORDER:
        g = tmp[tmp["benchmark_regime"] == regime].copy()
        if g.empty:
            continue
        mincer_alpha, mincer_beta, mincer_r2 = mincer_regression_stats(g["target_rv"], g["vh"])
        rows.append(
            {
                "estimator": estimator,
                "regime": regime,
                "ForecastSourceStrategy": representative_name,
                "QLIKE": float(g["qlike"].mean()) if g["qlike"].notna().any() else float("nan"),
                "OOS_R2": oos_r2_from_forecast(g["target_rv"], g["vh"]),
                "MSE": mse_from_forecast(g["target_rv"], g["vh"]),
                "MincerAlpha": mincer_alpha,
                "MincerBeta": mincer_beta,
                "MincerR2": mincer_r2,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data_map = load_result_files(results_dir)
    if not data_map:
        raise FileNotFoundError(f"No result parquet files found in {results_dir}")

    first_name = next(iter(data_map))
    benchmark_regime_df = build_realized_vol_benchmark_regime(
        index=data_map[first_name].index,
        min_coverage_ratio=args.min_coverage_ratio,
    )
    if benchmark_regime_df.empty:
        raise ValueError("Failed to build benchmark HMM regime series.")

    estimator_representatives: dict[str, tuple[str, pd.DataFrame]] = {}
    estimator_signatures: dict[str, tuple] = {}
    for strategy_name, df in data_map.items():
        estimator, _ = split_strategy_name(strategy_name)
        sig = estimate_forecast_signature(df)
        if estimator not in estimator_representatives:
            estimator_representatives[estimator] = (strategy_name, df)
            estimator_signatures[estimator] = sig
        elif sig and estimator_signatures.get(estimator) and sig != estimator_signatures[estimator]:
            print(
                f"⚠️ Estimator {estimator} has non-identical vol_estimate series across controllers. "
                f"Using {estimator_representatives[estimator][0]} as the forecast-metric source."
            )

    forecast_frames: list[pd.DataFrame] = []
    for estimator, (strategy_name, df) in estimator_representatives.items():
        out = compute_estimator_forecast_metrics(
            estimator=estimator,
            representative_name=strategy_name,
            representative_df=df,
            benchmark_regime_df=benchmark_regime_df,
            horizon=args.horizon,
            min_coverage_ratio=args.min_coverage_ratio,
        )
        if not out.empty:
            forecast_frames.append(out)

    if not forecast_frames:
        raise ValueError("No estimator-level forecast metrics were produced from the available result files.")
    forecast_summary = pd.concat(forecast_frames, ignore_index=True)

    frames: list[pd.DataFrame] = []
    for strategy_name, df in data_map.items():
        out = evaluate_strategy(
            strategy_name=strategy_name,
            df=df,
            benchmark_regime_df=benchmark_regime_df,
            ann_factor=args.ann_factor,
            horizon=args.horizon,
            min_coverage_ratio=args.min_coverage_ratio,
        )
        if not out.empty:
            frames.append(out)

    if not frames:
        raise ValueError("No metrics were produced from the available result files.")

    summary = pd.concat(frames, ignore_index=True)
    summary = summary.drop(
        columns=["QLIKE", "OOS_R2", "MSE", "MincerAlpha", "MincerBeta", "MincerR2"],
        errors="ignore",
    )
    summary = summary.merge(
        forecast_summary,
        on=["estimator", "regime"],
        how="left",
    )
    summary["regime"] = pd.Categorical(summary["regime"], categories=REGIME_ORDER, ordered=True)
    summary = summary.sort_values(["estimator", "controller", "regime"]).reset_index(drop=True)
    summary.to_csv(output_path, index=False)

    print(f"Saved regime metrics CSV to: {output_path}")
    print(f"Rows: {len(summary)}")


if __name__ == "__main__":
    main()
