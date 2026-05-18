"""Run the rolling 6-year regime-adaptive combination engine."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from src.backtest.regime_adaptive_mix import RollingRegimeMixEngine


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build three combined strategies by re-ranking estimator/controller "
            "pairs over a rolling training window and selecting the winner by "
            "intraday-vol regime."
        )
    )
    parser.add_argument(
        "--name",
        default="rolling_regime_mix",
        help="Base name used for parquet/csv outputs in results/.",
    )
    parser.add_argument(
        "--train-years",
        type=int,
        default=6,
        help="Rolling training window in calendar years.",
    )
    parser.add_argument(
        "--results-dir",
        default=None,
        help="Optional directory containing estimator/controller parquet results.",
    )
    parser.add_argument(
        "--min-coverage-ratio",
        type=float,
        default=0.95,
        help="Minimum intraday proxy coverage required.",
    )
    parser.add_argument(
        "--initial-capital",
        type=float,
        default=1000.0,
        help="Initial capital used for the combined equity curves.",
    )
    parser.add_argument(
        "--exclude-estimators",
        nargs="*",
        default=[],
        help="Estimator names to remove from the candidate pool, e.g. buy_and_hold xgb_vix ar1.",
    )
    parser.add_argument(
        "--exclude-controllers",
        nargs="*",
        default=[],
        help="Controller names to remove from the candidate pool, e.g. c_va_res_targeting.",
    )
    parser.add_argument(
        "--regime-method",
        default="proxy_quantile",
        choices=["proxy_quantile", "hmm"],
        help="How to classify regimes from the intraday vol proxy.",
    )
    parser.add_argument(
        "--hmm-input",
        default="intraday_vol",
        choices=["intraday_vol", "daily_returns"],
        help="Observation series for HMM regime prediction.",
    )
    parser.add_argument(
        "--sharpe-turnover-penalty",
        type=float,
        default=0.5,
        help="Turnover penalty used in Sharpe-based regime selection.",
    )
    parser.add_argument(
        "--maxdd-turnover-penalty",
        type=float,
        default=0.5,
        help="Turnover penalty used in MaxDrawdown-based regime selection.",
    )
    parser.add_argument(
        "--sharpe-turnover-cost",
        type=float,
        default=0.5,
        help="Turnover cost used in Sharpe-based regime selection.",
    )
    parser.add_argument(
        "--maxdd-turnover-cost",
        type=float,
        default=0.04,
        help="Turnover cost used in MaxDrawdown-based regime selection.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = {
        "train_years": args.train_years,
        "results_dir": args.results_dir,
        "min_coverage_ratio": args.min_coverage_ratio,
        "initial_capital": args.initial_capital,
        "exclude_estimators": args.exclude_estimators,
        "exclude_controllers": args.exclude_controllers,
        "regime_method": args.regime_method,
        "hmm_input": args.hmm_input,
        "sharpe_turnover_penalty": args.sharpe_turnover_penalty,
        "maxdd_turnover_penalty": args.maxdd_turnover_penalty,
        "sharpe_turnover_cost": args.sharpe_turnover_cost,
        "maxdd_turnover_cost": args.maxdd_turnover_cost,
    }

    engine = RollingRegimeMixEngine(name=args.name, config=cfg)
    engine.run()
    paths = engine.save()
    engine.summary()

    print(f"\nSaved detail parquet to: {paths['result_path']}")
    print(f"Saved combined summary to: {paths['summary_path']}")
    print(f"Saved initial 9 winners to: {paths['initial_winners_path']}")
    print(f"Saved rolling selection history to: {paths['selection_history_path']}")


if __name__ == "__main__":
    main()
