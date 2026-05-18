# JPM-Volatility-Targeting

.
├── configs/                # Strategy configuration files
│   └── strategies/         # Model-specific YAML configs (GARCH, EWMA, AR1, etc.)
├── src/                    # Core source code
│   ├── estimators/         # Volatility forecasting models (Realized Vol, EWMA, EGARCH, GARCH, etc.)
│   ├── controllers/        # Exposure control & leverage logic (Naive Scaling, Constant Weight)
│   ├── backtest/           # Backtesting engine core (engine.py, base classes)
│   ├── data/               # Data ingestion & ETL pipeline (importers, processors)
│   └── evaluation/         # Performance metrics & diagnostic tools
├── scripts/                # Execution & CLI scripts
│   ├── run_backtests.py    # Batch execution of backtesting suites
│   ├── analysus_result.py  # Automated 4-dimensional diagnostic report generation
│   ├── run_processors.py   # Execution of data cleaning & processing pipelines
│   └── tune_strategy.py    # Hyperparameter optimization & strategy tuning
├── data/                   # Data storage for Raw and Processed (Parquet) datasets
└── Literature-Review/      # Academic papers on Volatility-Managed Portfolios (Moreira, Harvey, etc.)



