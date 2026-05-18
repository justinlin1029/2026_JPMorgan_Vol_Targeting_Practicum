import sys
from pathlib import Path

root_path = Path(__file__).resolve().parents[1]
if str(root_path) not in sys.path:
    sys.path.insert(0, str(root_path))

from src.data.importers.strategy_data_builder import DataBentoImporter
from src.env import Env


def run_mirror_import():
    print("From CSV >> Parquet")

    importer = DataBentoImporter()
    source = "databento"

    out_dir = Env.path("prices", source)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Original mirror import
        df = importer.fetch_all_as_mirror()

        save_path = out_dir / "prices.parquet"
        df.to_parquet(save_path)

        print(f"\n✅ Successfully imported {save_path}")
        print(f"Scale: {len(df)} row")
        print(f"Head: {list(df.columns)}")

        # Minute returns from S&P500_Future_Price.csv
        minute_df = importer.save_minute_return_parquet(
            raw_file_name="S&P500_Future_Price.csv",
            out_name="SP500_Futures_Minute_Processed.parquet",
            use_log_return=True,
        )

        print("\n✅ Minute return parquet created successfully.")
        print(f"Minute dataset rows: {len(minute_df)}")

        # Daily realized volatility from minute returns
        rv_df = importer.save_intraday_realized_vol_parquet(
            raw_file_name="S&P500_Future_Price.csv",
            out_name="SP500_Intraday_RealizedVol.parquet",
            use_log_return=True,
            annualize=True,
            trading_days=252,
            min_coverage_ratio=None,   # set e.g. 0.95 if you want to filter incomplete days
        )

        print("\n✅ Intraday realized volatility parquet created successfully.")
        print(f"Realized vol dataset rows: {len(rv_df)}")
        print(f"Realized vol columns: {list(rv_df.columns)}")

        if "coverage" in rv_df.columns:
            print("\nCoverage summary:")
            print(rv_df["coverage"].describe())

    except Exception as e:
        print(f"\n❌ Fail to transfer from CSV to Parquet: {e}")


if __name__ == "__main__":
    run_mirror_import()