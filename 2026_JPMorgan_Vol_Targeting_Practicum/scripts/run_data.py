from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on sys.path (so "import src..." works)
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.importers.strategy_data_builder import DataBentoImporter, YahooVIXImporter
# (adjust import path if your importer file name differs)


def main():
    print("=" * 60)
    print("🚀 Data Build: raw CSV -> processed parquet")
    print(f"Project root: {ROOT}")
    print("=" * 60)

    # ES
    importer = DataBentoImporter()
    out_es = importer.save_processed_parquet(
        out_name="ES_Daily_Processed.parquet",
        use_log_return=True,
        winsor_q=0.01,
    )

    # VIX
    vix_importer = YahooVIXImporter()
    out_vix = vix_importer.save_processed_parquet(
        out_name="VIX_Daily_Processed.parquet",
    )

    print("\n✅ Done.")


if __name__ == "__main__":
    main()