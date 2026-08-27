from pathlib import Path

import pandas as pd

from python.data_engineering.run_pipeline import is_materialized
from python.modeling.monte_carlo_inventory import product_seed
from python.modeling.register_models import stable_version
from python.modeling.monitor_models import severity


def test_model_version_is_deterministic_and_lineage_sensitive():
    first = stable_version("P00001", "HoltWinters7", "data-a", "code-a")
    assert first == stable_version("P00001", "HoltWinters7", "data-a", "code-a")
    assert first != stable_version("P00001", "HoltWinters7", "data-b", "code-a")
    assert first != stable_version("P00001", "Naive", "data-a", "code-a")


def test_monte_carlo_product_seed_is_reproducible():
    assert product_seed("P00001") == product_seed("P00001")
    assert product_seed("P00001") != product_seed("P00002")


def test_monitoring_severity_contract():
    result = severity(pd.Series([0.2, 0.8, 1.6])).tolist()
    assert result == ["Stable", "Warning", "Critical"]


def test_lfs_pointer_is_not_a_materialized_dataset(tmp_path: Path):
    pointer = tmp_path / "data.csv"
    pointer.write_text(
        "version https://git-lfs.github.com/spec/v1\n"
        "oid sha256:abc\nsize 100\n",
        encoding="utf-8",
    )
    assert not is_materialized(pointer)


def test_regular_csv_is_materialized(tmp_path: Path):
    csv_file = tmp_path / "data.csv"
    csv_file.write_text("id,value\n1,10\n", encoding="utf-8")
    assert is_materialized(csv_file)
