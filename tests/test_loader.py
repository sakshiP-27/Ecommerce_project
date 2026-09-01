"""Tests for the data loader (Week 1 / Week 3 Step 9)."""

import pandas as pd
import pytest

import src.data.loader as loader


def test_load_data_raises_on_missing_file(monkeypatch, tmp_path):
    missing = tmp_path / "does_not_exist.csv"
    monkeypatch.setattr(loader, "dataset_path", str(missing))

    with pytest.raises(FileNotFoundError):
        loader.load_data()


def test_load_data_raises_on_missing_column(monkeypatch, tmp_path):
    # CSV missing required column "PageValues"
    bad_csv = tmp_path / "bad_sessions.csv"
    pd.DataFrame(
        {
            "Administrative": [1],
            "Informational": [0],
            "ProductRelated": [5],
            "BounceRates": [0.01],
            "ExitRates": [0.02],
            # "PageValues" intentionally omitted
            "Month": ["June"],
            "VisitorType": ["Returning_Visitor"],
            "Weekend": [False],
            "Converted": [0],
        }
    ).to_csv(bad_csv, index=False)

    monkeypatch.setattr(loader, "dataset_path", str(bad_csv))

    with pytest.raises(ValueError, match="Missing expected columns"):
        loader.load_data()


def test_load_data_succeeds_on_valid_file(monkeypatch, tmp_path):
    good_csv = tmp_path / "good_sessions.csv"
    pd.DataFrame(
        {
            "Administrative": [1],
            "Informational": [0],
            "ProductRelated": [5],
            "BounceRates": [0.01],
            "ExitRates": [0.02],
            "PageValues": [10.0],
            "Month": ["June"],
            "VisitorType": ["Returning_Visitor"],
            "Weekend": [False],
            "Converted": [1],
        }
    ).to_csv(good_csv, index=False)

    monkeypatch.setattr(loader, "dataset_path", str(good_csv))
    df = loader.load_data()
    assert not df.empty
    assert "Converted" in df.columns
