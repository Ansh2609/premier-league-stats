"""Sanity tests for the merge -> feature -> train steps, using the small
synthetic dataset in data/sample/ so these don't depend on network access.
"""
import pandas as pd

from src.clean_merge import build_players_table
from src.features import build_feature_matrix
from src.train_model import train


def test_build_players_table_merges_all_sample_players(tmp_path):
    output_csv = tmp_path / "players.csv"
    df = build_players_table(
        fbref_csv="data/sample/fbref_sample.csv",
        transfermarkt_csv="data/sample/transfermarkt_sample.csv",
        output_csv=str(output_csv),
    )
    assert len(df) == 20
    assert "MarketValue" in df.columns
    assert df["MarketValue"].notna().all()


def test_build_feature_matrix_shapes():
    df = pd.read_csv("data/sample/players_sample.csv", parse_dates=["ContractExpires"])
    X, y, feature_columns = build_feature_matrix(df)
    assert len(X) == len(df)
    assert len(y) == len(df)
    assert set(feature_columns) == set(X.columns)
    assert X.isna().sum().sum() == 0


def test_train_produces_predictions(tmp_path):
    model_path = tmp_path / "model.joblib"
    _, metrics, (X_test, y_test, predictions) = train(
        "data/sample/players_sample.csv", model_path=str(model_path)
    )
    assert model_path.exists()
    assert len(predictions) == metrics["n_test"]
    assert metrics["mae"] >= 0
