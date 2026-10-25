from pathlib import Path
import joblib
import pandas as pd
from src.train import audit_data, load_data
from src.features import GymFeatureEngineer

ROOT = Path(__file__).resolve().parents[1]


def test_dataset_shape_target_and_quality():
    df = load_data(ROOT / "data" / "gym_churn_us.csv")
    audit = audit_data(df)
    assert audit["n_rows"] == 4000
    assert audit["n_features_original"] == 13
    assert audit["missing_cells_total"] == 0
    assert audit["duplicate_rows"] == 0
    assert set(df["Churn"].unique()) == {0, 1}


def test_feature_engineering_adds_seven_expected_features():
    df = load_data(ROOT / "data" / "gym_churn_us.csv")
    X = df.drop(columns=["Churn"]).head(10)
    transformed = GymFeatureEngineer().fit_transform(X)
    assert transformed.shape[1] == X.shape[1] + 7
    assert "Frequency_change_current_vs_total" in transformed.columns
    assert "Is_monthly_contract" in transformed.columns


def test_final_model_predicts_probabilities_between_zero_and_one():
    bundle = joblib.load(ROOT / "models" / "gym_churn_model.joblib")
    df = load_data(ROOT / "data" / "gym_churn_us.csv")
    X = df.drop(columns=["Churn"]).iloc[:5]
    prob = bundle["pipeline"].predict_proba(X)[:, 1]
    assert len(prob) == 5
    assert ((prob >= 0) & (prob <= 1)).all()
    assert 0 < bundle["decision_threshold"] < 1


def test_features_do_not_include_target():
    df = load_data(ROOT / "data" / "gym_churn_us.csv")
    X = df.drop(columns=["Churn"])
    assert "Churn" not in X.columns
    assert not any(token in c.lower() for c in X.columns for token in ["target", "churn_label", "cancelled"])
