"""Load the saved model and predict the probability of the Churn label."""
from __future__ import annotations
import argparse
import joblib
import pandas as pd
from pathlib import Path


def predict_csv(model_path: str, input_csv: str, output_csv: str | None = None) -> pd.DataFrame:
    bundle = joblib.load(model_path)
    model = bundle["pipeline"]
    expected = bundle["feature_columns"]
    data = pd.read_csv(input_csv)
    # If a supplied row file contains Churn, remove it to avoid accidental target leakage.
    data = data.drop(columns=[c for c in data.columns if c.lower() == "churn"], errors="ignore")
    missing = [c for c in expected if c not in data.columns]
    extra = [c for c in data.columns if c not in expected]
    if missing:
        raise ValueError(f"أعمدة مطلوبة مفقودة: {missing}")
    data = data[expected].copy()
    probability = model.predict_proba(data)[:, 1]
    threshold = float(bundle["decision_threshold"])
    result = data.copy()
    result["churn_probability"] = probability
    result["predicted_churn"] = (probability >= threshold).astype(int)
    result["decision_threshold"] = threshold
    if extra:
        result.attrs["ignored_extra_columns"] = extra
    if output_csv:
        result.to_csv(output_csv, index=False)
    return result


def main():
    parser = argparse.ArgumentParser(description="Predict churn probabilities from a CSV")
    parser.add_argument("--model", default="models/gym_churn_model.joblib")
    parser.add_argument("--input", required=True, help="CSV with feature columns")
    parser.add_argument("--output", default="reports/predictions_new_data.csv")
    args = parser.parse_args()
    result = predict_csv(args.model, args.input, args.output)
    print(result.head().to_string(index=False))
    print(f"\nحُفظت التوقعات في: {args.output}")

if __name__ == "__main__":
    main()
