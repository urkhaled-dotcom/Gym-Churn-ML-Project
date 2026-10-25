"""Feature engineering for the gym member churn project."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class GymFeatureEngineer(BaseEstimator, TransformerMixin):
    """Add transparent, deterministic features derived only from input columns.

    Input DataFrame must not include the target column Churn.
    """
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        data = X.copy()
        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(data, columns=getattr(self, "feature_names_in_", None))
        total = pd.to_numeric(data["Avg_class_frequency_total"], errors="coerce")
        current = pd.to_numeric(data["Avg_class_frequency_current_month"], errors="coerce")
        contract = pd.to_numeric(data["Contract_period"], errors="coerce").replace(0, np.nan)
        lifetime = pd.to_numeric(data["Lifetime"], errors="coerce")
        months_left = pd.to_numeric(data["Month_to_end_contract"], errors="coerce")
        charges = pd.to_numeric(data["Avg_additional_charges_total"], errors="coerce")

        data["Frequency_change_current_vs_total"] = current - total
        data["Frequency_ratio_current_vs_total"] = current / total.replace(0, np.nan)
        data["Frequency_ratio_current_vs_total"] = data["Frequency_ratio_current_vs_total"].replace([np.inf, -np.inf], np.nan)
        data["Lifetime_to_contract_ratio"] = lifetime / contract
        data["Remaining_contract_ratio"] = months_left / contract
        data["Additional_charges_per_lifetime_month"] = charges / (lifetime + 1.0)
        data["Is_monthly_contract"] = (data["Contract_period"] == 1).astype(int)
        data["Is_long_contract"] = (data["Contract_period"] >= 6).astype(int)
        return data

    def get_feature_names_out(self, input_features=None):
        names = list(input_features if input_features is not None else getattr(self, "feature_names_in_", []))
        return np.asarray(names + [
            "Frequency_change_current_vs_total",
            "Frequency_ratio_current_vs_total",
            "Lifetime_to_contract_ratio",
            "Remaining_contract_ratio",
            "Additional_charges_per_lifetime_month",
            "Is_monthly_contract",
            "Is_long_contract",
        ], dtype=object)
