"""Reproducible experiment pipeline for gym member churn classification.

Important: the supplied dataset is a snapshot dataset without an explicit event
calendar. Metrics describe prediction of its recorded Churn label under a
stratified holdout; they do not alone prove temporal early-warning performance.
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (ExtraTreesClassifier, GradientBoostingClassifier,
                              HistGradientBoostingClassifier, RandomForestClassifier)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             confusion_matrix, f1_score, fbeta_score, precision_score,
                             recall_score, roc_auc_score, classification_report,
                             precision_recall_curve, roc_curve)
from sklearn.model_selection import (GridSearchCV, StratifiedKFold, cross_val_predict,
                                     cross_validate, train_test_split)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.inspection import permutation_importance

from .features import GymFeatureEngineer

RANDOM_STATE = 42
TARGET = "Churn"
TARGET_MEANING = {0: "لم ينسحب حسب الملصق", 1: "انسحب حسب الملصق"}

FEATURE_DESCRIPTION_AR = {
    "gender": "ترميز الجنس كما ورد في المصدر؛ لا نفترض دلالته 0/1 دون قاموس المصدر",
    "Near_Location": "هل يسكن/يعمل بالقرب من موقع النادي (مؤشر ثنائي)",
    "Partner": "هل ارتبط التسجيل بشركة شريكة (مؤشر ثنائي)",
    "Promo_friends": "هل جاء الاشتراك عبر عرض إحالة الأصدقاء (مؤشر ثنائي)",
    "Phone": "هل يوجد رقم هاتف مسجل (مؤشر ثنائي)",
    "Contract_period": "مدة العقد بالأشهر (1 أو 6 أو 12 في هذه العينة)",
    "Group_visits": "هل يشارك المشترك في الحصص الجماعية (مؤشر ثنائي)",
    "Age": "عمر المشترك",
    "Avg_additional_charges_total": "إجمالي الإنفاق على الخدمات/المشتريات الإضافية حسب تعريف المصدر",
    "Month_to_end_contract": "عدد الأشهر المتبقية حتى انتهاء العقد وفق السجل",
    "Lifetime": "مدة علاقة المشترك بالنادي حسب تعريف المصدر، بالأشهر",
    "Avg_class_frequency_total": "متوسط تكرار الحضور للفترة الإجمالية المسجلة",
    "Avg_class_frequency_current_month": "متوسط تكرار الحضور للفترة الحالية المذكورة في المصدر",
    "Churn": "الهدف: 1 = انسحاب، 0 = عدم انسحاب وفق ترميز البيانات المنشورة",
}


def load_data(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if TARGET not in df.columns:
        # Accept case-insensitive target spelling but preserve original feature names.
        found = next((c for c in df.columns if c.lower() == TARGET.lower()), None)
        if found is None:
            raise ValueError(f"لم يُعثر على عمود الهدف {TARGET!r}. الأعمدة: {list(df.columns)}")
        df = df.rename(columns={found: TARGET})
    return df


def audit_data(df: pd.DataFrame) -> dict:
    if TARGET not in df:
        raise ValueError(f"يجب أن يحتوي الملف على عمود {TARGET}.")
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    if y.isna().any():
        raise ValueError("عمود الهدف يحتوي على قيم مفقودة؛ يجب حسمها قبل التدريب.")
    if set(pd.Series(y).unique()) != {0, 1}:
        raise ValueError(f"يجب أن تكون قيم الهدف 0 و1 فقط. القيم الموجودة: {sorted(pd.Series(y).unique())}")
    return {
        "n_rows": int(df.shape[0]),
        "n_columns_total": int(df.shape[1]),
        "n_features_original": int(X.shape[1]),
        "missing_cells_total": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "target_counts": {str(int(k)): int(v) for k, v in y.value_counts().sort_index().items()},
        "target_rate": float(y.mean()),
        "all_features_numeric": bool(all(pd.api.types.is_numeric_dtype(X[c]) for c in X.columns)),
        "feature_names": list(X.columns),
    }


def make_models():
    """Return reasonable models; optional libraries are skipped only if unavailable."""
    models = {
        "LogisticRegression": (LogisticRegression(max_iter=2500, random_state=RANDOM_STATE), True),
        "DecisionTree": (DecisionTreeClassifier(min_samples_leaf=5, random_state=RANDOM_STATE), False),
        "RandomForest": (RandomForestClassifier(n_estimators=140, min_samples_leaf=2, max_features="sqrt", random_state=RANDOM_STATE, n_jobs=1), False),
        "ExtraTrees": (ExtraTreesClassifier(n_estimators=140, min_samples_leaf=2, max_features="sqrt", random_state=RANDOM_STATE, n_jobs=1), False),
        "GradientBoosting": (GradientBoostingClassifier(n_estimators=100, learning_rate=0.05, max_depth=2, random_state=RANDOM_STATE), False),
        "HistGradientBoosting": (HistGradientBoostingClassifier(max_iter=120, learning_rate=0.08, max_leaf_nodes=15, l2_regularization=1.0, random_state=RANDOM_STATE), False),
        "KNN": (KNeighborsClassifier(n_neighbors=15), True),
        "SVC": (SVC(C=1.0, kernel="rbf", probability=True, random_state=RANDOM_STATE), True),
    }
    # Optional boosting libraries are omitted from the baseline grid to keep the
    # project reproducible on modest student laptops without compiler/library hassles.
    # CatBoost is intentionally not included in the default experiment matrix because
    # some installed versions do not support the newest scikit-learn estimator tags.
    # It can be added after confirming library compatibility in the target environment.
    return models


def build_pipeline(model, scaled: bool, engineered: bool) -> Pipeline:
    steps = []
    if engineered:
        steps.append(("features", GymFeatureEngineer()))
    steps.append(("imputer", SimpleImputer(strategy="median")))
    if scaled:
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", clone(model)))
    return Pipeline(steps)


def metric_bundle(y_true, proba, threshold=0.5):
    pred = (np.asarray(proba) >= threshold).astype(int)
    y_true = np.asarray(y_true).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "f2": float(fbeta_score(y_true, pred, beta=2, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "pr_auc_average_precision": float(average_precision_score(y_true, proba)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def select_threshold_by_f2(y_true, oof_proba):
    # Threshold is chosen using training-only out-of-fold probabilities.
    candidates = np.linspace(0.10, 0.90, 161)
    rows = []
    for t in candidates:
        p = (np.asarray(oof_proba) >= t).astype(int)
        rows.append({"threshold": float(t), "precision": precision_score(y_true, p, zero_division=0),
                     "recall": recall_score(y_true, p, zero_division=0),
                     "f1": f1_score(y_true, p, zero_division=0),
                     "f2": fbeta_score(y_true, p, beta=2, zero_division=0)})
    curve = pd.DataFrame(rows)
    best_idx = curve["f2"].idxmax()
    return float(curve.loc[best_idx, "threshold"]), curve


def run_experiments(data_path: str | Path, out_dir: str | Path, force: bool = False) -> dict:
    out_dir = Path(out_dir)
    reports = out_dir / "reports"
    figs = reports / "figures"
    models_dir = out_dir / "models"
    for d in (reports, figs, models_dir): d.mkdir(parents=True, exist_ok=True)
    results_path = reports / "final_results.json"
    if results_path.exists() and (models_dir / "gym_churn_model.joblib").exists() and not force:
        with open(results_path, "r", encoding="utf-8") as f:
            return json.load(f)

    df = load_data(data_path)
    audit = audit_data(df)
    X = df.drop(columns=[TARGET]).copy()
    y = df[TARGET].astype(int).copy()
    # Basic source-specific validity checks; do not silently delete rows.
    checks = {
        "binary_columns_values_valid": all(set(X[c].dropna().unique()).issubset({0, 1}) for c in ["gender", "Near_Location", "Partner", "Promo_friends", "Phone", "Group_visits"] if c in X),
        "contract_period_values": sorted([int(v) for v in X["Contract_period"].dropna().unique()]) if "Contract_period" in X else [],
        "remaining_months_never_exceed_contract": bool((X["Month_to_end_contract"] <= X["Contract_period"]).all()) if {"Month_to_end_contract", "Contract_period"}.issubset(X.columns) else None,
        "features_directly_named_as_target_or_churn": [c for c in X.columns if any(token in c.lower() for token in ["churn", "cancel", "target", "left_member"])],
    }
    if audit["missing_cells_total"]:
        warnings.warn("توجد قيم مفقودة؛ ستتم معالجتها داخل خط المعالجة باستخدام median للخصائص الرقمية.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    scoring = {
        "accuracy": "accuracy", "balanced_accuracy": "balanced_accuracy",
        "precision": "precision", "recall": "recall", "f1": "f1",
        "roc_auc": "roc_auc", "pr_auc": "average_precision",
    }
    baseline = DummyClassifier(strategy="prior")
    baseline_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", baseline)])
    baseline_cv = cross_validate(baseline_pipe, X_train, y_train, cv=cv, scoring=scoring, n_jobs=1, error_score="raise")
    rows = [{"model": "DummyPrior", "feature_set": "original", **{k: float(np.mean(v)) for k, v in baseline_cv.items() if k.startswith("test_") and k != "test_fit_time" and k != "test_score_time"}}]
    # More readable CV summary, with averages and standard deviations.
    rows = []
    rows.append({"model": "DummyPrior", "feature_set": "original", **{m: float(np.mean(baseline_cv[f"test_{m}"])) for m in scoring}, **{f"{m}_std": float(np.std(baseline_cv[f"test_{m}"])) for m in scoring}})
    models = make_models()
    cv_folds_rows = []
    for feature_set, engineered in [("original", False), ("engineered", True)]:
        for name, (estimator, scaled) in models.items():
            pipe = build_pipeline(estimator, scaled=scaled, engineered=engineered)
            fold_results = cross_validate(pipe, X_train, y_train, cv=cv, scoring=scoring, n_jobs=1, error_score="raise", return_train_score=False)
            record = {"model": name, "feature_set": feature_set}
            for metric in scoring:
                values = fold_results[f"test_{metric}"]
                record[metric] = float(np.mean(values))
                record[f"{metric}_std"] = float(np.std(values))
            rows.append(record)
            for i in range(len(fold_results["test_accuracy"])):
                cv_folds_rows.append({"model": name, "feature_set": feature_set, "fold": i + 1,
                                      **{m: float(fold_results[f"test_{m}"][i]) for m in scoring}})
    cv_summary = pd.DataFrame(rows).sort_values(["pr_auc", "roc_auc", "f1"], ascending=False).reset_index(drop=True)
    cv_summary.to_csv(reports / "model_comparison_cv.csv", index=False)
    pd.DataFrame(cv_folds_rows).to_csv(reports / "cv_fold_results.csv", index=False)

    # Choose by mean PR-AUC on cross-validation, not by holdout test.
    best_row = cv_summary.iloc[0].to_dict()
    best_name, best_feature_set = str(best_row["model"]), str(best_row["feature_set"])
    if best_name == "DummyPrior":
        # A real model should beat dummy; keep selection robust if data is unusual.
        non_dummy = cv_summary[cv_summary.model != "DummyPrior"]
        best_row = non_dummy.iloc[0].to_dict()
        best_name, best_feature_set = str(best_row["model"]), str(best_row["feature_set"])
    engineered = best_feature_set == "engineered"
    base_estimator, scaled = models[best_name]
    base_pipe = build_pipeline(base_estimator, scaled=scaled, engineered=engineered)

    # Small, bounded search for the selected family. CV uses only training split.
    if best_name == "RandomForest":
        grid = {"model__n_estimators": [100, 160], "model__max_depth": [None, 12], "model__min_samples_leaf": [2, 4]}
    elif best_name == "ExtraTrees":
        grid = {"model__n_estimators": [100, 160], "model__max_depth": [None, 12], "model__min_samples_leaf": [2, 4]}
    elif best_name == "LogisticRegression":
        grid = {"model__C": [0.1, 0.3, 1.0, 3.0, 10.0], "model__class_weight": [None, "balanced"]}
    elif best_name == "LightGBM":
        grid = {"model__n_estimators": [160, 240, 320], "model__num_leaves": [7, 15, 23], "model__learning_rate": [0.025, 0.04, 0.07]}
    elif best_name == "XGBoost":
        grid = {"model__n_estimators": [160, 240, 320], "model__max_depth": [2, 3, 4], "model__learning_rate": [0.025, 0.04, 0.07]}
    elif best_name == "HistGradientBoosting":
        grid = {"model__max_iter": [100, 160], "model__learning_rate": [0.05, 0.08], "model__max_leaf_nodes": [7, 15]}
    elif best_name == "GradientBoosting":
        grid = {"model__n_estimators": [80, 120], "model__learning_rate": [0.04, 0.07], "model__max_depth": [1, 2]}
    elif best_name == "DecisionTree":
        grid = {"model__max_depth": [5, 8, None], "model__min_samples_leaf": [2, 5]}
    elif best_name == "KNN":
        grid = {"model__n_neighbors": [11, 21], "model__weights": ["uniform", "distance"]}
    else:  # SVC or any unlisted supported estimator
        grid = {"model__C": [0.3, 1.0, 3.0, 10.0], "model__gamma": ["scale", "auto"]}
    search = GridSearchCV(base_pipe, grid, scoring="average_precision", cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE), n_jobs=1, refit=True, return_train_score=False)
    search.fit(X_train, y_train)
    tuned_model = search.best_estimator_
    pd.DataFrame(search.cv_results_).sort_values("rank_test_score").to_csv(reports / "hyperparameter_tuning.csv", index=False)

    # Nested out-of-fold probabilities on training only for decision-threshold selection.
    # Each outer fold tunes a fresh model only on that fold's sub-training data, so its
    # validation labels do not influence the hyperparameter selection used for its score.
    outer_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE + 2)
    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE + 1)
    threshold_estimator = GridSearchCV(base_pipe, grid, scoring="average_precision", cv=inner_cv,
                                       n_jobs=1, refit=True, return_train_score=False)
    oof_proba = cross_val_predict(threshold_estimator, X_train, y_train, cv=outer_cv,
                                  method="predict_proba", n_jobs=1)[:, 1]
    threshold, threshold_curve = select_threshold_by_f2(y_train, oof_proba)
    threshold_curve.to_csv(reports / "oof_threshold_curve.csv", index=False)
    oof_results = pd.DataFrame({"y_true": np.asarray(y_train), "oof_probability": oof_proba})
    oof_results["predicted_churn_threshold_0_5"] = (oof_results.oof_probability >= 0.5).astype(int)
    oof_results["predicted_churn_selected_threshold"] = (oof_results.oof_probability >= threshold).astype(int)
    oof_results.to_csv(reports / "oof_threshold_predictions.csv", index=False)

    # Final holdout is assessed only after model family, features, parameters and threshold are fixed.
    tuned_model.fit(X_train, y_train)
    test_proba = tuned_model.predict_proba(X_test)[:, 1]
    test_metrics = metric_bundle(y_test, test_proba, threshold=threshold)
    test_metrics_default = metric_bundle(y_test, test_proba, threshold=0.5)
    test_pred = (test_proba >= threshold).astype(int)
    test_predictions = X_test.reset_index(drop=True).copy()
    test_predictions["actual_Churn"] = np.asarray(y_test)
    test_predictions["predicted_churn_probability"] = test_proba
    test_predictions["predicted_Churn"] = test_pred
    test_predictions.to_csv(reports / "final_test_predictions.csv", index=False)
    pd.DataFrame(classification_report(y_test, test_pred, output_dict=True, zero_division=0)).T.to_csv(reports / "classification_report.csv")

    # Permutation importance on held-out data is used only for interpretation after model selection.
    perm = permutation_importance(tuned_model, X_test, y_test, n_repeats=8, random_state=RANDOM_STATE, scoring="average_precision", n_jobs=1)
    importance = pd.DataFrame({"feature": list(X_test.columns), "importance_mean": perm.importances_mean, "importance_std": perm.importances_std}).sort_values("importance_mean", ascending=False)
    importance.to_csv(reports / "feature_importance_permutation.csv", index=False)

    final_model_path = models_dir / "gym_churn_model.joblib"
    joblib.dump({"pipeline": tuned_model, "feature_columns": list(X.columns), "target": TARGET,
                 "decision_threshold": threshold, "model_name": best_name,
                 "feature_set": best_feature_set, "random_state": RANDOM_STATE,
                 "notes": "Model predicts the recorded Churn label in this dataset. Temporal early-warning validity needs time-indexed external validation."}, final_model_path)

    # Plots are generated from real data/results, not illustrative values.
    make_plots(df, cv_summary, importance, threshold_curve, y_test, test_proba, threshold, figs)
    results = {
        "project_title_ar": "نظام تحليل مخاطر انسحاب المشتركين من النادي الرياضي باستخدام تعلم الآلة",
        "dataset_file": "data/gym_churn_us.csv",
        "audit": audit,
        "data_checks": checks,
        "split": {"train_rows": int(len(X_train)), "test_rows": int(len(X_test)), "test_size": 0.20, "stratified": True, "random_state": RANDOM_STATE, "cv_folds": 3},
        "best_model": best_name,
        "best_feature_set": best_feature_set,
        "cv_best_mean_pr_auc": float(best_row["pr_auc"]),
        "cv_best_mean_roc_auc": float(best_row["roc_auc"]),
        "best_hyperparameters": {k: (v.item() if hasattr(v, "item") else v) for k, v in search.best_params_.items()},
        "threshold_selection": {"method": "maximize F2 using nested OOF probabilities from training split only; each outer validation fold is excluded from hyperparameter selection", "selected_threshold": float(threshold), "oof_metrics": metric_bundle(y_train, oof_proba, threshold)},
        "final_test_metrics_selected_threshold": test_metrics,
        "final_test_metrics_default_threshold_0_5": test_metrics_default,
        "test_confusion_matrix": [[test_metrics["tn"], test_metrics["fp"]], [test_metrics["fn"], test_metrics["tp"]]],
        "model_path": "models/gym_churn_model.joblib",
        "important_interpretation_note": "لأن الملف لا يحتوي على عمود تاريخ/لقطة زمنية لكل عضو، فإن اختبارًا عشوائيًا طبقيًا يقيّم التنبؤ بتصنيف الانسحاب المرفق، ولا يثبت وحده أداء إنذار مبكر زمني على بيانات مستقبلية أو نادٍ آخر.",
        "classification_report": classification_report(y_test, test_pred, output_dict=True, zero_division=0),
    }
    # JSON-safe cleanup for numpy scalar values
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=lambda x: x.item() if hasattr(x, "item") else str(x))
    cv_summary.to_csv(reports / "model_comparison_cv.csv", index=False)
    return results


def make_plots(df, cv_summary, importance, threshold_curve, y_test, test_proba, threshold, figs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_theme(style="whitegrid", font_scale=0.92)
    # 1 target distribution
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    counts = df[TARGET].value_counts().reindex([0, 1])
    bars = ax.bar(["Retained (Churn=0)", "Churned (Churn=1)"], counts.values, color=["#357E9A", "#D77A61"])
    ax.set_title("Target Class Distribution", fontsize=14, weight="bold")
    ax.set_ylabel("Number of members")
    for b, val in zip(bars, counts.values): ax.text(b.get_x()+b.get_width()/2, val+15, f"{val:,}\n({val/len(df):.1%})", ha="center", va="bottom")
    fig.tight_layout(); fig.savefig(figs / "target_distribution.png", dpi=170, bbox_inches="tight"); plt.close(fig)
    # 2 target rates by contract group
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    rate = df.groupby("Contract_period")[TARGET].agg(["mean", "count"]).reset_index()
    bars = ax.bar(rate["Contract_period"].astype(str)+" شهر", rate["mean"]*100, color="#6B75B8")
    ax.set_title("Churn Rate by Contract Length", fontsize=14, weight="bold"); ax.set_ylabel("Churn rate (%)"); ax.set_xlabel("Contract length")
    for b, val in zip(bars, rate["mean"]): ax.text(b.get_x()+b.get_width()/2, val*100+1, f"{val:.1%}", ha="center")
    ax.set_ylim(0, min(100, max(rate["mean"]*100)*1.22)); fig.tight_layout(); fig.savefig(figs / "churn_by_contract.png", dpi=170, bbox_inches="tight"); plt.close(fig)
    # 3 class frequency distributions
    fig, ax = plt.subplots(figsize=(8, 4.8))
    sns.kdeplot(data=df, x="Avg_class_frequency_current_month", hue=TARGET, common_norm=False, fill=False, ax=ax, palette={0:"#357E9A",1:"#D77A61"})
    ax.set_title("Current Attendance Frequency by Churn Label", fontsize=14, weight="bold")
    ax.set_xlabel("Average attendance frequency (source definition)"); ax.set_ylabel("Density")
    ax.legend(title="Churn", labels=["Retained (Churn=0)", "Churned (Churn=1)"])
    fig.tight_layout(); fig.savefig(figs / "frequency_by_churn.png", dpi=170, bbox_inches="tight"); plt.close(fig)
    # 4 correlation
    corr_cols = [c for c in df.select_dtypes(include=np.number).columns]
    corr = df[corr_cols].corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, cmap="vlag", center=0, linewidths=.25, ax=ax, cbar_kws={"label":"Correlation coefficient"})
    ax.set_title("Correlation Matrix of Numeric Features", fontsize=14, weight="bold")
    fig.tight_layout(); fig.savefig(figs / "correlation_matrix.png", dpi=160, bbox_inches="tight"); plt.close(fig)
    # 5 model comparison PR AUC
    c = cv_summary.sort_values("pr_auc", ascending=True).copy()
    fig, ax = plt.subplots(figsize=(9, max(5.5, len(c)*0.35)))
    labels = c["model"] + " — " + c["feature_set"].map({"original":"Original", "engineered":"Engineered"})
    ax.barh(labels, c["pr_auc"], color=["#357E9A" if "original" in f else "#8D79B8" for f in c["feature_set"]])
    ax.set_xlim(0, 1); ax.set_xlabel("Mean cross-validation PR-AUC"); ax.set_title("Model Comparison — Higher PR-AUC Is Better", fontsize=14, weight="bold")
    fig.tight_layout(); fig.savefig(figs / "model_comparison_pr_auc.png", dpi=170, bbox_inches="tight"); plt.close(fig)
    # 6 feature importance
    im = importance.head(12).sort_values("importance_mean", ascending=True)
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(im.feature, im.importance_mean, xerr=im.importance_std, color="#3D9A91", alpha=.9)
    ax.axvline(0, color="#444444", linewidth=.8); ax.set_title("Feature Importance — Permutation Importance", fontsize=14, weight="bold")
    ax.set_xlabel("Decrease in average precision when feature is shuffled")
    fig.tight_layout(); fig.savefig(figs / "feature_importance.png", dpi=170, bbox_inches="tight"); plt.close(fig)
    # 7 threshold F2 and recall/precision
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(threshold_curve.threshold, threshold_curve.f2, label="F2", linewidth=2.4, color="#357E9A")
    ax.plot(threshold_curve.threshold, threshold_curve.recall, label="Recall", linewidth=1.8, color="#D77A61")
    ax.plot(threshold_curve.threshold, threshold_curve.precision, label="Precision", linewidth=1.8, color="#6B75B8")
    ax.axvline(threshold, linestyle="--", color="#333333", label=f"العتبة المختارة = {threshold:.2f}")
    ax.set_xlabel("Probability threshold"); ax.set_ylabel("Metric value"); ax.set_ylim(0, 1.02); ax.set_title("Decision Threshold Selection from Training OOF Predictions", fontsize=13, weight="bold"); ax.legend()
    fig.tight_layout(); fig.savefig(figs / "threshold_selection.png", dpi=170, bbox_inches="tight"); plt.close(fig)
    # 8 test PR curve
    from sklearn.metrics import precision_recall_curve, roc_curve, auc
    precision, recall, _ = precision_recall_curve(y_test, test_proba)
    fig, ax = plt.subplots(figsize=(6.8, 5.2)); ax.plot(recall, precision, color="#8D79B8", linewidth=2.3)
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.set_title("Precision–Recall Curve on Final Test Set", fontsize=12, weight="bold"); ax.set_xlim(0,1); ax.set_ylim(0,1.02)
    fig.tight_layout(); fig.savefig(figs / "test_precision_recall_curve.png", dpi=170, bbox_inches="tight"); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Run gym churn experiments")
    parser.add_argument("--data", default="data/gym_churn_us.csv", help="CSV data path")
    parser.add_argument("--out", default=".", help="Output project directory")
    parser.add_argument("--force", action="store_true", help="Re-run training and overwrite artifacts")
    args = parser.parse_args()
    result = run_experiments(args.data, args.out, force=args.force)
    print(json.dumps({"best_model": result.get("best_model"), "best_feature_set": result.get("best_feature_set"),
                      "threshold": result.get("threshold_selection", {}).get("selected_threshold"),
                      "test_metrics": result.get("final_test_metrics_selected_threshold")}, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
