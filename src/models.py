"""Small-sample exploratory classifiers and cross-validation helpers."""
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.base import clone
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score


def _variance_scores(X, y=None):
    """Score features by variance within the current training fold."""
    return np.nanvar(X, axis=0), None


def _align_series(values, index, name):
    """Align series by matching labels when possible, else by row order."""
    if isinstance(values, pd.Series):
        if values.index.equals(index):
            return values.rename(name)
        if index.isin(values.index).all():
            return values.reindex(index).rename(name)
        if len(values) == len(index):
            return pd.Series(values.to_numpy(), index=index, name=name)
        raise ValueError(f"{name} length or index does not match the feature rows.")
    if len(values) != len(index):
        raise ValueError(f"{name} length does not match the feature rows.")
    return pd.Series(values, index=index, name=name)


def encode_target(values):
    """Map a binary clinical label to integer classes and display labels."""
    s = pd.Series(values).dropna().astype(str)
    labels = sorted(s.unique())
    if len(labels) != 2:
        raise ValueError(f"Need exactly 2 target classes; found {len(labels)}: {labels}")
    positive = next((v for v in labels if any(word in v.lower() for word in ["resist", "progress", "recurr", "event", "yes", "non"])), labels[-1])
    mapping = {label: int(label == positive) for label in labels}
    return mapping


def make_model(kind="Logistic Regression", n_features=50, seed=42, n_samples=80):
    """Create a leakage-safe feature-selection/model pipeline."""
    estimator = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed) if kind == "Logistic Regression" else RandomForestClassifier(n_estimators=250, class_weight="balanced", random_state=seed, min_samples_leaf=2)
    steps = [("impute", SimpleImputer(strategy="median")), ("select", SelectKBest(_variance_scores, k=max(1, min(n_features, 500)))), ("scale", StandardScaler()), ("model", estimator)]
    return Pipeline(steps)


def grouped_splits(X, y, groups, seed=42, max_folds=5):
    """Return patient-disjoint folds, stratifying at patient level when possible."""
    y = _align_series(y, X.index, "target").astype(int)
    groups = _align_series(groups, X.index, "patient_id")
    if groups.isna().any():
        raise ValueError("Patient ID is missing for one or more samples; map patient_id in config.py.")
    per_patient_classes = pd.DataFrame({"group": groups, "y": y}).groupby("group").y.nunique()
    if (per_patient_classes > 1).any():
        raise ValueError("Outcome labels vary within a patient; use one patient-level outcome per patient.")
    patient_y = pd.DataFrame({"group": groups, "y": y}).drop_duplicates("group").set_index("group").y
    min_class_groups = int(patient_y.value_counts().min()) if patient_y.nunique() > 1 else 0
    folds = min(max_folds, int(groups.nunique()), min_class_groups)
    if folds < 2:
        raise ValueError("Patient-grouped CV needs at least two patients in each target class.")
    if StratifiedGroupKFold is not None:
        splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    else:  # pragma: no cover - compatibility for older scikit-learn
        splitter = GroupKFold(n_splits=folds)
    splits = list(splitter.split(X, y, groups))
    for train_idx, test_idx in splits:
        if set(groups.iloc[train_idx]) & set(groups.iloc[test_idx]):
            raise RuntimeError("Patient leakage detected: a patient occurs in train and test for one fold.")
    return splitter, splits, groups


def cross_validate_model(X, y, groups, kind="Logistic Regression", n_features=50, seed=42):
    """Fit patient-grouped CV and return metrics, OOF predictions, and fitted model."""
    X = pd.DataFrame(X).replace([np.inf, -np.inf], np.nan)
    y = _align_series(y, X.index, "target").astype(int)
    cv, splits, groups = grouped_splits(X, y, groups, seed)
    model = make_model(kind, n_features, seed, len(X))
    pred = np.full(len(X), np.nan)
    fold_auc, fold_accuracy, fold_patient_ids = [], [], []
    fold_positive_counts, fold_negative_counts, fold_models = [], [], []
    for train_idx, test_idx in splits:
        train_groups = set(groups.iloc[train_idx])
        test_groups = set(groups.iloc[test_idx])
        fold_model = clone(model).fit(X.iloc[train_idx], y.iloc[train_idx])
        fold_pred = fold_model.predict_proba(X.iloc[test_idx])[:, 1]
        pred[test_idx] = fold_pred
        fold_positive_counts.append(int((y.iloc[test_idx] == 1).sum()))
        fold_negative_counts.append(int((y.iloc[test_idx] == 0).sum()))
        fold_accuracy.append(accuracy_score(y.iloc[test_idx], fold_pred >= .5))
        if y.iloc[test_idx].nunique() == 2:
            fold_auc.append(roc_auc_score(y.iloc[test_idx], fold_pred))
        fold_models.append(fold_model)
        fold_patient_ids.append({"train": train_groups, "test": test_groups})
    model.fit(X, y)
    return {"model": model, "probabilities": pred, "auc_mean": float(np.mean(fold_auc)) if fold_auc else np.nan,
            "auc_std": float(np.std(fold_auc)) if fold_auc else np.nan,
            "pooled_auc": roc_auc_score(y, pred) if y.nunique() == 2 else np.nan,
            "accuracy": accuracy_score(y, pred >= .5),
            "baseline_accuracy": float(y.value_counts(normalize=True).max()),
            "confusion": confusion_matrix(y, pred >= .5), "X": X, "y": y, "groups": groups,
            "fold_patient_ids": fold_patient_ids, "fold_positive_counts": fold_positive_counts,
            "fold_negative_counts": fold_negative_counts,
            "fold_models": fold_models, "splits": splits, "cv": cv}


def _grouped_oof_auc(X, y, groups, kind, n_features, seed):
    """Return pooled OOF AUC for one label assignment without fitting a final model."""
    cv, splits, groups = grouped_splits(X, y, groups, seed)
    model = make_model(kind, n_features, seed, len(X))
    pred = np.full(len(X), np.nan)
    for train_idx, test_idx in splits:
        fold_model = clone(model).fit(X.iloc[train_idx], y.iloc[train_idx])
        pred[test_idx] = fold_model.predict_proba(X.iloc[test_idx])[:, 1]
    return roc_auc_score(y, pred)


def patient_label_shuffle_test(X, y, groups, observed_auc, kind="Logistic Regression", n_features=50, seed=42, n_shuffles=200):
    """Estimate a right-tail empirical AUC p-value by shuffling labels between patients."""
    X = pd.DataFrame(X).replace([np.inf, -np.inf], np.nan)
    y = _align_series(y, X.index, "target").astype(int)
    _, splits, groups = grouped_splits(X, y, groups, seed)
    patient_labels = pd.DataFrame({"patient_id": groups, "target": y}).drop_duplicates("patient_id")
    patient_ids = patient_labels.patient_id.to_numpy()
    labels = patient_labels.target.to_numpy()
    pipeline = make_model(kind, n_features, seed, len(X))

    # Feature selection, imputation, and scaling are label-independent here, so
    # compute each fold's transforms once and only refit the classifier per shuffle.
    prepared_folds = []
    for train_idx, test_idx in splits:
        imputer = clone(pipeline.named_steps["impute"])
        train_imputed = imputer.fit_transform(X.iloc[train_idx])
        test_imputed = imputer.transform(X.iloc[test_idx])
        selector = clone(pipeline.named_steps["select"])
        selector.fit(train_imputed, y.iloc[train_idx])
        train_selected = selector.transform(train_imputed)
        test_selected = selector.transform(test_imputed)
        scaler = clone(pipeline.named_steps["scale"])
        train_scaled = scaler.fit_transform(train_selected)
        test_scaled = scaler.transform(test_selected)
        prepared_folds.append((train_idx, test_idx, train_scaled, test_scaled))

    rng = np.random.default_rng(seed)
    shuffled_aucs = []
    for _ in range(n_shuffles):
        shuffled = dict(zip(patient_ids, rng.permutation(labels)))
        permuted_y = groups.map(shuffled).astype(int)
        pred = np.full(len(X), np.nan)
        for train_idx, test_idx, train_scaled, test_scaled in prepared_folds:
            estimator = clone(pipeline.named_steps["model"])
            estimator.fit(train_scaled, permuted_y.iloc[train_idx])
            pred[test_idx] = estimator.predict_proba(test_scaled)[:, 1]
        shuffled_aucs.append(roc_auc_score(permuted_y, pred))
    if not shuffled_aucs:
        raise ValueError("No valid patient-level label shuffles could be scored.")
    p_value = (1 + int(np.sum(np.asarray(shuffled_aucs) >= observed_auc))) / (1 + len(shuffled_aucs))
    return {"p_value": float(p_value), "valid_shuffles": len(shuffled_aucs), "shuffled_aucs": shuffled_aucs}


def train_progression(X, y, groups, seed=42):
    """Return a fitted progression model and patient-held-out OOF risk scores."""
    result = cross_validate_model(X, y, groups, "Logistic Regression", n_features=50, seed=seed)
    return result["model"], result["probabilities"]


def make_design(expression, clinical, include_clinical=True, exclude_clinical=None):
    """Join sample expression with encoded categorical clinical predictors."""
    X = expression.T.copy()
    if include_clinical:
        excluded = set(exclude_clinical or [])
        for col in ["treatment", "subtype"]:
            if col in excluded:
                continue
            if col in clinical:
                d = pd.get_dummies(clinical.set_index("sample")[col].astype(str), prefix=col, dummy_na=True)
                X = X.join(d, how="left")
    return X.replace([np.inf, -np.inf], np.nan)
