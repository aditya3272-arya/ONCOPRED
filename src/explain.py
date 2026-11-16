"""Model interpretation helpers using scikit-learn permutation importance."""
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.base import clone
from src.models import make_model, grouped_splits


def permutation_scores(model, X, y, seed=42):
    """Calculate permutation importance of input features on fitted pipeline."""
    result = permutation_importance(model, X, y, n_repeats=5, random_state=seed, scoring="roc_auc")
    return pd.Series(result.importances_mean, index=X.columns).sort_values(ascending=False)


def grouped_oof_permutation_scores(X, y, groups, seed=42, kind="Logistic Regression", n_features=50, repeats=3):
    """Average gene-only permutation importance on held-out patient folds."""
    X = pd.DataFrame(X).replace([np.inf, -np.inf], np.nan)
    y = pd.Series(y, index=X.index).astype(int)
    cv, splits, _ = grouped_splits(X, y, groups, seed)
    base_model = make_model(kind, n_features, seed, len(X))
    fold_scores = []
    for fold_number, (train_idx, test_idx) in enumerate(splits):
        if y.iloc[test_idx].nunique() < 2:
            continue
        model = clone(base_model).fit(X.iloc[train_idx], y.iloc[train_idx])
        support = model.named_steps["select"].get_support()
        selected_test = model.named_steps["impute"].transform(X.iloc[test_idx])[:, support]
        selected_test = model.named_steps["scale"].transform(selected_test)
        selected_genes = X.columns[support]
        result = permutation_importance(
            model.named_steps["model"], selected_test, y.iloc[test_idx], n_repeats=repeats,
            random_state=seed + fold_number, scoring="roc_auc",
        )
        fold_scores.append(pd.Series(result.importances_mean, index=selected_genes))
    if not fold_scores:
        raise ValueError("No held-out patient fold contained both target classes for permutation importance.")
    return pd.concat(fold_scores, axis=1).mean(axis=1).sort_values(ascending=False)


def contributions(model, row):
    """Approximate per-feature log-odds contributions for logistic pipelines."""
    try:
        pipe = model
        x = pipe.named_steps["impute"].transform(pd.DataFrame([row]))
        x = pipe.named_steps["select"].transform(x)
        x = pipe.named_steps["scale"].transform(x)[0]
        names = np.asarray(row.index)[pipe.named_steps["select"].get_support()]
        estimator = pipe.named_steps["model"]
        if not hasattr(estimator, "coef_"):
            return pd.Series(dtype=float)
        return pd.Series(x * estimator.coef_[0], index=names).sort_values(key=abs, ascending=False)
    except (KeyError, ValueError, IndexError):
        return pd.Series(dtype=float)


def sample_gene_contributions(model, row, gene_columns):
    """Return per-sample gene pushes for logistic or tree-based pipelines."""
    gene_set = set(map(str, gene_columns))
    estimator = model.named_steps["model"]
    if hasattr(estimator, "coef_"):
        values = contributions(model, row)
        return values[[str(name) in gene_set for name in values.index]]

    selector = model.named_steps["select"]
    selected_names = np.asarray(row.index.astype(str))[selector.get_support()]
    selected_genes = [name for name in selected_names if name in gene_set]
    if not selected_genes:
        return pd.Series(dtype=float)
    baseline_probability = float(model.predict_proba(pd.DataFrame([row]))[0, 1])
    medians = model.named_steps["impute"].statistics_
    positions = {str(name): i for i, name in enumerate(row.index)}
    effects = {}
    for gene in selected_genes:
        changed = row.copy()
        changed.loc[gene] = medians[positions[gene]]
        changed_probability = float(model.predict_proba(pd.DataFrame([changed]))[0, 1])
        effects[gene] = baseline_probability - changed_probability
    return pd.Series(effects).sort_values(key=abs, ascending=False)
