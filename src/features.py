"""Expression preparation and paired-shift helpers."""
import numpy as np
import pandas as pd


def normalize_expression(expression):
    """CPM-normalize counts, log2-transform, and retain variable genes."""
    x = expression.copy().apply(pd.to_numeric, errors="coerce").fillna(0).clip(lower=0)
    if (x.max().max() > 100):
        totals = x.sum(axis=0).replace(0, 1)
        x = np.log2(x.div(totals, axis=1) * 1_000_000 + 1)
    else:
        x = np.log2(x + 1)
    x = x.loc[x.mean(axis=1) >= .5]
    if len(x) > 2000:
        x = x.loc[x.var(axis=1).nlargest(2000).index]
    return x


def paired_deltas(expression, clinical):
    """Calculate per-gene post minus pre deltas for patients with both timepoints."""
    if not {"patient_id", "timepoint"}.issubset(clinical.columns):
        return pd.DataFrame(), []
    pairs = []
    meta = clinical.set_index("sample", drop=False)
    for patient, group in meta.groupby("patient_id", dropna=True):
        pre = group[group["timepoint"].astype(str).str.lower().str.contains("pre|baseline")]
        post = group[group["timepoint"].astype(str).str.lower().str.contains("post|after")]
        if len(pre) and len(post):
            a, b = pre.iloc[0]["sample"], post.iloc[0]["sample"]
            if a in expression.columns and b in expression.columns:
                pairs.append((patient, expression[b] - expression[a]))
    if not pairs:
        return pd.DataFrame(), []
    return pd.DataFrame({str(pid): d for pid, d in pairs}), [p[0] for p in pairs]
