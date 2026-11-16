"""ONCOPRED Streamlit dashboard entry point."""
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path
import csv
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.stats import ttest_1samp
from config import COLUMN_MAPPING
from src.data_loader import load_data
from src.features import normalize_expression, paired_deltas
from src.models import encode_target, cross_validate_model, make_design, patient_label_shuffle_test
from src.explain import grouped_oof_permutation_scores, sample_gene_contributions

st.set_page_config(page_title="ONCOPRED", page_icon="🧬", layout="wide")

PLOTLY_PALETTE = ["#20C7B7", "#8A96A8", "#566274", "#CBD5E1"]
CONTINUOUS_PALETTE = ["#263442", "#566274", "#20C7B7", "#DDE7EF"]
px.defaults.template = "plotly_dark"
px.defaults.color_discrete_sequence = PLOTLY_PALETTE
px.defaults.color_continuous_scale = CONTINUOUS_PALETTE
pio.templates.default = "plotly_dark"


def show_chart(fig, how_to_read=None):
    """Apply shared chart styling and an optional short reading cue."""
    fig.update_layout(
        template="plotly_dark",
        colorway=PLOTLY_PALETTE,
        font={"family": "Arial, sans-serif", "color": "#E8EEF4", "size": 12},
        paper_bgcolor="#0B1018",
        plot_bgcolor="#0B1018",
        margin={"l": 45, "r": 25, "t": 55, "b": 45},
    )
    fig.update_xaxes(gridcolor="#263241", zerolinecolor="#445364", tickfont={"size": 11})
    fig.update_yaxes(gridcolor="#263241", zerolinecolor="#445364", tickfont={"size": 11})
    st.plotly_chart(fig, use_container_width=True)
    if how_to_read:
        st.caption(f"How to read this: {how_to_read}")


FRIENDLY_COLUMNS = {
    "sample": "Sample",
    "patient_id": "Patient ID",
    "timepoint": "Treatment timepoint",
    "treatment": "Treatment regimen",
    "chemotherapy": "Treatment regimen",
    "response": "Treatment outcome",
    "recurrence": "Recurrence (progression proxy)",
    "progression": "Progression status",
    "followup_time": "Follow-up time",
    "subtype": "Molecular subtype",
    "neoadjuvant_response_index_(nri)": "Neoadjuvant Response Index (NRI)",
    "risk": "Out-of-fold progression risk",
    "risk_group": "Risk group",
}


def friendly_column(name):
    """Translate common clinical field names into reader-facing labels."""
    return FRIENDLY_COLUMNS.get(str(name).lower(), str(name).replace("_", " ").capitalize())


def friendly_values(values, column_name):
    """Render binary recurrence values as readable outcome names."""
    if "recurrence" in str(column_name).lower():
        return pd.Series(values, index=values.index).astype(str).map({"0": "No recurrence", "1": "Recurrence (progression proxy)"}).fillna(values.astype(str))
    return values.astype(str)


def page_intro(title, subtitle):
    """Show a consistent page title and one-line description."""
    st.title(title)
    st.caption(subtitle)

@st.cache_data(show_spinner="Loading expression and clinical data…")
def cached_data(demo, seed):
    expr, clinical, mutations, detected, is_demo = load_data("data", demo, seed)
    return normalize_expression(expr), clinical, mutations, detected, is_demo


@st.cache_data(show_spinner="Running patient-level label shuffles…")
def cached_label_shuffle_test(X, y, groups, observed_auc, kind, n_features, seed):
    """Cache the 200-shuffle null test across Streamlit reruns."""
    return patient_label_shuffle_test(X, y, groups, observed_auc, kind, n_features, seed, n_shuffles=200)


def col_for(clin, key):
    """Resolve configured field name, then canonical fallback."""
    name = COLUMN_MAPPING.get(key, key)
    return name if name in clin.columns else (key if key in clin.columns else None)


def require_column(clin, key, label):
    col = col_for(clin, key)
    if not col:
        st.error(f"Missing {label} column. Edit COLUMN_MAPPING['{key}'] in config.py to match a detected clinical column.")
    return col


def target_series(clin, key):
    col = require_column(clin, key, key.replace("_", " "))
    if not col:
        return None, None
    mapping = encode_target(clin[col].dropna())
    y = clin.set_index("sample")[col].astype(str).map(mapping)
    return y.dropna().astype(int), mapping


def resistance_inputs(clin, design, target, target_values=None):
    """Use only baseline expression rows and return patient IDs for grouped CV."""
    patient_col = require_column(clin, "patient_id", "patient ID")
    time_col = require_column(clin, "timepoint", "timepoint")
    if not patient_col or not time_col:
        return None
    timepoint = clin[time_col].astype(str).str.strip().str.lower()
    is_pre = timepoint.str.contains(r"pre|baseline|biopsy", regex=True) | timepoint.isin(["b", "before"])
    if not is_pre.any():
        st.error("No pre-treatment samples found. Map config.py timepoint to a field with pre/baseline/biopsy values.")
        return None
    values = clin[target] if target_values is None else pd.Series(target_values, index=clin.index)
    rows = clin.loc[is_pre & values.notna()].copy()
    rows["_target_for_model"] = values.loc[rows.index]
    rows = rows.set_index("sample")
    if rows.empty:
        st.error("No labeled pre-treatment samples are available for resistance prediction.")
        return None
    mapping = encode_target(rows["_target_for_model"])
    y = rows["_target_for_model"].astype(str).map(mapping).astype(int)
    X = design.reindex(rows.index)
    groups = rows[patient_col]
    return X, y, groups, rows, mapping


def selected_resistance_target(clin, choice, nri_threshold):
    """Return the configured recurrence target or a thresholded NRI target."""
    if choice == "NRI threshold":
        nri_col = next((col for col in clin.columns if "nri" in str(col).lower()), None)
        if not nri_col:
            st.error("No NRI column was detected. Check the series matrix or update config.py.")
            return None, None
        nri = pd.to_numeric(clin[nri_col], errors="coerce")
        labels = pd.Series(np.where(nri <= nri_threshold, "resistant", "responsive"), index=clin.index, dtype=object)
        labels[nri.isna()] = np.nan
        return nri_col, labels
    return require_column(clin, "response", "recurrence/resistance"), None


st.title("ONCOPRED")
st.caption("AI-assisted breast cancer analysis · GSE191127 · exploratory research dashboard")
with st.sidebar:
    st.header("Data & navigation")
    with st.expander("Advanced settings", expanded=False):
        demo = st.toggle("Use demo data", value=False)
        seed = st.number_input("Random seed", min_value=0, max_value=999999, value=42, step=1)
    page = st.radio("Page", ["Overview", "Molecular Explorer", "Treatment-Resistance Predictor", "Disease-Progression Analyser", "AI Insights"])
    try:
        _, clinical, _, detected, is_demo = cached_data(demo, int(seed))
        with st.expander("About this data & limitations"):
            if is_demo:
                st.caption("DEMO DATA — NOT REAL")
            else:
                st.caption("Real GEO files loaded")
            st.markdown("**Detected source columns**")
            st.code(", ".join(map(str, detected)) if detected else "none")
            st.caption("Adjust COLUMN_MAPPING in config.py if a clinical field is mapped incorrectly.")
            st.caption("Small cohort, expression-focused analysis, and exploratory predictions only. Associations do not establish causation; not for clinical use.")
    except Exception as exc:
        st.error(f"Could not load data: {exc}")
        st.stop()

expression, clinical, mutations, detected, is_demo = cached_data(demo, int(seed))
if is_demo:
    st.warning("DEMO DATA - NOT REAL", icon="⚠️")

if "sample" not in clinical.columns:
    clinical["sample"] = expression.columns[:len(clinical)]
clinical = clinical.drop_duplicates("sample").copy()
clinical = clinical[clinical["sample"].isin(expression.columns)].copy()
design = make_design(expression, clinical)
gene_symbols = {}
gene_map_path = Path("data") / "gene_map.csv"
if gene_map_path.exists():
    try:
        mapping_frame = pd.read_csv(gene_map_path)
        if not {"ensembl_id", "symbol"}.issubset(mapping_frame.columns) and mapping_frame.shape[1] == 1:
            mapping_frame = pd.read_csv(gene_map_path, quoting=csv.QUOTE_NONE)
        mapping_frame.columns = [str(column).strip().strip('"') for column in mapping_frame.columns]
        for column in ("ensembl_id", "symbol"):
            if column in mapping_frame:
                mapping_frame[column] = mapping_frame[column].astype(str).str.strip().str.strip('"')
        if {"ensembl_id", "symbol"}.issubset(mapping_frame.columns):
            mapping_frame = mapping_frame.dropna(subset=["ensembl_id", "symbol"]).drop_duplicates("ensembl_id")
            gene_symbols = dict(zip(mapping_frame["ensembl_id"].astype(str), mapping_frame["symbol"].astype(str)))
        else:
            st.caption("The gene symbol file is missing its expected fields; Ensembl IDs will be shown.")
    except Exception as exc:
        st.caption(f"The gene symbol file could not be read; Ensembl IDs will be shown ({exc}).")


def gene_label(gene):
    """Show a mapped symbol alongside its Ensembl ID, when available."""
    gene = str(gene)
    symbol = gene_symbols.get(gene)
    return f"{symbol} [{gene}]" if symbol else gene


def feature_label(feature):
    """Use symbols for gene features and friendly names for clinical dummies."""
    feature = str(feature)
    if feature in gene_symbols or feature.startswith("ENSG"):
        return gene_label(feature)
    if feature.startswith("treatment_"):
        return f"Treatment regimen: {feature.removeprefix('treatment_')}"
    if feature.startswith("subtype_"):
        return f"Molecular subtype: {feature.removeprefix('subtype_')}"
    return feature


def show_gene_importance(scores, title):
    """Render the 15 largest gene-only importances with the largest at the top."""
    top = scores.head(15).sort_values(ascending=True)
    chart_data = pd.DataFrame({"Gene": [gene_label(g) for g in top.index], "Importance": top.to_numpy()})
    fig = px.bar(chart_data, x="Importance", y="Gene", orientation="h", title=title)
    fig.update_layout(xaxis_title="Importance", yaxis_title="Gene")
    fig.update_yaxes(categoryorder="array", categoryarray=chart_data["Gene"].tolist())
    show_chart(fig, "Bars average importance measured on held-out patient folds; larger values indicate a larger score drop when permuted.")


model_context = (bool(is_demo), int(seed), tuple(expression.columns.astype(str)))
if st.session_state.get("model_context") != model_context:
    st.session_state.pop("resistance_result", None)
    st.session_state.pop("progression_result", None)
    st.session_state["model_context"] = model_context

if page == "Overview":
    page_intro("Overview", "Cohort size, treatment outcomes, molecular subtype, and sample-level expression patterns.")
    pair_df, pair_ids = paired_deltas(expression, clinical)
    n_patients = clinical["patient_id"].nunique() if "patient_id" in clinical else clinical["sample"].nunique()
    if not pair_df.empty:
        average_shift = pair_df.mean(axis=1)
        shifted_gene = average_shift.abs().idxmax()
        shift_takeaway = f"Largest mean paired shift: {gene_label(shifted_gene)} ({average_shift.loc[shifted_gene]:+.2f} log2 units)."
    else:
        shift_takeaway = "No matched pre/post pairs are available for a paired expression summary."
    with st.container(border=True):
        st.subheader("Key takeaways")
        st.markdown(
            f"- Cohort: **{len(clinical)} samples** from **{n_patients} patients**.\n"
            f"- Matched pre/post pairs: **{pair_df.shape[1]}**.\n"
            f"- {shift_takeaway}"
        )
    a, b, c, d, e = st.columns(5)
    a.metric("Samples", len(clinical)); b.metric("Patients", n_patients); c.metric("Matched pairs", pair_df.shape[1]); d.metric("Genes", expression.shape[0])
    rcol = col_for(clinical, "response")
    e.metric("Outcome classes", f"{clinical[rcol].nunique()}" if rcol else "not mapped")
    l, r = st.columns(2)
    tcol = col_for(clinical, "treatment")
    scol = col_for(clinical, "subtype")
    if tcol:
        treatment_counts = clinical[tcol].value_counts().rename_axis("Treatment regimen").reset_index(name="Samples")
        show_chart(px.bar(treatment_counts, x="Treatment regimen", y="Samples", title="Treatment distribution"), "Each bar counts samples by regimen.")
    if rcol:
        outcome_counts = friendly_values(clinical[rcol], rcol).value_counts().rename_axis("Outcome").reset_index(name="Samples")
        show_chart(px.pie(outcome_counts, names="Outcome", values="Samples", title="Treatment outcome distribution"), "Slices show the share of samples in each outcome group.")
    if scol:
        subtype_counts = clinical[scol].value_counts().rename_axis("Molecular subtype").reset_index(name="Samples")
        show_chart(px.bar(subtype_counts, x="Molecular subtype", y="Samples", title="Molecular subtype distribution"), "Each bar counts samples by subtype.")
    if expression.shape[1] >= 3:
        vals = StandardScaler().fit_transform(expression.T)
        xy = PCA(n_components=2, random_state=int(seed)).fit_transform(vals)
        scatter = clinical.set_index("sample").reindex(expression.columns).copy()
        scatter["sample"] = expression.columns
        scatter = scatter.reset_index(drop=True)
        scatter["PC1"], scatter["PC2"] = xy[:, 0], xy[:, 1]
        color = col_for(clinical, "timepoint") or rcol or tcol
        if color:
            scatter["Color group"] = friendly_values(scatter[color], color)
        fig = px.scatter(scatter, x="PC1", y="PC2", color="Color group" if color else None, hover_name="sample",
                         title=f"PCA by {friendly_column(color) if color else 'sample'}")
        fig.update_layout(xaxis_title="Principal component 1", yaxis_title="Principal component 2", legend_title=friendly_column(color) if color else "")
        show_chart(fig, "Nearby points have more similar expression profiles; color identifies the selected clinical group.")

elif page == "Molecular Explorer":
    page_intro("Molecular Explorer", "Compare paired pre/post expression shifts and optionally explore a separate mutation table.")
    st.caption("GSE191127 provides expression profiles; paired shifts describe expression change, not mutation evolution.")
    deltas, pair_ids = paired_deltas(expression, clinical)
    volcano_tab, top_tab, heatmap_tab, lookup_tab, mutation_tab = st.tabs(["Volcano", "Top genes", "Heatmap", "Gene lookup", "Mutation table"])
    if deltas.empty:
        with volcano_tab:
            st.info("No matched pre/post pairs were found. Check the patient and treatment-timepoint mapping.")
        with top_tab:
            st.info("Top shifted genes need matched pre/post pairs.")
        with heatmap_tab:
            st.info("The paired shift heatmap needs matched pre/post pairs.")
        with lookup_tab:
            st.info("Paired gene lookup needs matched pre/post pairs.")
    else:
        means = deltas.mean(axis=1)
        pvals = []
        for gene in deltas.index:
            x = deltas.loc[gene].dropna().to_numpy()
            pvals.append(float(ttest_1samp(x, 0).pvalue) if len(x) > 1 else 1.0)
        order = np.argsort(pvals); ranked = np.asarray(pvals)[order]
        q = np.empty(len(pvals)); q[order] = np.minimum.accumulate((ranked * len(ranked) / np.arange(1, len(ranked)+1))[::-1])[::-1]
        result = pd.DataFrame({"gene": deltas.index, "mean_delta": means.values, "p_value": pvals, "FDR": q}).sort_values("mean_delta")
        result["gene_label"] = result["gene"].map(gene_label)
        result["significance"] = pd.Categorical(np.where(result.FDR < .05, "FDR < 0.05", "Not significant"), categories=["FDR < 0.05", "Not significant"])
        result["minus_log10_p"] = -np.log10(np.maximum(result.p_value, 1e-300))
        with volcano_tab:
            fig = px.scatter(result, x="mean_delta", y="minus_log10_p", color="significance", hover_name="gene_label",
                             hover_data={"mean_delta": ":.3f", "p_value": ":.3g", "FDR": ":.3g"},
                             title="Paired expression shifts (post − pre)",
                             color_discrete_map={"FDR < 0.05": PLOTLY_PALETTE[0], "Not significant": PLOTLY_PALETTE[1]},
                             labels={"significance": "FDR status"})
            fig.update_layout(xaxis_title="Mean expression change (post − pre)", yaxis_title="−log10 p-value", legend_title="FDR status")
            show_chart(fig, "Higher points have smaller p-values; color marks whether the multiple-testing FDR is below 0.05.")
        with top_tab:
            st.subheader("Most increased and decreased genes")
            left, right = st.columns(2)
            left.dataframe(result.nlargest(15, "mean_delta")[["gene", "gene_label", "mean_delta", "p_value", "FDR"]], use_container_width=True)
            right.dataframe(result.nsmallest(15, "mean_delta")[["gene", "gene_label", "mean_delta", "p_value", "FDR"]], use_container_width=True)
            st.caption("How to read this: positive changes are higher after treatment; negative changes are lower after treatment.")
        with heatmap_tab:
            top = result.assign(abs_delta=result.mean_delta.abs()).nlargest(min(30, len(result)), "abs_delta").gene
            heatmap = deltas.loc[top].copy()
            heatmap.index = [gene_label(gene) for gene in heatmap.index]
            fig = px.imshow(heatmap, aspect="auto", color_continuous_scale=CONTINUOUS_PALETTE, title="Largest paired shifts across patients",
                            labels={"x": "Matched patient", "y": "Gene", "color": "Post − pre"})
            fig.update_layout(xaxis_title="Matched patient", yaxis_title="Gene")
            fig.update_coloraxes(colorbar_title_text="Post − pre")
            show_chart(fig, "Each cell is post-treatment minus pre-treatment expression for one matched patient.")
        with lookup_tab:
            gene = st.selectbox("Select a gene", expression.index.astype(str), index=0, format_func=gene_label)
            paired = []
            meta = clinical.set_index("sample")
            for pid, grp in meta.groupby("patient_id"):
                pre = grp[grp.timepoint.astype(str).str.lower().str.contains("pre|baseline")]
                post = grp[grp.timepoint.astype(str).str.lower().str.contains("post|after")]
                if len(pre) and len(post): paired.append({"patient": pid, "Pre-treatment": expression.loc[gene, pre.index[0]], "Post-treatment": expression.loc[gene, post.index[0]]})
            if paired:
                fig = go.Figure()
                for row in paired:
                    fig.add_trace(go.Scatter(x=["Pre-treatment", "Post-treatment"], y=[row["Pre-treatment"], row["Post-treatment"]],
                                             mode="lines+markers", name=str(row["patient"]), hovertemplate="Patient %{fullData.name}<br>%{x}: %{y:.2f}<extra></extra>"))
                fig.update_layout(title=f"Paired expression: {gene_label(gene)}", showlegend=False,
                                  xaxis_title="Treatment timepoint", yaxis_title="Normalized log2 expression")
                show_chart(fig, "Each line connects one patient's pre- and post-treatment samples.")
            else:
                st.info("No paired samples are available for this gene.")
    with mutation_tab:
        st.subheader("Optional mutation table")
        st.caption("GSE191127 expression counts do not include mutation calls. Upload a separate table to explore mutations.")
        uploaded = st.file_uploader("Mutation CSV with sample, gene, variant type, and VAF fields", type=["csv"])
        if uploaded is not None: mutations = pd.read_csv(uploaded)
        if not mutations.empty:
            if is_demo: st.caption("Displayed mutation records are synthetic demo data.")
            need = {"sample", "gene", "variant_type", "VAF"}
            if need.issubset(mutations.columns):
                left, right = st.columns(2)
                mutation_counts = mutations.gene.value_counts().head(20).rename_axis("Gene").reset_index(name="Mutation count")
                variant_counts = mutations.variant_type.value_counts().rename_axis("Variant type").reset_index(name="Mutation count")
                with left:
                    show_chart(px.bar(mutation_counts, x="Gene", y="Mutation count", title="Most frequently mutated genes"), "Bars count mutation records by gene.")
                with right:
                    show_chart(px.pie(variant_counts, names="Variant type", values="Mutation count", title="Variant type breakdown"), "Slices show the share of mutation records by variant type.")
                pivot = mutations.assign(value=1).pivot_table(index="sample", columns="gene", values="value", aggfunc="max", fill_value=0)
                fig = px.imshow(pivot, aspect="auto", title="Sample-by-gene mutation matrix",
                                labels={"x": "Gene", "y": "Sample", "color": "Mutation present"})
                fig.update_layout(xaxis_title="Gene", yaxis_title="Sample")
                fig.update_coloraxes(colorbar_title_text="Mutation present")
                show_chart(fig, "A value of 1 marks a mutation record for that sample and gene.")
            else: st.error("Mutation CSV must include sample, gene, variant type, and VAF fields.")

elif page == "Treatment-Resistance Predictor":
    page_intro("Treatment-Resistance Predictor", "Estimate treatment outcome from baseline expression while holding out whole patients during validation.")
    st.warning("Exploratory estimates from a small cohort; not for clinical decisions.")
    st.caption("Only pre-treatment samples are features; patient groups stay together in every stratified validation fold.")
    with st.sidebar:
        target_options = ["Recurrence (proxy)", "NRI threshold"]
        remembered_target = st.session_state.get("resistance_target_mode", target_options[0])
        target_choice = st.selectbox("Resistance target", target_options, index=target_options.index(remembered_target), key="resistance_target_widget")
        nri_threshold = st.number_input("NRI cutoff (≤ is resistant)", min_value=0.0, max_value=1.0,
                                        value=float(st.session_state.get("nri_threshold_value", .25)), step=.05,
                                        disabled=target_choice != "NRI threshold", key="nri_threshold_widget")
    st.session_state["resistance_target_mode"] = target_choice
    st.session_state["nri_threshold_value"] = float(nri_threshold)
    target, target_values = selected_resistance_target(clinical, target_choice, nri_threshold)
    if target:
        if target_choice == "Recurrence (proxy)":
            st.caption("Target: Recurrence (progression proxy). This is an exploratory proxy for treatment resistance.")
        else:
            st.caption(f"Target: Neoadjuvant Response Index (NRI) split at {nri_threshold:.2f}; values at or below the cutoff are labeled lower-response.")
        try:
            inputs = resistance_inputs(clinical, design, target, target_values)
            if inputs is None:
                st.stop()
            X, y, groups, usable, mapping = inputs
            with st.sidebar:
                model_kind = st.selectbox("Resistance model", ["Logistic Regression", "Random Forest"])
                top_n = st.selectbox("Top genes", [50, 100, 200], index=0)
            result = cross_validate_model(X, y, groups, model_kind, top_n, int(seed))
            shuffle_result = cached_label_shuffle_test(X, y, groups, result["pooled_auc"], model_kind, int(top_n), int(seed))
            m_pool, m_perm, m_acc, m_base = st.columns(4)
            m_pool.metric("Pooled OOF ROC-AUC", f"{result['pooled_auc']:.3f}")
            m_perm.metric("Patient-level permutation p", f"{shuffle_result['p_value']:.3f}")
            m_acc.metric("OOF model accuracy", f"{result['accuracy']:.3f}")
            m_base.metric("Majority-class baseline", f"{result['baseline_accuracy']:.3f}")
            st.caption(f"Empirical p-value from {shuffle_result['valid_shuffles']} patient-level label shuffles.")
            per_fold_reliable = all(pos >= 2 and neg >= 2 for pos, neg in zip(result["fold_positive_counts"], result["fold_negative_counts"]))
            if per_fold_reliable:
                st.caption(f"per-fold (unstable): Mean fold ROC-AUC {result['auc_mean']:.3f} ± {result['auc_std']:.3f}")
            else:
                st.caption("Per-fold ROC-AUC omitted: at least one fold has fewer than 2 positive or 2 negative samples.")
            if target_choice == "NRI threshold":
                negative_label, positive_label = f"NRI > {nri_threshold:.2f}", f"NRI ≤ {nri_threshold:.2f}"
            else:
                negative_label, positive_label = "No recurrence", "Recurrence (proxy)"
            st.caption("Out-of-fold confusion matrix")
            st.dataframe(pd.DataFrame(result["confusion"], index=[f"Actual {negative_label}", f"Actual {positive_label}"],
                                      columns=[f"Predicted {negative_label}", f"Predicted {positive_label}"]))
            fpr, tpr = [], []
            for threshold in np.linspace(0, 1, 101):
                fpr.append(np.mean(result["probabilities"][result["y"].to_numpy() == 0] >= threshold))
                tpr.append(np.mean(result["probabilities"][result["y"].to_numpy() == 1] >= threshold))
            fig = go.Figure(go.Scatter(x=fpr, y=tpr, mode="lines", name="OOF ROC",
                                       hovertemplate="False positive rate: %{x:.2f}<br>True positive rate: %{y:.2f}<extra></extra>"))
            fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(dash="dash", color=PLOTLY_PALETTE[2]))
            fig.update_layout(xaxis_title="False positive rate", yaxis_title="True positive rate", title="Out-of-fold ROC curve")
            show_chart(fig, "The curve summarizes held-out predictions; a curve nearer the upper-left has better discrimination.")
            sample = st.selectbox("Predict for a sample", X.index.astype(str))
            oof_probabilities = pd.Series(result["probabilities"], index=X.index)
            prob = float(oof_probabilities.loc[sample])
            gauge_title = "Out-of-fold recurrence risk (resistance proxy)" if target_choice == "Recurrence (proxy)" else "Out-of-fold lower-response probability (NRI)"
            gauge = go.Figure(go.Indicator(mode="gauge+number", value=prob*100, title={"text":gauge_title}, number={"suffix":"%"},
                                           gauge={"axis":{"range":[0,100]}, "bar":{"color":PLOTLY_PALETTE[0]}, "bgcolor":"#263241", "bordercolor":"#566274"}))
            show_chart(gauge, "This value comes from the fold that held out the selected patient's samples.")
            st.caption(f"A model fitted on all {groups.nunique()} patients would be overconfident; this gauge uses a held-out-patient prediction.")
            result["target_mode"] = target_choice
            result["target_name"] = target
            result["nri_threshold"] = float(nri_threshold)
            result["kind"] = model_kind
            result["n_features"] = int(top_n)
            st.session_state["resistance_result"] = result
        except Exception as exc: st.error(f"Could not train resistance model: {exc}")

elif page == "Disease-Progression Analyser":
    page_intro("Disease-Progression Analyser", "Rank samples by out-of-fold progression risk and inspect survival data when available.")
    st.warning("Exploratory risk estimates from a small cohort; not for clinical use.")
    st.caption("Each risk score is out-of-fold; that patient's samples were excluded from the model that scored them.")
    target = require_column(clinical, "progression", "progression/recurrence")
    if target:
        try:
            usable = clinical.dropna(subset=[target]).set_index("sample")
            vals = usable[target]
            if vals.dtype == object: mapping = encode_target(vals); y = vals.astype(str).map(mapping).astype(int)
            else: y = vals.astype(int)
            patient_col = require_column(clinical, "patient_id", "patient ID")
            if not patient_col:
                st.stop()
            outcome_tokens = ("recurr", "nri", "progress", "response", "event", "outcome", "survival", "followup")
            excluded = {target, "response", "progression", "recurrence", "event", "followup_time", "survival_time", "outcome"}
            excluded.update(col for col in clinical.columns if any(token in str(col).lower() for token in outcome_tokens))
            excluded.update(key for key, mapped in COLUMN_MAPPING.items() if mapped == target)
            progress_design = make_design(expression, clinical, exclude_clinical=excluded)
            X = progress_design.reindex(usable.index)
            blocked = {str(col).lower() for col in excluded if col}
            leakage_features = [feature for feature in X.columns if any(str(feature).lower() == col or str(feature).lower().startswith(col + "_") for col in blocked)]
            assert target not in X.columns and not leakage_features, f"Outcome leakage in progression features: {leakage_features or target}"
            groups = usable[patient_col]
            cv_result = cross_validate_model(X, y, groups, "Logistic Regression", 50, int(seed))
            risk = cv_result["probabilities"]
            shuffle_result = cached_label_shuffle_test(X, y, groups, cv_result["pooled_auc"], "Logistic Regression", 50, int(seed))
            st.metric("Pooled OOF ROC-AUC", f"{cv_result['pooled_auc']:.3f}", delta=f"patient shuffle p={shuffle_result['p_value']:.3f}", delta_color="off")
            st.caption(f"Empirical p-value from {shuffle_result['valid_shuffles']} patient-level label shuffles.")
            feature_names = [feature_label(feature) for feature in X.columns]
            with st.expander(f"Features used in the model ({len(feature_names)})"):
                st.dataframe(pd.DataFrame({"Feature": feature_names}), hide_index=True, use_container_width=True, height=260)
            scored = usable.copy(); scored["risk"] = risk
            if len(scored) < 3:
                raise ValueError("At least three labeled samples are needed to create tertile risk groups.")
            scored["risk_group"] = pd.qcut(scored.risk.rank(method="first"), 3, labels=["Low", "Medium", "High"])
            risk_colors = {"Low": PLOTLY_PALETTE[0], "Medium": PLOTLY_PALETTE[1], "High": PLOTLY_PALETTE[3]}
            risk_hist = px.histogram(scored, x="risk", color="risk_group", title="Out-of-fold progression risk",
                                     color_discrete_map=risk_colors, category_orders={"risk_group": ["Low", "Medium", "High"]})
            risk_hist.update_layout(xaxis_title="Out-of-fold progression risk", yaxis_title="Sample count", legend_title="Risk group")
            show_chart(risk_hist, "Bars show the number of samples at each held-out risk score; color marks ranked tertiles.")
            tcol = col_for(clinical, "followup_time")
            event_col = "event" if "event" in clinical else target
            followup = pd.to_numeric(usable[tcol], errors="coerce") if tcol else pd.Series(dtype=float)
            if tcol and followup.notna().any():
                fig = go.Figure()
                for group in ["Low", "Medium", "High"]:
                    subset = scored[scored.risk_group == group]
                    times = pd.to_numeric(subset[tcol], errors="coerce").dropna().to_numpy()
                    events = pd.to_numeric(subset[event_col], errors="coerce").fillna(0).to_numpy() if event_col in subset else np.ones(len(times))
                    if len(times):
                        order = np.argsort(times); times = times[order]; events = events[order]
                        surv, current = [1.0], 1.0
                        event_times = np.unique(times[events > 0])
                        for t in event_times:
                            at_risk = np.sum(times >= t); deaths = np.sum((times == t) & (events > 0))
                            if at_risk: current *= (1 - deaths / at_risk)
                            surv.append(current)
                        ticks = np.r_[0, event_times]
                        fig.add_trace(go.Scatter(x=ticks, y=surv, mode="lines", line_shape="hv", name=group,
                                                 line={"color": risk_colors[group]},
                                                 hovertemplate="Follow-up time: %{x}<br>Event-free fraction: %{y:.2f}<extra>%{fullData.name}</extra>"))
                fig.update_layout(title="Kaplan–Meier-style event-free curve (manual estimate)", xaxis_title="Follow-up time", yaxis_title="Estimated event-free fraction")
                show_chart(fig, "Step drops mark observed events; curves compare the ranked risk groups.")
            else: st.caption("No follow-up time exists in the loaded clinical data, so survival curves are unavailable.")
            ordered = scored.sort_values("risk", ascending=False).reset_index()
            table_cols = list(dict.fromkeys(["sample", patient_col, target, "risk", "risk_group"]))
            if "recurrence" in str(target).lower():
                ordered[target] = friendly_values(ordered[target], target)
            ordered = ordered[table_cols].rename(columns={"sample": "Sample", patient_col: "Patient ID", target: friendly_column(target),
                                                          "risk": "Out-of-fold progression risk", "risk_group": "Risk group"})
            st.dataframe(ordered, use_container_width=True)
            st.download_button("Download risk table CSV", ordered.to_csv(index=False), "oncopred_progression_risk.csv", "text/csv")
            st.session_state["progression_result"] = {**cv_result, "X": X, "y": y, "groups": groups, "target_name": target}
        except Exception as exc: st.error(f"Could not analyse progression: {exc}")

else:
    page_intro("AI Insights", "See which genes most affect held-out predictions and inspect one sample's model contributions.")
    st.warning("Exploratory associations only; not for clinical use.")
    st.caption("Treatment and other clinical dummy columns are excluded from permutation importance because the sample is small.")
    trained = st.session_state.get("resistance_result")
    if trained is None:
        target_choice = st.session_state.get("resistance_target_mode", "Recurrence (proxy)")
        nri_threshold = float(st.session_state.get("nri_threshold_value", .25))
        target, target_values = selected_resistance_target(clinical, target_choice, nri_threshold)
        if target:
            try:
                inputs = resistance_inputs(clinical, design, target, target_values)
                if inputs is None:
                    st.stop()
                X, y, groups, rows, mapping = inputs
                trained = cross_validate_model(X, y, groups, "Logistic Regression", 50, int(seed))
                trained["kind"] = "Logistic Regression"
                trained["n_features"] = 50
                trained["target_mode"] = target_choice
                trained["target_name"] = target
                trained["nri_threshold"] = nri_threshold
                st.session_state["resistance_result"] = trained
            except Exception as exc: st.warning(f"Resistance model unavailable: {exc}")
    if trained:
        try:
            gene_cols = [col for col in trained["X"].columns if str(col) in set(expression.index.astype(str))]
            imp = grouped_oof_permutation_scores(trained["X"][gene_cols], trained["y"], trained["groups"], int(seed),
                                                 trained.get("kind", "Logistic Regression"), trained.get("n_features", 50), repeats=3)
            show_gene_importance(imp, "Resistance model gene-only grouped OOF permutation importance")
            sample = st.selectbox("Sample for contribution view", trained["X"].index.astype(str))
            patient_id = trained["groups"].loc[sample]
            fold_index = next(i for i, fold in enumerate(trained["fold_patient_ids"]) if patient_id in fold["test"])
            oof_model = trained["fold_models"][fold_index]
            oof_prob = float(pd.Series(trained["probabilities"], index=trained["X"].index).loc[sample])
            gene_cols = [col for col in trained["X"].columns if str(col) in set(expression.index.astype(str))]
            contrib = sample_gene_contributions(oof_model, trained["X"].loc[sample], gene_cols).head(15).sort_values()
            if not contrib.empty:
                contribution_data = pd.DataFrame({"Gene": [gene_label(g) for g in contrib.index], "Contribution": contrib.values})
                contribution_fig = px.bar(contribution_data, x="Contribution", y="Gene", orientation="h", color=contrib.values > 0,
                                          title="Approximate gene contribution to the positive-class estimate")
                contribution_fig.update_layout(xaxis_title="Contribution toward positive target", yaxis_title="Gene", showlegend=False)
                show_chart(contribution_fig, "Positive values push the held-out estimate toward the positive target; negative values push it away.")
                toward = contrib[contrib > 0].nlargest(2)
                away = contrib[contrib < 0].nsmallest(2)
                if trained.get("target_mode", "Recurrence (proxy)") == "Recurrence (proxy)":
                    positive_direction, negative_direction = "recurrence", "no recurrence"
                    probability_label = "recurrence-proxy probability"
                else:
                    cutoff = trained.get("nri_threshold", .25)
                    positive_direction, negative_direction = f"lower response (NRI ≤ {cutoff:.2f})", f"higher response (NRI > {cutoff:.2f})"
                    probability_label = f"lower-response NRI probability (cutoff {cutoff:.2f})"
                predicted_direction = positive_direction if oof_prob >= .5 else negative_direction
                st.write(f"The out-of-fold prediction is **{oof_prob:.1%}** {probability_label}; the model leans toward **{predicted_direction}**.")
                st.write(f"Top genes pushing toward {positive_direction}: {', '.join(gene_label(g) for g in toward.index) or 'none'}.")
                st.write(f"Top genes pushing away from {positive_direction}: {', '.join(gene_label(g) for g in away.index) or 'none'}.")
            else: st.info("No selected gene features produced a sample-specific contribution for this model.")
        except Exception as exc: st.warning(f"Could not calculate explanations: {exc}")
    progression = st.session_state.get("progression_result")
    if progression:
        try:
            gene_cols = [col for col in progression["X"].columns if str(col) in set(expression.index.astype(str))]
            imp = grouped_oof_permutation_scores(progression["X"][gene_cols], progression["y"], progression["groups"], int(seed), repeats=3)
            show_gene_importance(imp, "Progression model gene-only grouped OOF permutation importance")
        except Exception as exc: st.info(f"Progression importance unavailable: {exc}")

