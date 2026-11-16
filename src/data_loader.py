"""Load GEO series metadata and expression matrices, or create deterministic demo data."""
from pathlib import Path
import gzip
import re
import numpy as np
import pandas as pd


def _open_text(path):
    return gzip.open(path, "rt", encoding="utf-8", errors="replace") if str(path).endswith(".gz") else open(path, encoding="utf-8", errors="replace")


def parse_series_matrix(path):
    """Parse selected GEO series-matrix sample lines into one row per sample."""
    sample_ids, sample_titles, characteristics = [], [], {}
    with _open_text(path) as handle:
        for line in handle:
            if line.startswith("!Sample_geo_accession"):
                sample_ids = [x.strip().strip('"') for x in line.split("\t")[1:]]
            elif line.startswith("!Sample_title"):
                sample_titles = [x.strip().strip('"') for x in line.split("\t")[1:]]
            elif line.startswith("!Sample_characteristics_ch1"):
                values = [x.strip().strip('"') for x in line.rstrip("\n").split("\t")[1:]]
                for i, value in enumerate(values):
                    if ":" in value:
                        key, val = value.split(":", 1)
                        characteristics.setdefault(key.strip().lower().replace(" ", "_"), [None] * len(values))[i] = val.strip()
                    elif value:
                        characteristics.setdefault("characteristic", [None] * len(values))[i] = value
    data = pd.DataFrame(characteristics)
    if sample_ids and len(sample_ids) == len(data):
        data.insert(0, "geo_accession", sample_ids)
        data.insert(0, "sample", sample_titles if len(sample_titles) == len(sample_ids) else sample_ids)
    return data


def _demo(seed=42):
    """Return synthetic expression, clinical, and mutation tables with weak outcome signals."""
    rng = np.random.default_rng(seed)
    n_patients, n_genes = 40, 500
    patients = [f"P{i+1:03d}" for i in range(n_patients)]
    response_patient = rng.binomial(1, .55, n_patients)
    progression_patient = rng.binomial(1, .32, n_patients)
    samples, rows = [], []
    for i, pid in enumerate(patients):
        times = ["pre", "post"] if i < 20 else ["pre"]
        for tp in times:
            sid = f"{pid}_{tp}"
            samples.append(sid)
            expr = rng.normal(5.2, 1.15, n_genes)
            # A few reproducible, moderately noisy marker signals keep demo models informative.
            expr[0:4] = rng.normal(5.2, .55, 4) + (1-response_patient[i]) * 1.0 + progression_patient[i] * .45
            expr[4:8] = rng.normal(5.2, .55, 4) + response_patient[i] * .75
            if tp == "post":
                expr[8:12] += .6
            rows.append(expr)
    genes = [f"GENE_{i+1:03d}" for i in range(n_genes)]
    expr = pd.DataFrame(np.asarray(rows).T, index=genes, columns=samples)
    clinical = pd.DataFrame({
        "sample": samples,
        "patient_id": [s.split("_")[0] for s in samples],
        "timepoint": [s.rsplit("_", 1)[1] for s in samples],
        "treatment": [rng.choice(["chemotherapy_A", "chemotherapy_B", "targeted_combo"]) for _ in samples],
        "response": ["responder" if response_patient[int(s[1:4])-1] else "resistant" for s in samples],
        "progression": [int(progression_patient[int(s[1:4])-1]) for s in samples],
        "followup_time": rng.integers(8, 72, len(samples)),
        "event": [int(progression_patient[int(s[1:4])-1]) for s in samples],
        "subtype": [rng.choice(["ER+", "ER-", "unknown"]) for _ in samples],
    })
    mutation_rows = []
    for sid in samples:
        for _ in range(rng.integers(0, 4)):
            mutation_rows.append({"sample": sid, "gene": rng.choice(["TP53", "PIK3CA", "GATA3", "ERBB2", "MAP3K1"]),
                                 "variant_type": rng.choice(["missense", "frameshift", "splice_site"]), "VAF": round(float(rng.uniform(.05, .65)), 3)})
    return expr, clinical, pd.DataFrame(mutation_rows)


def _find_mapping_column(frame, key, aliases):
    """Find a mapped clinical column, trying common GEO characteristic names."""
    from config import COLUMN_MAPPING
    wanted = [COLUMN_MAPPING.get(key, "").lower(), key.lower(), *aliases]
    cols = {str(c).lower().replace(" ", "_"): c for c in frame.columns}
    for name in wanted:
        name = name.lower().replace(" ", "_")
        for normalized, original in cols.items():
            if name and (name == normalized or name in normalized):
                return original
    return None


def load_data(data_dir="data", use_demo=False, seed=42):
    """Load real files if present; otherwise create demo data. Returns expression, clinical, mutations, detected."""
    folder = Path(data_dir)
    if use_demo or not folder.exists():
        expr, clinical, mutations = _demo(seed)
        return expr, clinical, mutations, list(clinical.columns), True
    matrix_files = list(folder.glob("*_series_matrix.txt")) + list(folder.glob("*_series_matrix.txt.gz"))
    expr_files = sorted(
        [p for p in folder.iterdir()
         if p.name.lower() != "gene_map.csv"
         and p.suffix.lower() in {".csv", ".tsv", ".txt", ".gz"}
         and "series_matrix" not in p.name.lower()],
        key=lambda p: p.name.lower(),
    )
    if not matrix_files or not expr_files:
        expr, clinical, mutations = _demo(seed)
        return expr, clinical, mutations, list(clinical.columns), True
    meta = parse_series_matrix(matrix_files[0])
    f = expr_files[0]
    sep = "," if ".csv" in f.name.lower() else "\t"
    expression = pd.read_csv(f, sep=sep, index_col=0, low_memory=False)
    annotation_columns = {"gene_biotype", "chromosome_name", "start_position", "end_position", "external_gene_id", "description"}
    expression = expression.drop(columns=[c for c in expression.columns if str(c).lower() in annotation_columns], errors="ignore")
    expression = expression.apply(pd.to_numeric, errors="coerce").fillna(0)
    if meta.empty:
        meta = pd.DataFrame({"sample": expression.columns})
    aliases = {"patient_id": ["patient", "subject_id", "patientid"], "timepoint": ["timepoint", "treatment_time", "sample_type"],
               "treatment": ["treatment", "drug", "therapy"], "response": ["response", "outcome", "responder"],
               "progression": ["progression", "recurrence", "event"], "followup_time": ["followup", "survival_time", "time"], "subtype": ["subtype", "er_status"]}
    for key, candidates in aliases.items():
        col = _find_mapping_column(meta, key, candidates)
        if col and col != key:
            meta[key] = meta[col]
    # GEO titles carry the patient/timepoint identity used in the count matrix (for example 97_B/97_S).
    if "patient_id" not in meta.columns and "sample" in meta.columns:
        meta["patient_id"] = meta["sample"].astype(str).str.extract(r"^(.+?)_[BS]$", flags=re.IGNORECASE)[0]
    if "timepoint" not in meta.columns:
        if "tissue_type" in meta.columns:
            tissue = meta["tissue_type"].astype(str).str.lower()
            timepoint = pd.Series(index=meta.index, dtype=object)
            timepoint.loc[tissue.str.contains("biopsy|baseline|pre")] = "pre"
            timepoint.loc[tissue.str.contains("surgery|post")] = "post"
            meta["timepoint"] = timepoint
        elif "sample" in meta.columns:
            suffix = meta["sample"].astype(str).str.extract(r"_([BS])$", flags=re.IGNORECASE)[0].str.upper()
            meta["timepoint"] = suffix.map({"B": "pre", "S": "post"})
    if "sample" not in meta.columns:
        meta["sample"] = expression.columns[:len(meta)]
    # The count matrix names append sequencing/library fields around the GEO sample title.
    title_to_column = {}
    titles = meta["sample"].astype(str).tolist()
    for column in expression.columns.astype(str):
        match = next((title for title in titles if re.search(rf"(?:^|_){re.escape(title)}(?:_|$)", column, re.IGNORECASE)), None)
        if match:
            title_to_column[column] = match
    expression = expression.rename(columns=title_to_column)
    meta = meta.set_index("sample", drop=False).reindex(expression.columns).reset_index(drop=True)
    detected = list(meta.columns)
    return expression, meta, pd.DataFrame(columns=["sample", "gene", "variant_type", "VAF"]), detected, False
