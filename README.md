# ONCOPRED

An exploratory Streamlit dashboard for expression and clinical annotations from GEO series GSE191127. It runs on generated demo data on first launch.

## Run

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

The sidebar has a **Use demo data** switch (on by default) and a reproducible random seed. Demo mode is clearly marked in the app.

## Real data

Place one GEO series matrix (`*_series_matrix.txt[.gz]`) and one expression matrix (CSV, TSV, or TXT, optionally gzipped; genes in rows, samples in columns) in `data/`. The loader parses `!Sample_characteristics_ch1`, aligns GEO sample titles to expression column names, and derives patient ID and pre/post timepoint where possible. Check detected columns in the sidebar and edit `COLUMN_MAPPING` in `config.py` to match your annotations. The bundled GSE191127 mapping uses recurrence as an exploratory binary resistance proxy; its NRI field is ordinal and is not thresholded. Expression counts are CPM-normalized and log2 transformed.

The resistance predictor uses pre-treatment samples only. Its sidebar lets you select recurrence (shown as an exploratory resistance proxy) or binarise NRI at a user-selected cutoff (NRI at or below the cutoff is the resistant group). It evaluates outcomes with stratified patient-grouped cross-validation, reports both per-fold and pooled out-of-fold AUC, and compares accuracy with the majority-class baseline.

Progression scores are out-of-fold predictions from patient-grouped cross-validation, so every patient's samples are held out together. The bundled clinical files have no follow-up time; survival curves are unavailable. AI Insights reports gene-only permutation importance measured on held-out patient folds. To show gene symbols alongside Ensembl IDs, add `data/gene_map.csv` with `ensembl_id` and `symbol` columns.

## Limitations

- Small sample sizes make cross-validation and individual risk scores unstable; the demo is synthetic and is not clinical evidence.
- GSE191127 expression is not mutation sequencing. Mutation charts are synthetic in demo mode or require a separately supplied mutation table.
- Outcomes, treatment, and pairing depend on the quality and format of the GEO annotations and the editable mapping.
- The analyses show associations, not causation, and are for exploratory research only. They are not for clinical use.
