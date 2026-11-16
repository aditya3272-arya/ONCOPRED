"""Editable clinical column mapping used after GEO metadata parsing."""

COLUMN_MAPPING = {
    "patient_id": "patient_id",
    "timepoint": "timepoint",
    "treatment": "chemotherapy",
    # GSE191127 provides recurrence as its binary outcome; this serves as an
    # exploratory resistance proxy because the NRI field is ordinal, not binary.
    "response": "recurrence",
    "progression": "recurrence",
    "followup_time": "followup_time",
    "subtype": "subtype",
}
