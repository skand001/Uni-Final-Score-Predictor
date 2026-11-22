"""Student Grade Predictor - Flask web app.

Run with:   python app.py
Then open:  http://127.0.0.1:5000

Needs grade_models.joblib (saved from the notebook) in the same folder.
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from flask import Flask, jsonify, render_template, request

BASE_DIR = Path(__file__).parent
MODELS = joblib.load(BASE_DIR / "grade_models.joblib")

if MODELS["sklearn_version"] != sklearn.__version__:
    print(f"WARNING: models were trained with scikit-learn {MODELS['sklearn_version']}, "
          f"but this server runs {sklearn.__version__}. Predictions may be unreliable.")

app = Flask(__name__)

# Allowed answers for each dropdown / choice field
CHOICES = {
    "course": ["Biomedical Science", "Computer Science", "Economics", "English Literature",
               "Law", "Mechanical Engineering", "Nursing", "Psychology"],
    "gender": ["Female", "Male", "Non-binary"],
    "fee_status": ["Home", "International"],
    "first_generation": ["No", "Yes"],
    "accommodation": ["Halls", "Private rented", "Family home"],
    "society_member": ["No", "Yes"],
}

# Same real-life limits as the cleaning rules in the notebook
LIMITS = {
    "year_of_study": (1, 3), "age": (17, 70), "commute_mins": (0, 120),
    "parttime_job_hrs": (0, 40), "ucas_tariff": (40, 250), "attendance_pct": (0, 100),
    "study_hrs_week": (0, 70), "sleep_hrs": (3, 12), "vle_logins_month": (0, 300),
    "late_submissions": (0, 20), "coursework_avg": (0, 100), "midterm_score": (0, 100),
}

FIELDS = {
    "enrolment": {
        "choices": ["course", "gender", "fee_status", "first_generation", "accommodation"],
        "numbers": ["year_of_study", "age", "commute_mins", "parttime_job_hrs", "ucas_tariff"],
    },
    "midyear": {
        "choices": ["course", "gender", "fee_status", "first_generation", "accommodation",
                    "society_member"],
        "numbers": ["year_of_study", "age", "commute_mins", "parttime_job_hrs", "ucas_tariff",
                    "attendance_pct", "study_hrs_week", "sleep_hrs", "vle_logins_month",
                    "late_submissions", "coursework_avg", "midterm_score"],
    },
}
CAN_BE_UNKNOWN = {"ucas_tariff", "attendance_pct"}   # the "not known" checkboxes


def degree_class(g):
    if g >= 70: return "First"
    if g >= 60: return "2:1"
    if g >= 50: return "2:2"
    if g >= 40: return "Third"
    return "Fail"


def build_row(data, mode):
    """Check the submitted answers and turn them into one row like the training data."""
    errors, row = [], {}

    for field in FIELDS[mode]["choices"]:
        value = data.get(field)
        if value not in CHOICES[field]:
            errors.append(f"{field}: must be one of {CHOICES[field]}")
        row[field] = value

    for field in FIELDS[mode]["numbers"]:
        value = data.get(field)
        if value is None and field in CAN_BE_UNKNOWN:
            row[field] = np.nan
            continue
        try:
            value = float(value)
        except (TypeError, ValueError):
            errors.append(f"{field}: must be a number")
            continue
        low, high = LIMITS[field]
        if not low <= value <= high:
            errors.append(f"{field}: must be between {low} and {high}")
        row[field] = value

    if errors:
        return None, errors

    # same extra columns as the notebook
    row["ucas_missing"] = int(np.isnan(row["ucas_tariff"]))
    if mode == "midyear":
        row["attendance_missing"] = int(np.isnan(row["attendance_pct"]))
        row["sleep_gap"] = abs(row["sleep_hrs"] - 7.5)
        row["study_hrs_sq"] = row["study_hrs_week"] ** 2
    return row, []


@app.get("/")
def index():
    return render_template("index.html", choices=CHOICES)


@app.post("/api/predict")
def predict():
    data = request.get_json(silent=True) or {}
    mode = data.get("mode")
    if mode not in MODELS or mode == "sklearn_version":
        return jsonify({"errors": ["mode: must be 'enrolment' or 'midyear'"]}), 400

    row, errors = build_row(data, mode)
    if errors:
        return jsonify({"errors": errors}), 400

    bundle = MODELS[mode]
    X_new = pd.DataFrame([row])[bundle["columns"]]   # this model's columns, in training order
    grade = float(np.clip(bundle["model"].predict(X_new)[0], 0, 100))
    mae = bundle["mae"]
    low, high = max(grade - mae, 0), min(grade + mae, 100)

    return jsonify({
        "mode": mode, "grade": round(grade, 1), "degree_class": degree_class(grade),
        "mae": mae, "low": round(low, 1), "high": round(high, 1),
        "low_class": degree_class(low), "high_class": degree_class(high),
    })


if __name__ == "__main__":
    app.run(debug=True)
