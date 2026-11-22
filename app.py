"""Student grade predictor - Streamlit app with two modes and light/dark styling.

Run with:  streamlit run app.py
Needs, in the same folder:
  grade_models.joblib       (saved from the notebook)
  .streamlit/config.toml    (the light/dark theme)
"""
import joblib
import numpy as np
import pandas as pd
import sklearn
import streamlit as st

st.set_page_config(page_title="Student Grade Predictor", page_icon="🎓", layout="centered")

# ---------- styling (works in both light and dark mode) ----------
st.markdown("""
<style>
.block-container {max-width: 880px; padding-top: 2.5rem; padding-bottom: 3rem;}
[data-testid="stDecoration"] {display: none;}
footer {visibility: hidden;}

.hero {margin-bottom: 1.5rem;}
.hero .badge {display: inline-block; font-size: .75rem; font-weight: 600; letter-spacing: .08em;
  text-transform: uppercase; padding: .3rem .75rem; border-radius: 999px;
  background: rgba(99,102,241,.12); color: #6366F1; margin-bottom: .75rem;}
.hero h1 {font-size: 2.6rem; font-weight: 800; letter-spacing: -.035em; line-height: 1.1; margin: 0;
  color: inherit;}  /* follows the theme: white in dark mode, black in light mode */
.hero p {opacity: .65; margin: .5rem 0 0; font-size: 1.02rem;}

div[data-testid="stButton"] button[kind="primary"],
button[data-testid="stBaseButton-primary"] {
  background: linear-gradient(90deg, #6366F1, #A855F7); border: 0; color: #fff;
  font-weight: 600; padding: .7rem 1rem; box-shadow: 0 10px 28px -10px rgba(99,102,241,.7);
  transition: transform .15s ease, box-shadow .15s ease;}
div[data-testid="stButton"] button[kind="primary"]:hover,
button[data-testid="stBaseButton-primary"]:hover {
  transform: translateY(-1px); box-shadow: 0 14px 32px -10px rgba(168,85,247,.75); color: #fff;}

.result-card {border: 1px solid rgba(128,128,128,.22); border-radius: 18px; padding: 1.6rem 1.8rem;
  margin-top: .5rem; background: linear-gradient(135deg, rgba(99,102,241,.10), rgba(236,72,153,.05));}
.result-top {display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 1rem;}
.result-label {font-size: .8rem; font-weight: 600; letter-spacing: .08em; text-transform: uppercase; opacity: .6;}
.grade {font-size: 3.6rem; font-weight: 800; letter-spacing: -.04em; line-height: 1;}
.pill {display: inline-block; padding: .45rem 1.1rem; border-radius: 999px; font-weight: 700; font-size: 1.05rem;}
.range-text {margin-top: 1.1rem; opacity: .75; font-size: .95rem;}

.track {position: relative; height: 12px; margin-top: 1.4rem; border-radius: 999px;
  background: linear-gradient(90deg, rgba(239,68,68,.25) 0 40%, rgba(249,115,22,.25) 40% 50%,
    rgba(245,158,11,.25) 50% 60%, rgba(99,102,241,.25) 60% 70%, rgba(16,185,129,.25) 70% 100%);}
.track .band {position: absolute; top: -3px; height: 18px; border-radius: 999px;
  background: linear-gradient(90deg, #6366F1, #A855F7); opacity: .55;}
.track .marker {position: absolute; top: -6px; width: 6px; height: 24px; border-radius: 3px;
  transform: translateX(-50%); background: currentColor;}
.ticks {position: relative; height: 1.2rem; margin-top: .45rem; font-size: .72rem; opacity: .55;}
.ticks span {position: absolute; transform: translateX(-50%); white-space: nowrap;}
.note {font-size: .85rem; opacity: .6; margin-top: .9rem;}
</style>
""", unsafe_allow_html=True)


@st.cache_resource  # load the models once, not on every click
def load_models():
    return joblib.load("grade_models.joblib")


saved = load_models()

if saved["sklearn_version"] != sklearn.__version__:
    st.warning(f"Models were trained with scikit-learn {saved['sklearn_version']}, "
               f"but this app is running {sklearn.__version__}. Predictions may be unreliable.")

st.markdown("""
<div class="hero">
<span class="badge">Machine learning demo</span>
<h1>Student Grade Predictor</h1>
<p>Estimate a final university grade from a student's profile and engagement.</p>
</div>
""", unsafe_allow_html=True)

# ---------- choose the mode ----------
mode_label = st.segmented_control(
    "When are you predicting?",
    ["At enrolment", "Mid-year"],
    default="Mid-year",
    help="At enrolment uses only what's known on day one, so it is less precise. "
         "Mid-year adds attendance, coursework and exam results.",
) or "Mid-year"   # if the user clicks the selected option off, fall back to Mid-year
mode = "enrolment" if mode_label == "At enrolment" else "midyear"
bundle = saved[mode]
model, columns, mae = bundle["model"], bundle["columns"], bundle["mae"]

# ---------- day-one inputs (both modes) ----------
with st.container(border=True):
    st.subheader("About the student")
    col1, col2 = st.columns(2)
    with col1:
        course = st.selectbox("Course", ["Biomedical Science", "Computer Science", "Economics",
                                         "English Literature", "Law", "Mechanical Engineering",
                                         "Nursing", "Psychology"])
        year_of_study = st.selectbox("Year of study", [1, 2, 3])
        age = st.number_input("Age", min_value=17, max_value=70, value=20)
        gender = st.selectbox("Gender", ["Female", "Male", "Non-binary"])
        fee_status = st.selectbox("Fee status", ["Home", "International"])
    with col2:
        first_generation = st.radio("First in family at university?", ["No", "Yes"], horizontal=True)
        accommodation = st.selectbox("Accommodation", ["Halls", "Private rented", "Family home"])
        commute_mins = st.slider("Commute (minutes, one way)", 0, 120, 20)
        parttime_job_hrs = st.slider("Part-time job (hrs/week)", 0, 40, 0)
        ucas_known = st.checkbox("UCAS tariff known", value=True)
        ucas_tariff = st.slider("UCAS tariff", 40, 250, 128, disabled=not ucas_known)

row = {
    "course": course, "year_of_study": year_of_study, "age": age, "gender": gender,
    "fee_status": fee_status, "first_generation": first_generation,
    "accommodation": accommodation, "commute_mins": commute_mins,
    "parttime_job_hrs": parttime_job_hrs,
    "ucas_tariff": ucas_tariff if ucas_known else np.nan,
    "ucas_missing": int(not ucas_known),
}

# ---------- extra inputs (mid-year only) ----------
if mode == "midyear":
    with st.container(border=True):
        st.subheader("Study and engagement so far")
        col3, col4 = st.columns(2)
        with col3:
            attendance_known = st.checkbox("Attendance recorded", value=True)
            attendance_pct = st.slider("Attendance (%)", 0, 100, 80, disabled=not attendance_known)
            study_hrs_week = st.slider("Independent study (hrs/week)", 0.0, 70.0, 14.0, 0.5)
            sleep_hrs = st.slider("Average sleep (hrs/night)", 3.0, 12.0, 7.5, 0.5)
            society_member = st.radio("Society member?", ["No", "Yes"], horizontal=True)
        with col4:
            vle_logins_month = st.slider("VLE logins per month", 0, 300, 25)
            late_submissions = st.number_input("Late submissions", min_value=0, max_value=20, value=0)
            coursework_avg = st.slider("Coursework average (%)", 0.0, 100.0, 60.0, 0.5)
            midterm_score = st.slider("Midterm score (%)", 0.0, 100.0, 58.0, 0.5)

    row.update({
        "attendance_pct": attendance_pct if attendance_known else np.nan,
        "attendance_missing": int(not attendance_known),
        "study_hrs_week": study_hrs_week, "sleep_hrs": sleep_hrs,
        "society_member": society_member, "vle_logins_month": vle_logins_month,
        "late_submissions": late_submissions, "coursework_avg": coursework_avg,
        "midterm_score": midterm_score,
        # same engineered features as the notebook
        "sleep_gap": abs(sleep_hrs - 7.5), "study_hrs_sq": study_hrs_week ** 2,
    })

X_new = pd.DataFrame([row])[columns]   # only this model's columns, in training order

CLASS_COLOURS = {"First": "#10B981", "2:1": "#6366F1", "2:2": "#F59E0B",
                 "Third": "#F97316", "Fail": "#EF4444"}


def degree_class(g):
    if g >= 70: return "First"
    if g >= 60: return "2:1"
    if g >= 50: return "2:2"
    if g >= 40: return "Third"
    return "Fail"


# ---------- predict ----------
st.write("")
if st.button("Predict grade", type="primary", use_container_width=True):
    pred = float(np.clip(model.predict(X_new)[0], 0, 100))
    low, high = max(pred - mae, 0), min(pred + mae, 100)
    cls = degree_class(pred)
    colour = CLASS_COLOURS[cls]
    note = ("Switch to Mid-year once attendance and coursework marks are available "
            "for a more precise prediction.") if mode == "enrolment" else ""

    # HTML lines start at the left margin on purpose: indented lines would be shown as code
    st.markdown(f"""
<div class="result-card">
<div class="result-top">
<div><div class="result-label">Predicted final grade</div><div class="grade">{pred:.1f}%</div></div>
<div><div class="result-label">Classification</div>
<span class="pill" style="background:{colour}22; color:{colour};">{cls}</span></div>
</div>
<div class="track">
<div class="band" style="left:{low:.1f}%; width:{high - low:.1f}%;"></div>
<div class="marker" style="left:{pred:.1f}%;"></div>
</div>
<div class="ticks">
<span style="left:20%">Fail</span><span style="left:45%">Third</span><span style="left:55%">2:2</span>
<span style="left:65%">2:1</span><span style="left:85%">First</span>
</div>
<div class="range-text">{mode_label} predictions are typically within <b>±{mae:.1f} marks</b>:
likely range <b>{low:.1f}%–{high:.1f}%</b> ({degree_class(low)} to {degree_class(high)}).</div>
<div class="note">{note}</div>
</div>
""", unsafe_allow_html=True)

st.write("")
st.caption("Trained on synthetic data, for demonstration only. "
           "Switch light/dark mode in the ⋮ menu → Settings.")
