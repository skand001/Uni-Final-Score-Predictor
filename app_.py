"""Student grade predictor - Streamlit app with two modes, light/dark styling
and four languages (English, German, Spanish, Hungarian).

Run with:  streamlit run app.py
Needs, in the same folder:
  grade_models.joblib       (saved from the notebook)
  .streamlit/config.toml    (the light/dark theme)

How the languages work: every dropdown and radio button keeps the ENGLISH value
the model was trained on (e.g. "Law"), and only the label shown on screen is
translated (e.g. "Rechtswissenschaft"). So the model always gets what it expects.
"""
import joblib
import numpy as np
import pandas as pd
import sklearn
import streamlit as st

# =====================================================================
# Translations
# =====================================================================
LANGUAGES = {"en": "English", "de": "Deutsch", "es": "Español", "hu": "Magyar"}

TEXT = {
    "en": {
        "title": "Student Grade Predictor",
        "badge": "Machine learning demo",
        "subtitle": "Estimate a final university grade from a student's profile and engagement.",
        "mode_question": "When are you predicting?",
        "mode_help": "At enrolment uses only what's known on day one, so it is less precise. "
                     "Mid-year adds attendance, coursework and exam results.",
        "mode": {"enrolment": "At enrolment", "midyear": "Mid-year"},
        "mode_in_sentence": {"enrolment": "At enrolment", "midyear": "Mid-year"},
        "about": "About the student",
        "course": "Course", "year": "Year of study", "age": "Age", "gender": "Gender",
        "fee": "Fee status", "first_gen": "First in family at university?",
        "accommodation": "Accommodation", "commute": "Commute (minutes, one way)",
        "parttime": "Part-time job (hrs/week)", "ucas_known": "UCAS tariff known",
        "ucas": "UCAS tariff",
        "study_section": "Study and engagement so far",
        "attendance_known": "Attendance recorded", "attendance": "Attendance (%)",
        "study": "Independent study (hrs/week)", "sleep": "Average sleep (hrs/night)",
        "society": "Society member?", "vle": "VLE logins per month",
        "late": "Late submissions", "coursework": "Coursework average (%)",
        "midterm": "Midterm score (%)",
        "predict": "Predict grade",
        "result_grade": "Predicted final grade", "result_class": "Classification",
        "range": "{mode} predictions are typically within <b>±{mae} marks</b>: "
                 "likely range <b>{low}–{high}</b> ({low_cls} to {high_cls}).",
        "note_enrolment": "Switch to Mid-year once attendance and coursework marks are "
                          "available for a more precise prediction.",
        "footer": "Trained on synthetic data, for demonstration only. "
                  "Switch light/dark mode in the ⋮ menu → Settings.",
        "version_warning": "Models were trained with scikit-learn {saved}, but this app is "
                           "running {running}. Predictions may be unreliable.",
        "options": {
            "course": {"Biomedical Science": "Biomedical Science", "Computer Science": "Computer Science",
                       "Economics": "Economics", "English Literature": "English Literature", "Law": "Law",
                       "Mechanical Engineering": "Mechanical Engineering", "Nursing": "Nursing",
                       "Psychology": "Psychology"},
            "gender": {"Female": "Female", "Male": "Male", "Non-binary": "Non-binary"},
            "fee": {"Home": "Home", "International": "International"},
            "yes_no": {"No": "No", "Yes": "Yes"},
            "accommodation": {"Halls": "Halls", "Private rented": "Private rented",
                              "Family home": "Family home"},
        },
        "class_long": {"First": "First", "2:1": "2:1", "2:2": "2:2", "Third": "Third", "Fail": "Fail"},
        "class_short": {"First": "First", "2:1": "2:1", "2:2": "2:2", "Third": "Third", "Fail": "Fail"},
    },

    "de": {
        "title": "Notenprognose für Studierende",
        "badge": "Machine-Learning-Demo",
        "subtitle": "Schätzt die Abschlussnote an der Universität anhand von Profil und Engagement "
                    "eines Studierenden.",
        "mode_question": "Wann wird die Prognose erstellt?",
        "mode_help": "„Bei der Einschreibung“ nutzt nur, was am ersten Tag bekannt ist, und ist daher "
                     "ungenauer. „Zur Jahresmitte“ berücksichtigt zusätzlich Anwesenheit, "
                     "Studienleistungen und Prüfungsergebnisse.",
        "mode": {"enrolment": "Bei der Einschreibung", "midyear": "Zur Jahresmitte"},
        "mode_in_sentence": {"enrolment": "„Bei der Einschreibung“", "midyear": "„Zur Jahresmitte“"},
        "about": "Angaben zur Person",
        "course": "Studiengang", "year": "Studienjahr", "age": "Alter", "gender": "Geschlecht",
        "fee": "Gebührenstatus", "first_gen": "Erste Person in der Familie mit Studium?",
        "accommodation": "Wohnsituation", "commute": "Pendelzeit (Minuten, einfach)",
        "parttime": "Nebenjob (Std./Woche)", "ucas_known": "UCAS-Punktzahl bekannt",
        "ucas": "UCAS-Punktzahl",
        "study_section": "Studium und Engagement bisher",
        "attendance_known": "Anwesenheit erfasst", "attendance": "Anwesenheit (%)",
        "study": "Selbststudium (Std./Woche)", "sleep": "Durchschnittlicher Schlaf (Std./Nacht)",
        "society": "Mitglied in einer Studierendengruppe?",
        "vle": "Logins auf der Lernplattform pro Monat",
        "late": "Verspätete Abgaben", "coursework": "Durchschnitt der Studienleistungen (%)",
        "midterm": "Ergebnis der Zwischenprüfung (%)",
        "predict": "Note vorhersagen",
        "result_grade": "Prognostizierte Abschlussnote", "result_class": "Einstufung",
        "range": "Prognosen {mode} liegen typischerweise innerhalb von <b>±{mae} Punkten</b>: "
                 "wahrscheinlicher Bereich <b>{low}–{high}</b> ({low_cls} bis {high_cls}).",
        "note_enrolment": "Wechseln Sie zu „Zur Jahresmitte“, sobald Anwesenheit und Noten der "
                          "Studienleistungen vorliegen, um eine genauere Prognose zu erhalten.",
        "footer": "Mit synthetischen Daten trainiert, nur zu Demonstrationszwecken. "
                  "Hell-/Dunkelmodus im Menü ⋮ → Settings umschalten.",
        "version_warning": "Die Modelle wurden mit scikit-learn {saved} trainiert, diese App nutzt "
                           "jedoch {running}. Die Prognosen sind möglicherweise unzuverlässig.",
        "options": {
            "course": {"Biomedical Science": "Biomedizin", "Computer Science": "Informatik",
                       "Economics": "Volkswirtschaftslehre", "English Literature": "Englische Literatur",
                       "Law": "Rechtswissenschaft", "Mechanical Engineering": "Maschinenbau",
                       "Nursing": "Pflege", "Psychology": "Psychologie"},
            "gender": {"Female": "Weiblich", "Male": "Männlich", "Non-binary": "Nicht-binär"},
            "fee": {"Home": "Inländisch", "International": "International"},
            "yes_no": {"No": "Nein", "Yes": "Ja"},
            "accommodation": {"Halls": "Wohnheim", "Private rented": "Privat gemietet",
                              "Family home": "Elternhaus"},
        },
        "class_long": {"First": "First (höchste Stufe)", "2:1": "2:1 (obere zweite Stufe)",
                       "2:2": "2:2 (untere zweite Stufe)", "Third": "Third (dritte Stufe)",
                       "Fail": "Nicht bestanden"},
        "class_short": {"First": "First", "2:1": "2:1", "2:2": "2:2", "Third": "Third",
                        "Fail": "Nicht bestanden"},
    },

    "es": {
        "title": "Predictor de notas universitarias",
        "badge": "Demo de aprendizaje automático",
        "subtitle": "Estima la nota final universitaria a partir del perfil y la implicación "
                    "del estudiante.",
        "mode_question": "¿Cuándo se hace la predicción?",
        "mode_help": "«Al matricularse» usa solo lo que se sabe el primer día, por lo que es menos "
                     "precisa. «A mitad de curso» añade asistencia, trabajos y resultados de exámenes.",
        "mode": {"enrolment": "Al matricularse", "midyear": "A mitad de curso"},
        "mode_in_sentence": {"enrolment": "«Al matricularse»", "midyear": "«A mitad de curso»"},
        "about": "Sobre el estudiante",
        "course": "Titulación", "year": "Curso", "age": "Edad", "gender": "Género",
        "fee": "Tipo de matrícula", "first_gen": "¿Primera persona de la familia en la universidad?",
        "accommodation": "Alojamiento", "commute": "Desplazamiento (minutos, solo ida)",
        "parttime": "Trabajo a tiempo parcial (h/semana)", "ucas_known": "Puntuación UCAS conocida",
        "ucas": "Puntuación UCAS",
        "study_section": "Estudio e implicación hasta ahora",
        "attendance_known": "Asistencia registrada", "attendance": "Asistencia (%)",
        "study": "Estudio autónomo (h/semana)", "sleep": "Sueño medio (h/noche)",
        "society": "¿Miembro de una asociación estudiantil?",
        "vle": "Accesos al campus virtual al mes",
        "late": "Entregas fuera de plazo", "coursework": "Media de los trabajos (%)",
        "midterm": "Nota del examen parcial (%)",
        "predict": "Predecir nota",
        "result_grade": "Nota final prevista", "result_class": "Clasificación",
        "range": "Las predicciones {mode} suelen tener un margen de <b>±{mae} puntos</b>: "
                 "rango probable <b>{low}–{high}</b> ({low_cls} a {high_cls}).",
        "note_enrolment": "Cambia a «A mitad de curso» cuando haya datos de asistencia y notas de "
                          "trabajos para obtener una predicción más precisa.",
        "footer": "Entrenado con datos sintéticos, solo con fines de demostración. "
                  "Cambia entre modo claro y oscuro en el menú ⋮ → Settings.",
        "version_warning": "Los modelos se entrenaron con scikit-learn {saved}, pero esta app usa "
                           "{running}. Las predicciones pueden no ser fiables.",
        "options": {
            "course": {"Biomedical Science": "Ciencias Biomédicas", "Computer Science": "Informática",
                       "Economics": "Economía", "English Literature": "Literatura Inglesa",
                       "Law": "Derecho", "Mechanical Engineering": "Ingeniería Mecánica",
                       "Nursing": "Enfermería", "Psychology": "Psicología"},
            "gender": {"Female": "Mujer", "Male": "Hombre", "Non-binary": "No binario"},
            "fee": {"Home": "Nacional", "International": "Internacional"},
            "yes_no": {"No": "No", "Yes": "Sí"},
            "accommodation": {"Halls": "Residencia universitaria", "Private rented": "Alquiler privado",
                              "Family home": "Casa familiar"},
        },
        "class_long": {"First": "First (máxima calificación)", "2:1": "2:1 (segunda superior)",
                       "2:2": "2:2 (segunda inferior)", "Third": "Third (tercera)",
                       "Fail": "Suspenso"},
        "class_short": {"First": "First", "2:1": "2:1", "2:2": "2:2", "Third": "Third",
                        "Fail": "Suspenso"},
    },

    "hu": {
        "title": "Egyetemi eredmény előrejelző",
        "badge": "Gépi tanulási demó",
        "subtitle": "Megbecsüli a hallgató egyetemi záró eredményét a profilja és aktivitása alapján.",
        "mode_question": "Mikor készül az előrejelzés?",
        "mode_help": "A „Beiratkozáskor” mód csak az első napon ismert adatokat használja, ezért "
                     "kevésbé pontos. A „Tanév közben” mód a jelenlétet, a beadandókat és a "
                     "vizsgaeredményeket is figyelembe veszi.",
        "mode": {"enrolment": "Beiratkozáskor", "midyear": "Tanév közben"},
        "mode_in_sentence": {"enrolment": "beiratkozáskor", "midyear": "tanév közben"},
        "about": "A hallgatóról",
        "course": "Szak", "year": "Évfolyam", "age": "Életkor", "gender": "Neme",
        "fee": "Tandíjstátusz", "first_gen": "Első a családban, aki egyetemre jár?",
        "accommodation": "Lakhatás", "commute": "Ingázás (perc, egy irányba)",
        "parttime": "Részmunkaidős munka (óra/hét)", "ucas_known": "UCAS-pontszám ismert",
        "ucas": "UCAS-pontszám",
        "study_section": "Tanulás és aktivitás eddig",
        "attendance_known": "Jelenlét rögzítve", "attendance": "Jelenlét (%)",
        "study": "Önálló tanulás (óra/hét)", "sleep": "Átlagos alvás (óra/éjszaka)",
        "society": "Diákszervezet tagja?",
        "vle": "Bejelentkezések a tanulási platformra havonta",
        "late": "Késve leadott feladatok", "coursework": "Beadandók átlaga (%)",
        "midterm": "Félévközi vizsga eredménye (%)",
        "predict": "Eredmény előrejelzése",
        "result_grade": "Várható záró eredmény", "result_class": "Minősítés",
        "range": "A {mode} készült előrejelzések jellemzően <b>±{mae} ponton</b> belül vannak: "
                 "valószínű tartomány <b>{low}–{high}</b> ({low_cls} – {high_cls}).",
        "note_enrolment": "Válts „Tanév közben” módra, amint a jelenléti adatok és a beadandók "
                          "eredményei rendelkezésre állnak, a pontosabb előrejelzésért.",
        "footer": "Szintetikus adatokon tanítva, kizárólag bemutató célra. "
                  "Világos/sötét mód: ⋮ menü → Settings.",
        "version_warning": "A modellek a scikit-learn {saved} verziójával készültek, de ez az "
                           "alkalmazás a(z) {running} verziót futtatja. Az előrejelzések "
                           "pontatlanok lehetnek.",
        "options": {
            "course": {"Biomedical Science": "Orvosbiológia", "Computer Science": "Informatika",
                       "Economics": "Közgazdaságtan", "English Literature": "Angol irodalom",
                       "Law": "Jog", "Mechanical Engineering": "Gépészmérnöki",
                       "Nursing": "Ápolás", "Psychology": "Pszichológia"},
            "gender": {"Female": "Nő", "Male": "Férfi", "Non-binary": "Nem bináris"},
            "fee": {"Home": "Hazai", "International": "Nemzetközi"},
            "yes_no": {"No": "Nem", "Yes": "Igen"},
            "accommodation": {"Halls": "Kollégium", "Private rented": "Albérlet",
                              "Family home": "Családi otthon"},
        },
        "class_long": {"First": "First (legmagasabb fokozat)", "2:1": "2:1 (felső második fokozat)",
                       "2:2": "2:2 (alsó második fokozat)", "Third": "Third (harmadik fokozat)",
                       "Fail": "Nem felelt meg"},
        "class_short": {"First": "First", "2:1": "2:1", "2:2": "2:2", "Third": "Third",
                        "Fail": "Nem felelt meg"},
    },
}


def fmt_number(x, lang):
    """One decimal place, with a decimal comma for German, Spanish and Hungarian."""
    s = f"{x:.1f}"
    return s if lang == "en" else s.replace(".", ",")


def fmt_percent(x, lang):
    """English and Hungarian write 63.9% / 63,9%; German and Spanish put a space: 63,9 %."""
    s = fmt_number(x, lang)
    return f"{s} %" if lang in ("de", "es") else f"{s}%"


# Language for the browser tab title: the last choice, or ?lang= in the address, or English
initial_lang = st.session_state.get("lang") or st.query_params.get("lang", "en")
if initial_lang not in LANGUAGES:
    initial_lang = "en"

st.set_page_config(page_title=TEXT[initial_lang]["title"], page_icon="🎓", layout="centered")

# ---------- styling (works in both light and dark mode) ----------
st.markdown("""
<style>
.block-container {max-width: 880px; padding-top: 2.5rem; padding-bottom: 3rem;}
[data-testid="stDecoration"] {display: none;}
footer {visibility: hidden;}

.hero {margin-bottom: 1.5rem;}
.hero .badge {display: inline-block; font-size: .72rem; font-weight: 600; letter-spacing: .1em;
  text-transform: uppercase; padding: .3rem .8rem; border-radius: 999px;
  border: 1px solid rgba(128,128,128,.35); color: inherit; opacity: .7; margin-bottom: .75rem;}
.hero h1 {font-size: 2.6rem; font-weight: 800; letter-spacing: -.035em; line-height: 1.1; margin: 0;
  color: inherit;}  /* follows the theme: white in dark mode, black in light mode */
.hero p {opacity: .65; margin: .5rem 0 0; font-size: 1.02rem;}

/* Neutral button: text follows the theme (white in dark mode, black in light mode) */
div[data-testid="stButton"] button[kind="primary"],
button[data-testid="stBaseButton-primary"] {
  background: rgba(128,128,128,.08); border: 1px solid rgba(128,128,128,.35); color: inherit;
  font-weight: 600; letter-spacing: .01em; padding: .7rem 1rem; box-shadow: none;
  transition: background .15s ease, border-color .15s ease;}
div[data-testid="stButton"] button[kind="primary"]:hover,
button[data-testid="stBaseButton-primary"]:hover {
  background: rgba(128,128,128,.16); border-color: rgba(128,128,128,.6); color: inherit;}
div[data-testid="stButton"] button[kind="primary"]:active,
button[data-testid="stBaseButton-primary"]:active {
  background: rgba(128,128,128,.24); color: inherit;}

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

# ---------- header: title on the left, language picker on the right ----------
head_left, head_right = st.columns([5, 1.4], vertical_alignment="top")
with head_right:
    codes = list(LANGUAGES)
    lang = st.selectbox("🌐", codes, index=codes.index(initial_lang),
                        format_func=LANGUAGES.get, key="lang", label_visibility="collapsed")
st.query_params["lang"] = lang   # keeps the language in the address, so a shared link opens in it
T = TEXT[lang]
opt = T["options"]

with head_left:
    st.markdown(f"""
<div class="hero">
<span class="badge">{T['badge']}</span>
<h1>{T['title']}</h1>
<p>{T['subtitle']}</p>
</div>
""", unsafe_allow_html=True)


@st.cache_resource  # load the models once, not on every click
def load_models():
    return joblib.load("grade_models.joblib")


saved = load_models()

if saved["sklearn_version"] != sklearn.__version__:
    st.warning(T["version_warning"].format(saved=saved["sklearn_version"], running=sklearn.__version__))

# ---------- choose the mode ----------
mode = st.segmented_control(
    T["mode_question"],
    ["enrolment", "midyear"],                 # the values stay in English...
    format_func=lambda m: T["mode"][m],       # ...only the labels are translated
    default="midyear",
    help=T["mode_help"],
    key="mode",
) or "midyear"   # if the user clicks the selected option off, fall back to Mid-year
bundle = saved[mode]
model, columns, mae = bundle["model"], bundle["columns"], bundle["mae"]

# ---------- day-one inputs (both modes) ----------
# Every widget has a key, so its value is kept when the language changes.
with st.container(border=True):
    st.subheader(T["about"])
    col1, col2 = st.columns(2)
    with col1:
        course = st.selectbox(T["course"], list(opt["course"]),
                              format_func=opt["course"].get, key="course")
        year_of_study = st.selectbox(T["year"], [1, 2, 3], key="year_of_study")
        age = st.number_input(T["age"], min_value=17, max_value=70, value=20, key="age")
        gender = st.selectbox(T["gender"], list(opt["gender"]),
                              format_func=opt["gender"].get, key="gender")
        fee_status = st.selectbox(T["fee"], list(opt["fee"]),
                                  format_func=opt["fee"].get, key="fee_status")
    with col2:
        first_generation = st.radio(T["first_gen"], list(opt["yes_no"]), horizontal=True,
                                    format_func=opt["yes_no"].get, key="first_generation")
        accommodation = st.selectbox(T["accommodation"], list(opt["accommodation"]),
                                     format_func=opt["accommodation"].get, key="accommodation")
        commute_mins = st.slider(T["commute"], 0, 120, 20, key="commute_mins")
        parttime_job_hrs = st.slider(T["parttime"], 0, 40, 0, key="parttime_job_hrs")
        ucas_known = st.checkbox(T["ucas_known"], value=True, key="ucas_known")
        ucas_tariff = st.slider(T["ucas"], 40, 250, 128, disabled=not ucas_known, key="ucas_tariff")

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
        st.subheader(T["study_section"])
        col3, col4 = st.columns(2)
        with col3:
            attendance_known = st.checkbox(T["attendance_known"], value=True, key="attendance_known")
            attendance_pct = st.slider(T["attendance"], 0, 100, 80,
                                       disabled=not attendance_known, key="attendance_pct")
            study_hrs_week = st.slider(T["study"], 0.0, 70.0, 14.0, 0.5, key="study_hrs_week")
            sleep_hrs = st.slider(T["sleep"], 3.0, 12.0, 7.5, 0.5, key="sleep_hrs")
            society_member = st.radio(T["society"], list(opt["yes_no"]), horizontal=True,
                                      format_func=opt["yes_no"].get, key="society_member")
        with col4:
            vle_logins_month = st.slider(T["vle"], 0, 300, 25, key="vle_logins_month")
            late_submissions = st.number_input(T["late"], min_value=0, max_value=20, value=0,
                                               key="late_submissions")
            coursework_avg = st.slider(T["coursework"], 0.0, 100.0, 60.0, 0.5, key="coursework_avg")
            midterm_score = st.slider(T["midterm"], 0.0, 100.0, 58.0, 0.5, key="midterm_score")

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
if st.button(T["predict"], type="primary", use_container_width=True, key="predict"):
    pred = float(np.clip(model.predict(X_new)[0], 0, 100))
    low, high = max(pred - mae, 0), min(pred + mae, 100)
    cls = degree_class(pred)
    colour = CLASS_COLOURS[cls]
    short = T["class_short"]
    range_text = T["range"].format(
        mode=T["mode_in_sentence"][mode], mae=fmt_number(mae, lang),
        low=fmt_percent(low, lang), high=fmt_percent(high, lang),
        low_cls=short[degree_class(low)], high_cls=short[degree_class(high)])
    note = T["note_enrolment"] if mode == "enrolment" else ""

    # HTML lines start at the left margin on purpose: indented lines would be shown as code
    st.markdown(f"""
<div class="result-card">
<div class="result-top">
<div><div class="result-label">{T['result_grade']}</div><div class="grade">{fmt_percent(pred, lang)}</div></div>
<div><div class="result-label">{T['result_class']}</div>
<span class="pill" style="background:{colour}22; color:{colour};">{T['class_long'][cls]}</span></div>
</div>
<div class="track">
<div class="band" style="left:{low:.1f}%; width:{high - low:.1f}%;"></div>
<div class="marker" style="left:{pred:.1f}%;"></div>
</div>
<div class="ticks">
<span style="left:20%">{short['Fail']}</span><span style="left:45%">{short['Third']}</span>
<span style="left:55%">2:2</span><span style="left:65%">2:1</span><span style="left:85%">First</span>
</div>
<div class="range-text">{range_text}</div>
<div class="note">{note}</div>
</div>
""", unsafe_allow_html=True)

st.write("")
st.caption(T["footer"])
