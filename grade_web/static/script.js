const form = document.getElementById("predict-form");
const resultBox = document.getElementById("result");
const errorBox = document.getElementById("error");
const predictBtn = document.getElementById("predict-btn");
const midyearSection = document.getElementById("midyear-section");

const CLASS_COLOURS = {
  "First": "#10B981", "2:1": "#6366F1", "2:2": "#F59E0B", "Third": "#F97316", "Fail": "#EF4444",
};

/* ---------- light / dark toggle ---------- */
document.getElementById("theme-toggle").addEventListener("click", () => {
  const next = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
  document.documentElement.dataset.theme = next;
  try { localStorage.setItem("theme", next); } catch (e) { /* storage blocked: still switches */ }
});

/* ---------- sliders: show the value and colour the filled part ---------- */
function updateSlider(slider) {
  const out = slider.closest(".field")?.querySelector("output");
  if (out) out.textContent = slider.value;
  const pct = ((slider.value - slider.min) / (slider.max - slider.min)) * 100;
  slider.style.setProperty("--fill", pct + "%");
}
document.querySelectorAll('input[type="range"]').forEach((s) => {
  updateSlider(s);
  s.addEventListener("input", () => updateSlider(s));
});

/* ---------- "known" checkboxes enable / disable their slider ---------- */
document.querySelectorAll("[data-known-for]").forEach((box) => {
  const slider = form.elements[box.dataset.knownFor];
  const sync = () => { slider.disabled = !box.checked; };
  box.addEventListener("change", sync);
  sync();
});

/* ---------- mode switch shows / hides the mid-year section ---------- */
function currentMode() {
  return form.elements.mode.value;
}
function syncMode() {
  midyearSection.hidden = currentMode() !== "midyear";
}
form.querySelectorAll('input[name="mode"]').forEach((r) => r.addEventListener("change", syncMode));
syncMode();

/* ---------- collect the answers ---------- */
function collect() {
  const mode = currentMode();
  const data = { mode };
  const fields = form.querySelectorAll("select[name], input[name]");
  fields.forEach((el) => {
    if (el.name === "mode") return;
    if (mode === "enrolment" && midyearSection.contains(el)) return;   // not used at enrolment
    if (el.type === "radio") {
      if (el.checked) data[el.name] = el.value;
    } else if (el.disabled) {
      data[el.name] = null;                                            // "not known"
    } else if (el.type === "range" || el.type === "number" || el.hasAttribute("data-number")) {
      data[el.name] = el.value === "" ? "" : Number(el.value);
    } else {
      data[el.name] = el.value;
    }
  });
  return data;
}

/* ---------- show the result ---------- */
function showResult(r) {
  const colour = CLASS_COLOURS[r.degree_class];
  const modeLabel = r.mode === "enrolment" ? "At enrolment" : "Mid-year";
  const note = r.mode === "enrolment"
    ? '<div class="note">Switch to Mid-year once attendance and coursework marks are available for a more precise prediction.</div>'
    : "";
  resultBox.innerHTML = `
    <div class="result-top">
      <div><div class="result-label">Predicted final grade</div><div class="grade">${r.grade.toFixed(1)}%</div></div>
      <div><div class="result-label">Classification</div>
        <span class="pill" style="background:${colour}22; color:${colour};">${r.degree_class}</span></div>
    </div>
    <div class="track">
      <div class="band" style="left:${r.low}%; width:${(r.high - r.low).toFixed(1)}%;"></div>
      <div class="marker" style="left:${r.grade}%;"></div>
    </div>
    <div class="ticks">
      <span style="left:20%">Fail</span><span style="left:45%">Third</span><span style="left:55%">2:2</span>
      <span style="left:65%">2:1</span><span style="left:85%">First</span>
    </div>
    <div class="range-text">${modeLabel} predictions are typically within <b>±${r.mae.toFixed(1)} marks</b>:
      likely range <b>${r.low.toFixed(1)}%–${r.high.toFixed(1)}%</b> (${r.low_class} to ${r.high_class}).</div>
    ${note}`;
  resultBox.hidden = false;
}

/* ---------- ask the server for a prediction ---------- */
let hasPredicted = false;

async function predict({ scroll = false } = {}) {
  errorBox.hidden = true;
  predictBtn.disabled = true;
  try {
    const res = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(collect()),
    });
    const body = await res.json();
    if (!res.ok) throw new Error((body.errors || ["Something went wrong."]).join(" · "));
    showResult(body);
    hasPredicted = true;
    if (scroll) resultBox.scrollIntoView({ behavior: "smooth", block: "nearest" });
  } catch (err) {
    resultBox.hidden = true;
    errorBox.textContent = err.message.startsWith("Failed to fetch")
      ? "Can't reach the server. Is app.py still running?"
      : err.message;
    errorBox.hidden = false;
  } finally {
    predictBtn.disabled = false;
  }
}

form.addEventListener("submit", (e) => {
  e.preventDefault();
  predict({ scroll: true });
});

/* after the first prediction, update it live as answers change */
let timer;
form.addEventListener("input", () => {
  if (!hasPredicted) return;
  clearTimeout(timer);
  timer = setTimeout(() => predict(), 250);
});
form.addEventListener("change", () => {
  if (!hasPredicted) return;
  clearTimeout(timer);
  timer = setTimeout(() => predict(), 50);
});
