import html
from difflib import SequenceMatcher

import pandas as pd
import streamlit as st

from mtester.models import SentimentModel
from mtester.perturbations import Typo, Uppercase, Negation
from mtester.runner import run_tests

st.set_page_config(page_title="Metamorphic Tester", page_icon="🧪", layout="wide")

CSS = """
<style>
.stApp{background:radial-gradient(1200px 600px at 10% -10%,#1b2450 0%,transparent 60%),radial-gradient(900px 500px at 100% 0%,#3b1a5a 0%,transparent 55%),#0B1020;}
[data-testid="stSidebar"]{background:#0E1530;border-right:1px solid #232c52;}
#MainMenu,footer{visibility:hidden;}
.block-container{padding-top:2rem;max-width:1200px;}
.hero{padding:34px 38px;border-radius:22px;background:linear-gradient(135deg,rgba(139,92,246,.35),rgba(34,211,238,.18));border:1px solid rgba(255,255,255,.12);margin-bottom:26px;}
.badge{display:inline-block;font-size:12px;letter-spacing:.14em;padding:6px 12px;border-radius:999px;background:rgba(255,255,255,.1);color:#C4B5FD;font-weight:600;}
.hero h1{font-size:46px;margin:14px 0 6px;padding:0;background:linear-gradient(90deg,#fff,#A5B4FC,#67E8F9);-webkit-background-clip:text;-webkit-text-fill-color:transparent;}
.hero p{color:#B8C0E0;font-size:17px;margin:0;}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:16px;margin-bottom:10px;}
.card{background:rgba(255,255,255,.05);border:1px solid rgba(255,255,255,.1);border-radius:18px;padding:20px 22px;}
.card .lbl{color:#9AA4CC;font-size:13px;text-transform:uppercase;letter-spacing:.08em;}
.card .val{font-size:36px;font-weight:700;margin-top:6px;}
.sub{color:#9AA4CC;font-size:13px;margin-top:4px;}
.ring{width:100px;height:100px;border-radius:50%;display:flex;align-items:center;justify-content:center;margin:8px 0 2px;}
.ring div{width:76px;height:76px;border-radius:50%;background:#10172F;display:flex;align-items:center;justify-content:center;font-size:22px;font-weight:700;}
.sec{font-size:22px;font-weight:700;margin:30px 0 12px;}
.barrow{display:flex;align-items:center;gap:14px;margin:16px 0;flex-wrap:wrap;}
.barrow .name{width:150px;font-weight:600;}
.track{flex:1;min-width:140px;height:14px;background:#1c2547;border-radius:999px;overflow:hidden;}
.fill{height:100%;border-radius:999px;}
.barrow .pct{width:64px;text-align:right;font-weight:700;}
.barrow .cnt{width:90px;color:#9AA4CC;font-size:13px;}
.fcard{background:rgba(255,255,255,.04);border:1px solid rgba(255,255,255,.09);border-left:4px solid #EF4444;border-radius:14px;padding:16px 18px;margin:12px 0;}
.pill{display:inline-block;font-size:12px;padding:3px 10px;border-radius:999px;background:rgba(139,92,246,.25);color:#DDD6FE;margin-right:8px;font-weight:600;}
.pill.alt{background:rgba(255,255,255,.08);color:#B8C0E0;}
.txt{margin:10px 0 0;line-height:1.55;font-size:15px;}
.txt small{color:#8892BB;display:block;font-size:11px;letter-spacing:.1em;text-transform:uppercase;}
mark{background:rgba(250,204,21,.28);color:#FDE68A;padding:1px 4px;border-radius:5px;}
.chip{display:inline-block;padding:3px 10px;border-radius:8px;font-size:12px;font-weight:700;}
.chip.a{background:rgba(99,102,241,.25);color:#C7D2FE;}
.chip.b{background:rgba(244,63,94,.22);color:#FDA4AF;}
.steps{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:16px;}
.step .num{width:34px;height:34px;border-radius:50%;background:linear-gradient(135deg,#8B5CF6,#06B6D4);display:flex;align-items:center;justify-content:center;font-weight:700;margin-bottom:10px;}
.stButton>button{background:linear-gradient(90deg,#8B5CF6,#06B6D4);color:#fff;border:0;border-radius:12px;font-weight:700;padding:.65rem 1.2rem;width:100%;}
.stButton>button:hover{filter:brightness(1.12);color:#fff;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

AVAILABLE_TESTS = {
    "Negation (should flip)": Negation,
    "Typos (should stay)": Typo,
    "Casing (should stay)": Uppercase,
}
TEST_INFO = {
    "negation": ("Negation", "should flip"),
    "typo": ("Typos", "should stay"),
    "casing": ("Casing", "should stay"),
}
PRESET_MODELS = [
    "distilbert-base-uncased-finetuned-sst-2-english",
]
CUSTOM_HF = "Other (Hugging Face name)"


@st.cache_resource(max_entries=1)
def load_model(name):
    return SentimentModel(name)


def rate_color(rate):
    if rate >= 90:
        return "#22C55E"
    if rate >= 70:
        return "#F59E0B"
    return "#EF4444"


def highlight(original, perturbed, test):
    """Wrap the changed words of the perturbed sentence in <mark>."""
    if test == "casing":
        return html.escape(perturbed)
    a, b = original.split(), perturbed.split()
    parts = []
    for tag, i1, i2, j1, j2 in SequenceMatcher(None, a, b).get_opcodes():
        chunk = " ".join(html.escape(w) for w in b[j1:j2])
        if not chunk:
            continue
        parts.append(chunk if tag == "equal" else f"<mark>{chunk}</mark>")
    return " ".join(parts)


# ---------- Hero ----------
st.markdown(
    '<div class="hero"><span class="badge">NLP ROBUSTNESS TESTING</span>'
    "<h1>Metamorphic Tester</h1>"
    "<p>Change the text a little. See if your model reacts the way a human would.</p></div>",
    unsafe_allow_html=True,
)

# ---------- Sidebar ----------
with st.sidebar:
    st.markdown("### 🧠 1. Model")
    choice = st.selectbox("Choose a model", PRESET_MODELS + [CUSTOM_HF])
    if choice == CUSTOM_HF:
        model_name = st.text_input("Hugging Face model name", placeholder="owner/model-name")
        st.caption("Small models only (under about 300 MB) on the free server.")
    else:
        model_name = choice

    st.markdown("### 📄 2. Test data")
    uploaded = st.file_uploader("Upload CSV with a 'text' column", type="csv")
    max_rows = st.slider("Number of reviews to use", 10, 200, 100, step=10)

    st.markdown("### 🧪 3. Tests")
    chosen = [name for name in AVAILABLE_TESTS if st.checkbox(name, value=True)]

    run = st.button("🚀 Run tests")

# ---------- Run the engine ----------
if run:
    if not chosen:
        st.warning("Pick at least one test.")
    else:
        data = pd.read_csv(uploaded) if uploaded else pd.read_csv("data/sample_reviews.csv")
        if "text" not in data.columns:
            st.error("Your CSV needs a column named 'text'.")
        else:
            sentences = [str(v).strip() for v in data["text"].dropna().head(max_rows).tolist()]
            with st.spinner("Loading model and running tests..."):
                try:
                    model = load_model((model_name or "").strip())
                except Exception as e:
                    st.error(
                        f"Could not load '{model_name}'. Check the name, and make sure "
                        f"it is a small text-classification model.\n\n{e}"
                    )
                    st.stop()
                tests = [AVAILABLE_TESTS[name]() for name in chosen]
                st.session_state["results"] = run_tests(model, sentences, tests)
                st.session_state["model_name"] = model_name

# ---------- Results ----------
df = st.session_state.get("results")

if df is None:
    st.markdown(
        '<div class="steps">'
        '<div class="card step"><div class="num">1</div><b>Pick a model</b><div class="sub">Use the preset or type a small Hugging Face model name.</div></div>'
        '<div class="card step"><div class="num">2</div><b>Choose your tests</b><div class="sub">Typos, casing, and negation. Each has a rule for what should happen.</div></div>'
        '<div class="card step"><div class="num">3</div><b>Run and inspect</b><div class="sub">See pass rates and the exact sentences that broke the model.</div></div>'
        "</div>",
        unsafe_allow_html=True,
    )
elif df.empty:
    st.warning("No tests were applicable to this data.")
else:
    overall = df["passed"].mean() * 100
    n_fail = int((~df["passed"]).sum())
    summary = df.groupby("test")["passed"].agg(rate="mean", n="count")
    summary["rate"] = summary["rate"] * 100
    weakest = summary["rate"].idxmin()
    weakest_label = TEST_INFO.get(weakest, (weakest, ""))[0]
    col = rate_color(overall)

    st.markdown(
        '<div class="grid">'
        f'<div class="card"><div class="lbl">Overall pass rate</div>'
        f'<div class="ring" style="background:conic-gradient({col} {overall}%,#1c2547 0)"><div>{overall:.0f}%</div></div></div>'
        f'<div class="card"><div class="lbl">Tests run</div><div class="val">{len(df)}</div>'
        f'<div class="sub">{html.escape(str(st.session_state.get("model_name", "")))}</div></div>'
        f'<div class="card"><div class="lbl">Failures</div><div class="val" style="color:#F87171">{n_fail}</div>'
        f'<div class="sub">model broke the rule</div></div>'
        f'<div class="card"><div class="lbl">Weakest spot</div><div class="val" style="font-size:28px">{weakest_label}</div>'
        f'<div class="sub">{summary.loc[weakest, "rate"]:.0f}% pass rate</div></div>'
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown('<div class="sec">Pass rate by test</div>', unsafe_allow_html=True)
    rows = []
    for test, r in summary.sort_values("rate").iterrows():
        label, rule = TEST_INFO.get(test, (test, ""))
        c = rate_color(r["rate"])
        rows.append(
            f'<div class="barrow"><div class="name">{label}<div class="sub">{rule}</div></div>'
            f'<div class="track"><div class="fill" style="width:{r["rate"]:.1f}%;background:{c}"></div></div>'
            f'<div class="pct" style="color:{c}">{r["rate"]:.0f}%</div>'
            f'<div class="cnt">{int(r["n"])} tests</div></div>'
        )
    st.markdown('<div class="card">' + "".join(rows) + "</div>", unsafe_allow_html=True)

    st.markdown('<div class="sec">Where the model broke</div>', unsafe_allow_html=True)
    fails = df[~df["passed"]]
    if fails.empty:
        st.success("No failures. The model followed every rule on this data.")
    else:
        pick = st.selectbox("Filter by test", ["All"] + sorted(fails["test"].unique().tolist()))
        shown = fails if pick == "All" else fails[fails["test"] == pick]
        cards = []
        for _, r in shown.head(12).iterrows():
            label, rule = TEST_INFO.get(r["test"], (r["test"], ""))
            cards.append(
                '<div class="fcard">'
                f'<span class="pill">{label}</span><span class="pill alt">{rule}</span>'
                f'<span class="chip a">{html.escape(str(r["pred_before"]))}</span> → '
                f'<span class="chip b">{html.escape(str(r["pred_after"]))}</span>'
                f'<div class="txt"><small>Original</small>{html.escape(r["original"])}</div>'
                f'<div class="txt"><small>Changed</small>{highlight(r["original"], r["perturbed"], r["test"])}</div>'
                "</div>"
            )
        st.markdown("".join(cards), unsafe_allow_html=True)
        if len(shown) > 12:
            st.caption(f"Showing 12 of {len(shown)} failures. Open the full table below to see all.")

    with st.expander("Full results table"):
        st.dataframe(df)
    st.download_button(
        "⬇ Download full results (CSV)",
        df.to_csv(index=False),
        file_name="results.csv",
        mime="text/csv",
    )