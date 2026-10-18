
import json
import pickle
import random
from pathlib import Path

import streamlit as st

# -------------------- FILE PATHS --------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_FILE = BASE_DIR / "chatbot_model.pkl"
INTENTS_FILE = BASE_DIR / "intents.json"

# -------------------- PAGE SETTINGS --------------------
st.set_page_config(
    page_title="Nexa AI | Smart Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -------------------- DARK THEME CSS --------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root {
    --bg: #080d19;
    --panel: #111a2e;
    --border: rgba(148, 163, 184, 0.20);
    --text: #e8edff;
    --muted: #aab6d3;
    --purple: #8b5cf6;
    --blue: #60a5fa;
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
}

.stApp {
    background:
        radial-gradient(ellipse at 10% 0%, rgba(91,33,182,.20), transparent 38%),
        radial-gradient(ellipse at 95% 15%, rgba(37,99,235,.13), transparent 35%),
        linear-gradient(135deg, #080d19, #0d1426 55%, #101027);
    color: var(--text);
}

[data-testid="stHeader"] {
    background: rgba(8, 13, 25, 0.85);
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #10172b, #0a1020);
    border-right: 1px solid var(--border);
}

[data-testid="stSidebar"] * {
    color: var(--text);
}

[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
    color: var(--muted);
}

/* Hero banner */
.hero {
    padding: 36px;
    border-radius: 24px;
    background:
        radial-gradient(ellipse at top right, rgba(96,165,250,.30), transparent 45%),
        linear-gradient(120deg, #242052, #30266a 52%, #153a68);
    border: 1px solid rgba(167,139,250,.30);
    margin-bottom: 26px;
    box-shadow: 0 16px 45px rgba(0,0,0,.22);
}

.hero-kicker {
    display: inline-block;
    padding: 8px 13px;
    border: 1px solid rgba(196,181,253,.30);
    border-radius: 30px;
    color: #d9d6ff;
    font-size: 10px;
    font-weight: 800;
    letter-spacing: 1.5px;
    margin-bottom: 20px;
}

.status-dot {
    display: inline-block;
    height: 8px;
    width: 8px;
    background: #60a5fa;
    border-radius: 50%;
    margin-right: 7px;
}

.hero h1 {
    color: #ffffff !important;
    font-size: clamp(30px, 4vw, 44px);
    font-weight: 800;
    letter-spacing: -1.3px;
    margin: 0 0 15px 0;
}

.hero-sparkle {
    color: #c4b5fd;
}

.hero-description {
    color: #e0e7ff !important;
    font-size: 15px;
    line-height: 1.8;
    margin-bottom: 20px;
}

.hero-tags span {
    display: inline-block;
    padding: 9px 13px;
    margin: 3px 6px 0 0;
    border-radius: 12px;
    background: rgba(255,255,255,.08);
    border: 1px solid rgba(255,255,255,.12);
    color: #f1f5ff;
    font-size: 12px;
    font-weight: 600;
}

/* Metric cards */
.metric-card {
    background: linear-gradient(145deg, rgba(24,35,61,.95), rgba(16,25,45,.95));
    border: 1px solid var(--border);
    padding: 20px 12px;
    border-radius: 18px;
    text-align: center;
    min-height: 95px;
    box-shadow: 0 8px 25px rgba(0,0,0,.12);
}

.metric-number {
    background: linear-gradient(90deg, #a78bfa, #60a5fa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 27px;
    font-weight: 800;
}

.metric-label {
    color: var(--muted);
    font-size: 12px;
    margin-top: 5px;
}

.section-title {
    color: #e5eaff;
    font-size: 20px;
    font-weight: 750;
    margin: 25px 0 14px 0;
}

/* Conversation messages */
[data-testid="stChatMessage"] {
    background: rgba(19,29,51,.92) !important;
    border: 1px solid var(--border) !important;
    border-radius: 18px !important;
    padding: 15px !important;
    margin-bottom: 12px !important;
    box-shadow: 0 6px 20px rgba(0,0,0,.10);
}

[data-testid="stChatMessage"] p,
[data-testid="stChatMessage"] li {
    color: #e8edff !important;
    line-height: 1.75;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #172342, #211d48) !important;
    color: #f1f5ff !important;
    border: 1px solid rgba(139,92,246,.45) !important;
    border-radius: 13px !important;
    font-weight: 600 !important;
    min-height: 45px;
}

.stButton > button:hover {
    background: linear-gradient(135deg, #4338ca, #6d28d9) !important;
    border-color: #a78bfa !important;
    color: #ffffff !important;
}

/* Chat input and its bottom container */
[data-testid="stBottom"] {
    background: #080d19 !important;
}

[data-testid="stBottom"] > div,
[data-testid="stBottom"] [data-testid="stBottomBlockContainer"] {
    background: #080d19 !important;
}

[data-testid="stChatInput"] {
    background: #111a2e !important;
    border: 1px solid #8b5cf6 !important;
    border-radius: 16px !important;
    box-shadow: 0 0 18px rgba(139,92,246,.10);
}

[data-testid="stChatInput"] > div {
    background: transparent !important;
}

[data-testid="stChatInput"] textarea,
[data-testid="stChatInput"] textarea:focus,
[data-testid="stChatInput"] textarea:hover {
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
    caret-color: #a78bfa !important;
    background: transparent !important;
    opacity: 1 !important;
}

[data-testid="stChatInput"] textarea::placeholder {
    color: #aab6d3 !important;
    -webkit-text-fill-color: #aab6d3 !important;
    opacity: 1 !important;
}

/* Expanders */
[data-testid="stExpander"] {
    background: rgba(17,26,46,.85);
    border: 1px solid var(--border);
    border-radius: 12px;
}

[data-testid="stExpander"] summary,
[data-testid="stExpander"] summary p {
    color: #e8edff !important;
}

/* Footer */
.footer {
    text-align: center;
    color: #9ba9c5;
    font-size: 12px;
    padding: 25px 0 12px 0;
    border-top: 1px solid var(--border);
    margin-top: 30px;
}
</style>
""", unsafe_allow_html=True)


# -------------------- LOAD MODEL --------------------
@st.cache_resource
def load_bundle():
    if not MODEL_FILE.exists():
        return None, "Model file not found. Run: python train_model.py"

    try:
        with open(MODEL_FILE, "rb") as file:
            bundle = pickle.load(file)

        required = {"model", "encoder", "responses", "threshold"}

        if not isinstance(bundle, dict) or not required.issubset(bundle):
            return None, "Model format is outdated. Run: python train_model.py"

        return bundle, None

    except Exception as exc:
        return None, f"Could not load model: {exc}"


@st.cache_data
def load_intents():
    with open(INTENTS_FILE, "r", encoding="utf-8") as file:
        return json.load(file)


bundle, load_error = load_bundle()

if load_error:
    st.error(load_error)
    st.info("Check that chatbot_model.pkl exists and run python train_model.py if needed.")
    st.stop()

try:
    intents_data = load_intents()
except Exception as exc:
    st.error(f"Could not load intents.json: {exc}")
    st.stop()

model = bundle["model"]
encoder = bundle["encoder"]
responses = bundle["responses"]
threshold = bundle["threshold"]

intent_count = len([
    intent for intent in intents_data.get("intents", [])
    if intent.get("tag") != "fallback"
])

pattern_count = sum(
    len(intent.get("patterns", []))
    for intent in intents_data.get("intents", [])
    if intent.get("tag") != "fallback"
)


# -------------------- CHATBOT LOGIC --------------------
def get_response(message):
    probabilities = model.predict_proba([message])[0]
    best_index = probabilities.argmax()
    confidence = float(probabilities[best_index])

    encoded_label = model.named_steps["classifier"].classes_[best_index]
    predicted_tag = encoder.inverse_transform([encoded_label])[0]

    if confidence < threshold:
        fallback_responses = responses.get(
            "fallback",
            ["I'm not sure I understood. Please try another question."]
        )
        return random.choice(fallback_responses), "fallback", confidence

    answer_options = responses.get(predicted_tag)

    if not answer_options:
        return "Please try asking that another way.", "fallback", confidence

    return random.choice(answer_options), predicted_tag, confidence


# -------------------- SESSION STATE --------------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Hello! 👋 I'm Nexa AI. Ask me about Python, machine learning, "
                "NLP, study tips, or motivation."
            ),
            "tag": "greeting",
            "confidence": None,
        }
    ]

if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None


# -------------------- SIDEBAR --------------------
with st.sidebar:
    st.markdown("# 🤖 Nexa AI")
    st.caption("YOUR SMART LEARNING COMPANION")
    st.divider()

    st.markdown("### ✨ What I can help with")
    st.markdown("""
    - 🐍 Python & Programming
    - 🧠 Machine Learning
    - 💬 Natural Language Processing
    - 📚 Study & Exam Tips
    - 🌟 Motivation
    - 💡 Interesting Facts
    """)

    st.divider()
    st.markdown("### 📊 Project Overview")
    st.write(f"Intent categories: **{intent_count}**")
    st.write(f"Training examples: **{pattern_count}**")
    st.caption("Built with Python, TF-IDF and Logistic Regression.")

    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.pending_prompt = None
        st.rerun()

    st.divider()
    st.caption("Nexa AI · Student AI/ML Project")


# -------------------- MAIN HEADER --------------------
st.markdown("""
<div class="hero">
    <div class="hero-kicker">
        <span class="status-dot"></span>
        YOUR PERSONAL AI COMPANION
    </div>
    <h1>Meet Nexa AI <span class="hero-sparkle">✦</span></h1>
    <p class="hero-description">
        Your ideas deserve intelligent answers.
        Explore Python, machine learning, NLP, and smarter ways to study.
    </p>
    <div class="hero-tags">
        <span>✧ Learn</span>
        <span>⌘ Explore</span>
        <span>✦ Discover</span>
    </div>
</div>
""", unsafe_allow_html=True)


# -------------------- METRIC CARDS --------------------
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-number">{intent_count}</div>
        <div class="metric-label">Supported Topics</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-number">NLP</div>
        <div class="metric-label">Language Processing</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-number">ML</div>
        <div class="metric-label">Intent Classification</div>
    </div>
    """, unsafe_allow_html=True)


# -------------------- QUICK PROMPTS --------------------
st.markdown(
    '<div class="section-title">💡 Try asking me</div>',
    unsafe_allow_html=True,
)

quick_prompts = [
    ("🐍 Learn Python", "What is Python?"),
    ("🧠 Explore ML", "What is machine learning?"),
    ("💬 Discover NLP", "What is NLP?"),
    ("📚 Study smarter", "Give me study tips"),
]

prompt_columns = st.columns(4)

for index, (label, prompt) in enumerate(quick_prompts):
    with prompt_columns[index]:
        if st.button(label, key=f"quick_{index}", use_container_width=True):
            st.session_state.pending_prompt = prompt


# -------------------- CHAT HISTORY --------------------
st.markdown(
    '<div class="section-title">💬 Your conversation</div>',
    unsafe_allow_html=True,
)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant" and message.get("tag"):
            if message.get("confidence") is not None:
                with st.expander("View prediction details"):
                    st.write(f"Predicted intent: `{message['tag']}`")
                    st.write(
                        f"Model confidence: {message['confidence']:.1%}"
                    )


# -------------------- PROCESS USER MESSAGE --------------------
typed_message = st.chat_input("Ask Nexa AI something...")

user_message = typed_message or st.session_state.pending_prompt
st.session_state.pending_prompt = None

if user_message and user_message.strip():
    user_message = user_message.strip()

    st.session_state.messages.append({
        "role": "user",
        "content": user_message,
    })

    try:
        answer, tag, confidence = get_response(user_message)

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "tag": tag,
            "confidence": confidence,
        })

    except Exception as exc:
        st.session_state.messages.append({
            "role": "assistant",
            "content": f"Sorry, I encountered an error while answering. Details: {exc}",
            "tag": "error",
            "confidence": None,
        })

    st.rerun()



# ---------- Nexa AI Footer ----------
st.markdown("---")

st.markdown("""
<div style="
    text-align: center;
    padding: 12px 8px 18px 8px;
    color: #64748b;
    font-size: 13px;
    line-height: 1.9;
">
    <div style="
        font-size: 17px;
        font-weight: 700;
        color: #6366f1;
    ">
        ✨ Nexa AI
    </div>

    <div>© 2026 Tanya Patel</div>

    <div style="font-size: 12px;">
        Aspiring AI/ML Engineer | Python Developer
    </div>
</div>
""", unsafe_allow_html=True)
