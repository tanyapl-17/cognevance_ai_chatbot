
import json
import pickle
import random
from pathlib import Path

import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
MODEL_FILE = BASE_DIR / "chatbot_model.pkl"
INTENTS_FILE = BASE_DIR / "intents.json"

st.set_page_config(
    page_title="Nexa AI | Smart Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------- CUSTOM DESIGN --------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: linear-gradient(135deg, #f5f7ff 0%, #eef2ff 100%);
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #10162f 0%, #202b55 100%);
}

[data-testid="stSidebar"] * {
    color: #f3f5ff !important;
}

.hero {
    padding: 32px;
    border-radius: 24px;
    background: linear-gradient(120deg, #303f9f, #7654d6, #3684d8);
    color: white;
    margin-bottom: 25px;
    box-shadow: 0 12px 35px rgba(67, 74, 180, 0.18);
}

.hero h1 {
    color: white;
    font-size: 38px;
    margin-bottom: 8px;
    font-weight: 800;
}

.hero p {
    color: #f1f2ff;
    font-size: 16px;
    margin-bottom: 0;
}

.metric-card {
    background: white;
    border: 1px solid #e5e9fb;
    padding: 18px;
    border-radius: 17px;
    text-align: center;
    box-shadow: 0 5px 20px rgba(40, 55, 110, 0.05);
}

.metric-number {
    color: #4d50b8;
    font-size: 26px;
    font-weight: 800;
}

.metric-label {
    color: #69718a;
    font-size: 13px;
}

.section-title {
    color: #20294b;
    font-size: 21px;
    font-weight: 750;
    margin: 20px 0 12px 0;
}

div[data-testid="stChatMessage"] {
    background: white;
    border: 1px solid #e8ebf8;
    border-radius: 16px;
    padding: 12px;
    margin-bottom: 12px;
    box-shadow: 0 4px 14px rgba(36, 45, 95, 0.04);
}

.stButton button {
    border-radius: 12px;
    border: 1px solid #dce2fb;
    background: white;
    color: #343c79;
    font-weight: 600;
    min-height: 44px;
}

.stButton button:hover {
    border-color: #7065d9;
    color: #4c40bc;
}

div[data-testid="stChatInput"] {
    border-radius: 18px;
}

.footer {
    text-align: center;
    color: #8189a2;
    font-size: 12px;
    padding: 24px 0 8px 0;
}
</style>
""", unsafe_allow_html=True)


# -------------------- LOAD FILES --------------------
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
    st.info("Open the VS Code terminal and run `python train_model.py`.")
    st.stop()

intents_data = load_intents()
model = bundle["model"]
encoder = bundle["encoder"]
responses = bundle["responses"]
threshold = bundle["threshold"]

intent_count = len([
    intent for intent in intents_data["intents"]
    if intent["tag"] != "fallback"
])

pattern_count = sum(
    len(intent["patterns"])
    for intent in intents_data["intents"]
    if intent["tag"] != "fallback"
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
        st.rerun()

    st.divider()
    st.caption("Nexa AI · Student AI/ML Project")


# -------------------- MAIN HEADER --------------------
st.markdown("""
<div class="hero">
    <h1>Meet Nexa AI ✨</h1>
    <p>Your friendly AI learning companion — ask questions, explore concepts,
    and make learning a little easier.</p>
</div>
""", unsafe_allow_html=True)

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
st.markdown('<div class="section-title">💡 Try asking me</div>',
            unsafe_allow_html=True)

quick_prompts = [
    ("🐍 Learn Python", "What is Python?"),
    ("🧠 Explore ML", "What is machine learning?"),
    ("💬 Discover NLP", "What is NLP?"),
    ("📚 Study smarter", "Give me study tips")
]

prompt_columns = st.columns(4)

if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None

for index, (label, prompt) in enumerate(quick_prompts):
    with prompt_columns[index]:
        if st.button(label, key=f"quick_{index}", use_container_width=True):
            st.session_state.pending_prompt = prompt


# -------------------- CHAT HISTORY --------------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Hello! 👋 I'm Nexa AI. Ask me about Python, machine learning, "
                "NLP, study tips, or motivation."
            ),
            "tag": "greeting",
            "confidence": None
        }
    ]

st.markdown('<div class="section-title">💬 Your conversation</div>',
            unsafe_allow_html=True)

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

if user_message:
    st.session_state.messages.append({
        "role": "user",
        "content": user_message
    })

    answer, tag, confidence = get_response(user_message)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "tag": tag,
        "confidence": confidence
    })

    st.rerun()


st.markdown("""
<div class="footer">
    Made with Python & Streamlit · Nexa AI Smart Chatbot
</div>
""", unsafe_allow_html=True)
