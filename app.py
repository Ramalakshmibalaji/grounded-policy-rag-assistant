import streamlit as st
from rag.engine import GroundedEngine

# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------
st.set_page_config(
    page_title="Citizen Policy Assistant",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------------
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        font-size: 1.05rem;
        color: #666;
        margin-bottom: 1.5rem;
    }

    .answer-card {
        padding: 1.2rem;
        border-radius: 12px;
        border: 1px solid #ddd;
        background-color: #fafafa;
        margin-bottom: 1rem;
    }

    .grounded-box {
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #b7dfc1;
        background-color: #f2fff5;
    }

    .warning-box {
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #f0c36d;
        background-color: #fff9ed;
    }

    .section-title {
        font-size: 1.2rem;
        font-weight: 600;
        margin-top: 1rem;
    }

    .small-text {
        font-size: 0.9rem;
        color: #666;
    }

    div[data-testid="stMetric"] {
        border: 1px solid #ddd;
        padding: 12px;
        border-radius: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# ENGINE
# ---------------------------------------------------------
@st.cache_resource
def get_engine():
    return GroundedEngine("data")


try:
    engine = get_engine()
except Exception as e:
    st.error(f"Startup error: {e}")
    st.stop()


# ---------------------------------------------------------
# SESSION STATE
# ---------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "last_result" not in st.session_state:
    st.session_state.last_result = None


# ---------------------------------------------------------
# HEADER
# ---------------------------------------------------------
st.markdown(
    '<div class="main-title">🏛️ Citizen Policy Assistant</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="subtitle">
    Ask questions about benefits policies in simple language.
    The assistant answers only from the available policy evidence.
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------
with st.sidebar:
    st.header("🌐 Language")

    language = st.radio(
        "Choose your preferred language",
        ["English", "தமிழ்"],
        index=0,
    )

    st.divider()

    st.header("💡 What can I ask?")

    st.caption(
        "You can ask about eligibility, reporting deadlines, "
        "income thresholds, sanctions, and policy rules."
    )

    st.divider()

    st.header("🛡️ Grounded AI")

    st.write(
        "This assistant does not intentionally guess when "
        "the available policy evidence does not support an answer."
    )

    st.divider()

    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.last_result = None
        st.rerun()


# ---------------------------------------------------------
# LANGUAGE INTRO
# ---------------------------------------------------------
if language == "தமிழ்":
    st.info(
        "👋 வணக்கம்! Policy பற்றிய கேள்வியை English அல்லது Tamil-ல் "
        "type செய்து கேட்கலாம். கிடைக்கும் policy evidence-ஐ வைத்து "
        "பதில் வழங்கப்படும்."
    )
else:
    st.info(
        "👋 Welcome! You can ask your policy question in simple English. "
        "The assistant retrieves supporting policy evidence before answering."
    )


# ---------------------------------------------------------
# QUICK QUESTIONS
# ---------------------------------------------------------
st.subheader("💬 Start with a question")

quick_questions = [
    "What is the earnings disregard for a determination made in April 2026?",
    "A change of circumstances occurred in February 2026. How long do I have to report it?",
    "What is the income threshold for a household of 3 for a determination made in April 2026?",
    "What sanction applies when the unreported change would have increased the award?",
]

cols = st.columns(2)

selected_question = None

for i, q in enumerate(quick_questions):
    with cols[i % 2]:
        if st.button(
            q,
            key=f"quick_{i}",
            use_container_width=True,
        ):
            selected_question = q


# ---------------------------------------------------------
# CHAT HISTORY
# ---------------------------------------------------------
for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# ---------------------------------------------------------
# QUESTION INPUT
# ---------------------------------------------------------
question = st.chat_input(
    "Type your policy question here..."
)

if selected_question:
    question = selected_question


# ---------------------------------------------------------
# PROCESS QUESTION
# ---------------------------------------------------------
if question and question.strip():

    question = question.strip()

    # Store user message
    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):

        with st.spinner("🔎 Checking the policy evidence..."):
            try:
                result = engine.ask(question)
            except Exception as e:
                st.error(f"Error while processing the question: {e}")
                st.stop()

        st.session_state.last_result = result

        # -------------------------------------------------
        # ANSWER
        # -------------------------------------------------
        st.markdown("### 💬 Answer")

        st.markdown(
            '<div class="answer-card">',
            unsafe_allow_html=True,
        )

        st.write(result["answer"])

        st.markdown("</div>", unsafe_allow_html=True)

        # -------------------------------------------------
        # STATUS
        # -------------------------------------------------
        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "🎯 Confidence",
                result["confidence"],
            )

        with col2:
            grounded_text = (
                "✅ Grounded"
                if result["grounded"]
                else "⚠️ Not grounded"
            )

            st.metric(
                "📚 Evidence status",
                grounded_text,
            )

        # -------------------------------------------------
        # GROUNDED / ABSTENTION MESSAGE
        # -------------------------------------------------
        if result["grounded"]:

            st.markdown(
                """
                <div class="grounded-box">
                ✅ <b>Grounded answer</b><br>
                This response is supported by retrieved policy evidence.
                </div>
                """,
                unsafe_allow_html=True,
            )

        else:

            st.markdown(
                """
                <div class="warning-box">
                ⚠️ <b>I don't know</b><br>
                The available policy evidence does not provide enough
                support for a reliable answer. The assistant will not
                intentionally guess.
                </div>
                """,
                unsafe_allow_html=True,
            )

        # -------------------------------------------------
        # EVIDENCE
        # -------------------------------------------------
        if result["citation"]:

            st.markdown("### 📖 Official supporting clause")

            with st.expander(
                "View the exact policy clause",
                expanded=True,
            ):
                st.code(
                    result["citation"],
                    language="text",
                )

        # -------------------------------------------------
        # EFFECTIVE RULE
        # -------------------------------------------------
        if result["effective_rule"]:

            st.markdown("### 📅 Why this rule applies")

            with st.expander(
                "View policy-date reasoning",
                expanded=True,
            ):
                st.write(result["effective_rule"])

        # -------------------------------------------------
        # RETRIEVED EVIDENCE
        # -------------------------------------------------
        if result["evidence"]:

            st.markdown("### 🔎 Retrieved evidence")

            for i, item in enumerate(
                result["evidence"],
                start=1,
            ):

                score = item.get("score", 0)

                with st.expander(
                    f"Evidence {i} • {item['section']} • relevance {score:.3f}"
                ):

                    st.write(item["text"])

        # -------------------------------------------------
        # SAFE NEXT STEP
        # -------------------------------------------------
        st.markdown("### 👉 What you can do next")

        if result["grounded"]:

            st.info(
                "Review the supporting clause above and use the "
                "policy requirement that applies to your situation. "
                "If your situation is different from the evidence "
                "shown here, contact the appropriate benefits office."
            )

        else:

            st.warning(
                "Because the available policy does not support a "
                "reliable answer, contact the appropriate benefits "
                "policy office rather than relying on an assumption."
            )

    # Store assistant response in chat history
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": result["answer"],
        }
    )


# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------
st.divider()

st.caption(
    "🏛️ Citizen Policy Assistant • Evidence-grounded policy support • "
    "Designed to avoid unsupported answers"
)