import streamlit as st
from rag.engine import GroundedEngine


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Citizen Policy Assistant",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# CUSTOM CSS
# =========================================================

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


# =========================================================
# ENGINE
# =========================================================

@st.cache_resource
def get_engine():
    return GroundedEngine("data")


try:
    engine = get_engine()

except Exception as e:

    st.error(f"Startup error: {e}")
    st.stop()


# =========================================================
# SESSION STATE
# =========================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "last_result" not in st.session_state:
    st.session_state.last_result = None

if "guided_question" not in st.session_state:
    st.session_state.guided_question = None


# =========================================================
# HEADER
# =========================================================

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


# =========================================================
# SIDEBAR
# =========================================================

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
        "income thresholds, benefit amounts, sanctions, "
        "and policy rules."
    )

    st.divider()

    st.header("🛡️ Grounded AI")

    st.write(
        "This assistant answers only when the available "
        "policy evidence supports the response."
    )

    st.write(
        "If the policy does not provide enough evidence, "
        "the assistant will say: "
        "\"I don't know.\""
    )

    st.divider()

    if st.button(
        "🗑️ Clear conversation",
        use_container_width=True,
    ):

        st.session_state.messages = []
        st.session_state.last_result = None
        st.session_state.guided_question = None

        st.rerun()


# =========================================================
# LANGUAGE INTRO
# =========================================================

if language == "தமிழ்":

    st.info(
        "👋 வணக்கம்! Policy பற்றிய கேள்வியை English அல்லது "
        "Tamil-ல் கேட்கலாம். கிடைக்கும் policy evidence-ஐ "
        "வைத்து மட்டுமே பதில் வழங்கப்படும்."
    )

else:

    st.info(
        "👋 Welcome! You can ask your policy question in "
        "simple English. The assistant retrieves supporting "
        "policy evidence before answering."
    )


# =========================================================
# CITIZEN GUIDED MODE
# =========================================================

st.subheader("👋 How can I help you?")

citizen_options = {

    "🟢 Check my eligibility":
        "eligibility",

    "💰 How much can I get?":
        "amount",

    "📅 Report a change":
        "report_change",

    "⚠️ Understand a sanction":
        "sanction",

    "📖 Understand a policy rule":
        "policy",

    "❓ I don't know what to ask":
        "help",
}


selected_help = st.selectbox(
    "Choose what you need help with",
    ["Select an option"] + list(citizen_options.keys()),
)


guided_question = None


# =========================================================
# GUIDED MODE
# =========================================================

if selected_help != "Select an option":

    mode = citizen_options[selected_help]


    # =====================================================
    # ELIGIBILITY
    # =====================================================

    if mode == "eligibility":

        st.markdown(
            "#### 🟢 Tell us about your situation"
        )

        household_size = st.number_input(
            "How many people are in your household?",
            min_value=1,
            max_value=20,
            value=1,
            step=1,
        )

        monthly_income = st.number_input(
            "What is your approximate monthly household income?",
            min_value=0,
            value=0,
            step=50,
        )

        determination_date = st.date_input(
            "Determination date",
        )

        guided_question = (
            f"What are the eligibility requirements for a "
            f"household of {household_size} with a monthly "
            f"income of ${monthly_income:,} for a determination "
            f"made in {determination_date.strftime('%B %Y')}?"
        )


    # =====================================================
    # BENEFIT AMOUNT
    # =====================================================

    elif mode == "amount":

        st.markdown(
            "#### 💰 Tell us about your household"
        )

        household_size = st.number_input(
            "Household size",
            min_value=1,
            max_value=20,
            value=3,
            step=1,
            key="amount_household",
        )

        determination_date = st.date_input(
            "Determination date",
            key="amount_date",
        )

        guided_question = (
            f"What is the maximum benefit for a household "
            f"of {household_size} for a determination made "
            f"in {determination_date.strftime('%B %Y')}?"
        )


    # =====================================================
    # REPORT A CHANGE
    # =====================================================

    elif mode == "report_change":

        st.markdown(
            "#### 📅 What kind of change happened?"
        )

        change_type = st.radio(
            "Choose one",
            [
                "💼 I started a job",
                "💰 My income changed",
                "🏠 I moved house",
                "👨‍👩‍👧 Someone joined or left my household",
                "❓ Something else",
            ],
        )

        change_date = st.date_input(
            "When did the change happen?",
            key="change_date",
        )


        if change_type == "💼 I started a job":

            change_description = "I started a job"

        elif change_type == "💰 My income changed":

            change_description = "My income changed"

        elif change_type == "🏠 I moved house":

            change_description = "I moved house"

        elif change_type == "👨‍👩‍👧 Someone joined or left my household":

            change_description = (
                "Someone joined or left my household"
            )

        else:

            change_description = (
                "Something else changed in my circumstances"
            )


        guided_question = (
            f"{change_description}. "
            f"The change occurred in "
            f"{change_date.strftime('%B %Y')}. "
            f"How long do I have to report this change "
            f"of circumstances?"
        )


    # =====================================================
    # SANCTION
    # =====================================================

    elif mode == "sanction":

        st.markdown(
            "#### ⚠️ Tell us about the sanction situation"
        )

        sanction_date = st.date_input(
            "When did the change of circumstances occur?",
            key="sanction_date",
        )

        would_increase = st.radio(
            "Would the change have increased the benefit award?",
            [
                "Yes",
                "No",
                "I don't know",
            ],
        )


        if would_increase == "Yes":

            guided_question = (
                "What sanction applies when the unreported "
                "change would have increased the award?"
            )

        elif would_increase == "No":

            guided_question = (
                f"What sanction applies to a change of "
                f"circumstances occurring in "
                f"{sanction_date.strftime('%B %Y')}?"
            )

        else:

            guided_question = (
                f"What sanction rules apply to a change "
                f"of circumstances occurring in "
                f"{sanction_date.strftime('%B %Y')}?"
            )


    # =====================================================
    # POLICY RULE
    # =====================================================

    elif mode == "policy":

        st.markdown(
            "#### 📖 What policy rule do you want to understand?"
        )

        policy_topic = st.selectbox(
            "Choose a topic",
            [
                "Income threshold",
                "Earnings disregard",
                "Maximum benefit",
                "Reporting period",
                "Sanction",
                "Other policy rule",
            ],
        )


        if policy_topic == "Income threshold":

            household_size = st.number_input(
                "Household size",
                min_value=1,
                max_value=20,
                value=3,
                step=1,
                key="policy_income_size",
            )

            policy_date = st.date_input(
                "Determination date",
                key="policy_income_date",
            )

            guided_question = (
                f"What is the income threshold for a "
                f"household of {household_size} for a "
                f"determination made in "
                f"{policy_date.strftime('%B %Y')}?"
            )


        elif policy_topic == "Earnings disregard":

            policy_date = st.date_input(
                "Determination date",
                key="policy_earnings_date",
            )

            guided_question = (
                f"What is the earnings disregard for a "
                f"determination made in "
                f"{policy_date.strftime('%B %Y')}?"
            )


        elif policy_topic == "Maximum benefit":

            household_size = st.number_input(
                "Household size",
                min_value=1,
                max_value=20,
                value=3,
                step=1,
                key="policy_benefit_size",
            )

            policy_date = st.date_input(
                "Determination date",
                key="policy_benefit_date",
            )

            guided_question = (
                f"What is the maximum benefit for a "
                f"household of {household_size} for a "
                f"determination made in "
                f"{policy_date.strftime('%B %Y')}?"
            )


        elif policy_topic == "Reporting period":

            policy_date = st.date_input(
                "When did the change occur?",
                key="policy_report_date",
            )

            guided_question = (
                f"A change of circumstances occurred in "
                f"{policy_date.strftime('%B %Y')}. "
                f"How long do I have to report it?"
            )


        elif policy_topic == "Sanction":

            policy_date = st.date_input(
                "When did the change occur?",
                key="policy_sanction_date",
            )

            guided_question = (
                f"What sanction rules apply to a change "
                f"of circumstances occurring in "
                f"{policy_date.strftime('%B %Y')}?"
            )


        else:

            st.info(
                "You can type your own question in the "
                "chat box below."
            )


    # =====================================================
    # HELP
    # =====================================================

    elif mode == "help":

        st.info(
            "💡 You can ask about:\n\n"
            "- Eligibility\n"
            "- Maximum benefit amount\n"
            "- Income threshold\n"
            "- Earnings disregard\n"
            "- Reporting deadlines\n"
            "- Sanctions\n"
            "- Policy rules\n\n"
            "You can also type your own question below."
        )


# =========================================================
# GUIDED QUESTION PREVIEW
# =========================================================

if guided_question:

    st.markdown("#### 📝 Your question")

    st.info(guided_question)

    if st.button(
        "🔎 Get answer",
        type="primary",
        use_container_width=True,
        key="guided_answer_button",
    ):

        st.session_state.guided_question = guided_question

        st.rerun()


# =========================================================
# QUICK QUESTIONS
# =========================================================

st.subheader("💬 Example questions")

quick_questions = [

    "What is the earnings disregard for a determination made in April 2026?",

    "A change of circumstances occurred in February 2026. How long do I have to report it?",

    "A change of circumstances occurred in April 2026. How long do I have to report it?",

    "What is the income threshold for a household of 3 for a determination made in April 2026?",

    "What is the maximum benefit for a household of 3?",

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


# =========================================================
# CHAT HISTORY
# =========================================================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])


# =========================================================
# QUESTION INPUT
# =========================================================

question = st.chat_input(
    "Type your policy question here..."
)


# =========================================================
# QUESTION PRIORITY
# =========================================================

if st.session_state.guided_question:

    question = st.session_state.guided_question

    st.session_state.guided_question = None

elif selected_question:

    question = selected_question


# =========================================================
# PROCESS QUESTION
# =========================================================

if question and question.strip():

    question = question.strip()


    # -----------------------------------------------------
    # Store user message
    # -----------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )


    with st.chat_message("user"):

        st.markdown(question)


    # -----------------------------------------------------
    # Generate answer
    # -----------------------------------------------------

    with st.chat_message("assistant"):

        with st.spinner(
            "🔎 Checking the policy evidence..."
        ):

            try:

                result = engine.ask(question)

            except Exception as e:

                st.error(
                    f"Error while processing the question: {e}"
                )

                st.stop()


        st.session_state.last_result = result


        # =================================================
        # ANSWER
        # =================================================

        st.markdown("### 💬 Answer")


        st.markdown(
            '<div class="answer-card">',
            unsafe_allow_html=True,
        )


        st.write(result["answer"])


        st.markdown(
            "</div>",
            unsafe_allow_html=True,
        )


        # =================================================
        # STATUS
        # =================================================

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


        # =================================================
        # GROUNDED / ABSTENTION MESSAGE
        # =================================================

        if result["grounded"]:

            st.markdown(
                """
                <div class="grounded-box">
                ✅ <b>Grounded answer</b><br>
                This response is supported by retrieved
                policy evidence.
                </div>
                """,
                unsafe_allow_html=True,
            )

        else:

            st.markdown(
                """
                <div class="warning-box">
                ⚠️ <b>I don't know</b><br>
                The available policy evidence does not provide
                enough support for a reliable answer.
                The assistant will not intentionally guess.
                </div>
                """,
                unsafe_allow_html=True,
            )


        # =================================================
        # OFFICIAL SUPPORTING CLAUSE
        # =================================================

        if result["citation"]:

            st.markdown(
                "### 📖 Official supporting clause"
            )


            with st.expander(
                "View the exact policy clause",
                expanded=True,
            ):

                st.code(
                    result["citation"],
                    language="text",
                )


        # =================================================
        # EFFECTIVE RULE
        # =================================================

        if result["effective_rule"]:

            st.markdown(
                "### 📅 Why this rule applies"
            )


            with st.expander(
                "View policy-date reasoning",
                expanded=True,
            ):

                st.write(
                    result["effective_rule"]
                )


        # =================================================
        # RETRIEVED EVIDENCE
        # =================================================

        if result["evidence"]:

            st.markdown(
                "### 🔎 Retrieved evidence"
            )


            for i, item in enumerate(
                result["evidence"],
                start=1,
            ):

                score = item.get(
                    "score",
                    0
                )


                with st.expander(
                    f"Evidence {i} • "
                    f"{item['section']} • "
                    f"relevance {score:.3f}"
                ):

                    st.write(
                        item["text"]
                    )


        # =================================================
        # SAFE NEXT STEP
        # =================================================

        st.markdown(
            "### 👉 What you can do next"
        )


        if result["grounded"]:

            st.info(
                "Review the supporting clause above and use "
                "the policy requirement that applies to your "
                "situation. If your situation is different "
                "from the evidence shown here, contact the "
                "appropriate benefits office."
            )

        else:

            st.warning(
                "Because the available policy does not support "
                "a reliable answer, contact the appropriate "
                "benefits policy office rather than relying "
                "on an assumption."
            )


    # =====================================================
    # STORE ASSISTANT RESPONSE
    # =====================================================

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": result["answer"],
        }
    )


# =========================================================
# FOOTER
# =========================================================

st.divider()


st.caption(
    "🏛️ Citizen Policy Assistant • "
    "Evidence-grounded policy support • "
    "Designed to avoid unsupported answers"
)