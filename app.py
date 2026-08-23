import streamlit as st
from pathlib import Path
import pandas as pd
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
# LOAD CSS
# =========================================================

def load_css():

    css_path = Path(__file__).parent / "assests" / "style.css"

    if css_path.exists():

        with open(css_path, "r", encoding="utf-8") as f:

            st.markdown(
                f"<style>{f.read()}</style>",
                unsafe_allow_html=True,
            )


load_css()


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

defaults = {

    "page": "Dashboard",

    "messages": [],

    "last_result": None,

    "guided_question": None,

    "answer_count": 0,

    "grounded_count": 0,

    "abstained_count": 0,

}


for key, value in defaults.items():

    if key not in st.session_state:

        st.session_state[key] = value


# =========================================================
# HELPERS
# =========================================================

def navigate(page):

    st.session_state.page = page

    st.rerun()


def clear_conversation():

    st.session_state.messages = []

    st.session_state.last_result = None

    st.session_state.guided_question = None

    st.session_state.answer_count = 0

    st.session_state.grounded_count = 0

    st.session_state.abstained_count = 0


def engine_value(names, default="—"):

    for name in names:

        value = getattr(engine, name, None)

        if value is not None:

            try:

                if name in {
                    "sections",
                    "policy_sections",
                    "chunks",
                    "documents",
                }:

                    return len(value)

                return value

            except TypeError:

                return value

    return default


def ask_question(question):

    st.session_state.guided_question = question

    st.rerun()


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown(
        """
        <div class="sidebar-brand">
            🏛️ <b>Citizen Policy Assistant</b>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.caption(
        "Simple, evidence-grounded policy support."
    )

    st.divider()

    st.markdown("### 🧭 Navigation")

    pages = [

        ("📊 Dashboard", "Dashboard"),

        ("💬 Ask Policy", "Ask Policy"),

        ("📈 Policy Insights", "Policy Insights"),

        ("🛡️ About", "About"),

    ]


    for label, page in pages:

        if st.button(

            label,

            use_container_width=True,

            type=(
                "primary"
                if st.session_state.page == page
                else "secondary"
            ),

            key=f"side_{page}",

        ):

            navigate(page)


    st.divider()

    st.markdown("### 🛡️ Grounded AI")

    st.caption(
        "The assistant uses retrieved policy evidence "
        "and avoids unsupported answers."
    )

    st.divider()

    if st.button(

        "🗑️ Clear Conversation",

        use_container_width=True,

    ):

        clear_conversation()

        st.rerun()


# =========================================================
# MAIN HEADER
# =========================================================
# IMPORTANT:
# Do NOT use HTML here.
# This fixes the raw <div class="subtitle"> problem.
# =========================================================

st.markdown(
    '<div class="main-title">🏛️ Citizen Policy Assistant</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">'
    'Understand benefits policy in simple language, '
    'backed by official policy evidence.'
    '</div>',
    unsafe_allow_html=True,
)


# =========================================================
# TOP NAVIGATION
# =========================================================

nav = st.columns(4)

navigation_items = [

    ("📊 Dashboard", "Dashboard"),

    ("💬 Ask Policy", "Ask Policy"),

    ("📈 Policy Insights", "Policy Insights"),

    ("🛡️ About", "About"),

]


for col, (label, page) in zip(
    nav,
    navigation_items,
):

    with col:

        if st.button(

            label,

            use_container_width=True,

            key=f"top_{page}",

        ):

            navigate(page)


st.divider()


# =========================================================
# DASHBOARD
# =========================================================

def dashboard():

    st.markdown("## 📊 Policy Dashboard")

    st.caption(
        "A quick overview of policy content and "
        "assistant performance."
    )


    # -----------------------------------------------------
    # KPI CARDS
    # -----------------------------------------------------

    policy_sections = engine_value(
        [
            "sections",
            "policy_sections",
            "chunks",
            "documents",
        ]
    )


    total = st.session_state.answer_count

    grounded = st.session_state.grounded_count

    abstained = st.session_state.abstained_count


    c1, c2, c3, c4 = st.columns(4)


    with c1:

        st.metric(
            "📚 Policy Content",
            policy_sections,
        )


    with c2:

        st.metric(
            "💬 Questions Asked",
            total,
        )


    with c3:

        st.metric(
            "🛡️ Grounded Answers",
            grounded,
        )


    with c4:

        st.metric(
            "⚠️ More Evidence Needed",
            abstained,
        )


    st.divider()


    # -----------------------------------------------------
    # ANSWER SUPPORT
    # -----------------------------------------------------

    left, right = st.columns(
        [1.5, 1]
    )


    with left:

        st.markdown(
            "### 📈 Answer Support Overview"
        )


        if total > 0:

            df = pd.DataFrame(
                {
                    "Status": [
                        "Grounded",
                        "I don't know",
                    ],

                    "Answers": [
                        grounded,
                        abstained,
                    ],
                }
            ).set_index("Status")


            st.bar_chart(df)


            percentage = (
                grounded / total
            ) * 100


            st.caption(
                f"Grounding rate: {percentage:.0f}%"
            )


        else:

            st.info(
                "No questions answered yet. "
                "Go to **Ask Policy** to begin."
            )


    # -----------------------------------------------------
    # LATEST STATUS
    # -----------------------------------------------------

    with right:

        st.markdown(
            "### 🛡️ Latest AI Status"
        )


        result = st.session_state.last_result


        if result is None:

            st.success(
                "🟢 System Ready"
            )

            st.write(
                "Ask a policy question and the assistant "
                "will retrieve supporting evidence."
            )


        elif result.get(
            "grounded",
            False,
        ):

            st.success(
                "✅ Grounded Answer"
            )

            st.metric(
                "Confidence",
                str(
                    result.get(
                        "confidence",
                        "low",
                    )
                ).title(),
            )

            st.caption(
                "Supporting policy evidence was found."
            )


        else:

            st.warning(
                "⚠️ More Evidence Needed"
            )

            st.metric(
                "Confidence",
                str(
                    result.get(
                        "confidence",
                        "low",
                    )
                ).title(),
            )

            st.caption(
                "The assistant avoided guessing."
            )


    st.divider()


    # -----------------------------------------------------
    # RECENT QUESTIONS
    # -----------------------------------------------------

    st.markdown(
        "### 🕒 Recent Questions"
    )


    recent_questions = [

        message["content"]

        for message in st.session_state.messages

        if message.get("role") == "user"

    ]


    if recent_questions:

        for i, question in enumerate(
            recent_questions[-5:][::-1],
            start=1,
        ):

            st.markdown(
                f"""
                <div class="question-item">
                    <b>{i}.</b> {question}
                </div>
                """,
                unsafe_allow_html=True,
            )


    else:

        st.caption(
            "Your recent policy questions will appear here."
        )


    st.divider()


    # -----------------------------------------------------
    # QUICK ACTIONS
    # -----------------------------------------------------

    st.markdown(
        "### 🚀 Quick Actions"
    )


    q1, q2, q3 = st.columns(3)


    with q1:

        if st.button(
            "💬 Ask Policy",
            use_container_width=True,
            type="primary",
        ):

            navigate("Ask Policy")


    with q2:

        if st.button(
            "📈 Policy Insights",
            use_container_width=True,
        ):

            navigate("Policy Insights")


    with q3:

        if st.button(
            "🛡️ How It Works",
            use_container_width=True,
        ):

            navigate("About")


# =========================================================
# ASK POLICY
# =========================================================

def ask_policy():

    st.markdown(
        "## 💬 Ask Policy"
    )

    st.caption(
        "Ask your question in simple language. "
        "The assistant retrieves policy evidence before answering."
    )


    # -----------------------------------------------------
    # TOPIC SELECTOR
    # -----------------------------------------------------

    st.markdown(
        "### 👋 How can I help you?"
    )


    options = {

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


    selected = st.selectbox(

        "Choose a topic",

        ["Select an option"]
        +
        list(options.keys()),

        key="help_topic",

    )


    guided_question = None


    if selected != "Select an option":

        mode = options[selected]


        # -------------------------------------------------
        # ELIGIBILITY
        # -------------------------------------------------

        if mode == "eligibility":

            st.markdown(
                "#### 🟢 Eligibility"
            )


            size = st.number_input(

                "Household size",

                1,
                20,
                1,

                key="elig_size",

            )


            income = st.number_input(

                "Approximate monthly household income",

                min_value=0,

                value=0,

                step=50,

                key="elig_income",

            )


            dt = st.date_input(

                "Determination date",

                key="elig_date",

            )


            guided_question = (

                f"What are the eligibility requirements "
                f"for a household of {size} with a monthly "
                f"income of ${income:,} for a determination "
                f"made in {dt.strftime('%B %Y')}?"

            )


        # -------------------------------------------------
        # AMOUNT
        # -------------------------------------------------

        elif mode == "amount":

            st.markdown(
                "#### 💰 Benefit Amount"
            )


            size = st.number_input(

                "Household size",

                1,
                20,
                3,

                key="amount_size",

            )


            dt = st.date_input(

                "Determination date",

                key="amount_date",

            )


            guided_question = (

                f"What is the maximum benefit for a "
                f"household of {size} for a determination "
                f"made in {dt.strftime('%B %Y')}?"

            )


        # -------------------------------------------------
        # REPORT CHANGE
        # -------------------------------------------------

        elif mode == "report_change":

            st.markdown(
                "#### 📅 Report a Change"
            )


            change = st.radio(

                "What changed?",

                [
                    "💼 I started a job",
                    "💰 My income changed",
                    "🏠 I moved house",
                    "👨‍👩‍👧 Someone joined or left my household",
                    "❓ Something else",
                ],

                key="change_type",

            )


            dt = st.date_input(

                "When did the change happen?",

                key="change_date",

            )


            desc = {

                "💼 I started a job":
                    "I started a job",

                "💰 My income changed":
                    "My income changed",

                "🏠 I moved house":
                    "I moved house",

                "👨‍👩‍👧 Someone joined or left my household":
                    "Someone joined or left my household",

                "❓ Something else":
                    "Something else changed in my circumstances",

            }[change]


            guided_question = (

                f"{desc}. "

                f"The change occurred in "
                f"{dt.strftime('%B %Y')}. "

                f"How long do I have to report this "
                f"change of circumstances?"

            )


        # -------------------------------------------------
        # SANCTION
        # -------------------------------------------------

        elif mode == "sanction":

            st.markdown(
                "#### ⚠️ Sanction"
            )


            dt = st.date_input(

                "When did the change occur?",

                key="sanction_date",

            )


            increase = st.radio(

                "Would the change have increased the benefit award?",

                [
                    "Yes",
                    "No",
                    "I don't know",
                ],

                key="sanction_increase",

            )


            if increase == "Yes":

                guided_question = (
                    "What sanction applies when the "
                    "unreported change would have "
                    "increased the award?"
                )


            elif increase == "No":

                guided_question = (

                    f"What sanction applies to a change "
                    f"of circumstances occurring in "
                    f"{dt.strftime('%B %Y')}?"

                )


            else:

                guided_question = (

                    f"What sanction rules apply to a "
                    f"change of circumstances occurring "
                    f"in {dt.strftime('%B %Y')}?"

                )


        # -------------------------------------------------
        # POLICY
        # -------------------------------------------------

        elif mode == "policy":

            st.markdown(
                "#### 📖 Policy Rule"
            )


            topic = st.selectbox(

                "Choose a policy topic",

                [
                    "Income threshold",
                    "Earnings disregard",
                    "Maximum benefit",
                    "Reporting period",
                    "Sanction",
                    "Other policy rule",
                ],

                key="policy_topic",

            )


            if topic == "Income threshold":

                size = st.number_input(

                    "Household size",

                    1,
                    20,
                    3,

                    key="policy_size",

                )


                dt = st.date_input(

                    "Determination date",

                    key="policy_income_date",

                )


                guided_question = (

                    f"What is the income threshold for "
                    f"a household of {size} for a "
                    f"determination made in "
                    f"{dt.strftime('%B %Y')}?"

                )


            elif topic == "Earnings disregard":

                dt = st.date_input(

                    "Determination date",

                    key="policy_earnings_date",

                )


                guided_question = (

                    f"What is the earnings disregard for "
                    f"a determination made in "
                    f"{dt.strftime('%B %Y')}?"

                )


            elif topic == "Maximum benefit":

                size = st.number_input(

                    "Household size",

                    1,
                    20,
                    3,

                    key="policy_benefit_size",

                )


                dt = st.date_input(

                    "Determination date",

                    key="policy_benefit_date",

                )


                guided_question = (

                    f"What is the maximum benefit for "
                    f"a household of {size} for a "
                    f"determination made in "
                    f"{dt.strftime('%B %Y')}?"

                )


            elif topic == "Reporting period":

                dt = st.date_input(

                    "When did the change occur?",

                    key="policy_report_date",

                )


                guided_question = (

                    f"A change of circumstances occurred "
                    f"in {dt.strftime('%B %Y')}. "
                    f"How long do I have to report it?"

                )


            elif topic == "Sanction":

                dt = st.date_input(

                    "When did the change occur?",

                    key="policy_sanction_date",

                )


                guided_question = (

                    f"What sanction rules apply to a "
                    f"change of circumstances occurring "
                    f"in {dt.strftime('%B %Y')}?"

                )


            else:

                st.info(
                    "Use the chat box below to ask "
                    "your own policy question."
                )


        # -------------------------------------------------
        # HELP
        # -------------------------------------------------

        else:

            st.info(

                "You can ask about eligibility, benefit "
                "amounts, income thresholds, earnings "
                "disregard, reporting deadlines, sanctions, "
                "and policy rules."

            )


    # -----------------------------------------------------
    # GENERATED QUESTION
    # -----------------------------------------------------

    if guided_question:

        st.markdown(
            "#### 📝 Your question"
        )

        st.info(
            guided_question
        )


        if st.button(

            "🔎 Get Answer",

            type="primary",

            use_container_width=True,

            key="guided_answer",

        ):

            ask_question(
                guided_question
            )


    # -----------------------------------------------------
    # EXAMPLE QUESTIONS
    # -----------------------------------------------------

    st.markdown(
        "### 💬 Example Questions"
    )


    quick_questions = [

        "What is the earnings disregard for a determination made in April 2026?",

        "A change of circumstances occurred in February 2026. How long do I have to report it?",

        "A change of circumstances occurred in April 2026. How long do I have to report it?",

        "What is the income threshold for a household of 3 for a determination made in April 2026?",

        "What is the maximum benefit for a household of 3?",

        "What sanction applies when the unreported change would have increased the award?",

    ]


    cols = st.columns(2)


    for i, q in enumerate(
        quick_questions
    ):

        with cols[i % 2]:

            if st.button(

                q,

                key=f"quick_{i}",

                use_container_width=True,

            ):

                ask_question(q)


    # -----------------------------------------------------
    # CHAT HISTORY
    # -----------------------------------------------------

    for message in st.session_state.messages:

        with st.chat_message(
            message["role"]
        ):

            st.markdown(
                message["content"]
            )


    # -----------------------------------------------------
    # CHAT INPUT
    # -----------------------------------------------------

    question = st.chat_input(
        "Type your policy question here..."
    )


    if st.session_state.guided_question:

        question = st.session_state.guided_question

        st.session_state.guided_question = None


    if question and question.strip():

        process_question(
            question.strip()
        )


# =========================================================
# PROCESS QUESTION
# =========================================================

def process_question(question):

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )


    with st.chat_message("user"):

        st.markdown(question)


    with st.chat_message("assistant"):

        with st.spinner(
            "🔎 Checking policy evidence..."
        ):

            try:

                result = engine.ask(
                    question
                )

            except Exception as e:

                st.error(
                    f"Error while processing the question: {e}"
                )

                return


        st.session_state.last_result = result

        st.session_state.answer_count += 1


        grounded = result.get(
            "grounded",
            False,
        )


        if grounded:

            st.session_state.grounded_count += 1

        else:

            st.session_state.abstained_count += 1


        # -------------------------------------------------
        # ANSWER
        # -------------------------------------------------

        st.markdown(
            "### 💬 Answer"
        )


        st.markdown(
            '<div class="answer-card">',
            unsafe_allow_html=True,
        )


        st.write(
            result.get(
                "answer",
                "I don't know.",
            )
        )


        st.markdown(
            "</div>",
            unsafe_allow_html=True,
        )


        # -------------------------------------------------
        # STATUS
        # -------------------------------------------------

        c1, c2 = st.columns(2)


        with c1:

            st.metric(
                "🎯 Confidence",
                str(
                    result.get(
                        "confidence",
                        "low",
                    )
                ).title(),
            )


        with c2:

            st.metric(
                "📚 Evidence",
                (
                    "✅ Grounded"
                    if grounded
                    else
                    "⚠️ Not Grounded"
                ),
            )


        if grounded:

            st.success(
                "✅ This answer is supported by "
                "retrieved policy evidence."
            )

        else:

            st.warning(
                "⚠️ I don't know — the available policy "
                "evidence is not sufficient to provide "
                "a reliable answer."
            )


        # -------------------------------------------------
        # CITATION
        # -------------------------------------------------

        citation = result.get(
            "citation"
        )


        if citation:

            st.markdown(
                "### 📖 Official Supporting Clause"
            )


            with st.expander(
                "View exact policy clause",
                expanded=True,
            ):

                st.code(
                    citation,
                    language="text",
                )


        # -------------------------------------------------
        # EFFECTIVE RULE
        # -------------------------------------------------

        effective_rule = result.get(
            "effective_rule"
        )


        if effective_rule:

            st.markdown(
                "### 📅 Why This Rule Applies"
            )


            with st.expander(
                "View policy-date reasoning",
                expanded=True,
            ):

                st.write(
                    effective_rule
                )


        # -------------------------------------------------
        # EVIDENCE
        # -------------------------------------------------

        evidence = result.get(
            "evidence",
            [],
        )


        if evidence:

            st.markdown(
                "### 🔎 Retrieved Evidence"
            )


            for i, item in enumerate(
                evidence,
                1,
            ):

                score = item.get(
                    "score",
                    0,
                )


                section = item.get(
                    "section",
                    "Policy evidence",
                )


                with st.expander(

                    f"Evidence {i} • "
                    f"{section} • "
                    f"relevance {score:.3f}"

                ):

                    st.write(
                        item.get(
                            "text",
                            "",
                        )
                    )


        # -------------------------------------------------
        # NEXT STEP
        # -------------------------------------------------

        st.markdown(
            "### 👉 What You Can Do Next"
        )


        next_step = result.get(
            "next_step"
        )


        if grounded:

            st.info(

                next_step

                or

                "Review the supporting policy clause "
                "and contact the benefits office if needed."

            )

        else:

            st.warning(

                next_step

                or

                "Contact the appropriate county benefits "
                "policy office for further guidance."

            )


    # -----------------------------------------------------
    # STORE ASSISTANT MESSAGE
    # -----------------------------------------------------

    st.session_state.messages.append(

        {
            "role": "assistant",

            "content": result.get(
                "answer",
                "I don't know.",
            ),

        }

    )


# =========================================================
# POLICY INSIGHTS
# =========================================================

def policy_insights():

    st.markdown(
        "## 📈 Policy Insights"
    )


    st.caption(
        "Insights based on questions answered "
        "during the current session."
    )


    total = st.session_state.answer_count

    grounded = st.session_state.grounded_count

    abstained = st.session_state.abstained_count


    # -----------------------------------------------------
    # KPI
    # -----------------------------------------------------

    c1, c2, c3 = st.columns(3)


    c1.metric(
        "Total Questions",
        total,
    )


    c2.metric(
        "Grounded",
        grounded,
    )


    c3.metric(
        "More Evidence Needed",
        abstained,
    )


    st.divider()


    # -----------------------------------------------------
    # CHART
    # -----------------------------------------------------

    st.markdown(
        "### ⚖️ Grounded vs Abstained"
    )


    df = pd.DataFrame(

        {
            "Status": [
                "Grounded",
                "I don't know",
            ],

            "Answers": [
                grounded,
                abstained,
            ],

        }

    ).set_index(
        "Status"
    )


    st.bar_chart(df)


    # -----------------------------------------------------
    # LATEST EVIDENCE
    # -----------------------------------------------------

    result = st.session_state.last_result


    if result and result.get(
        "evidence"
    ):

        st.markdown(
            "### 🔎 Latest Evidence Relevance"
        )


        evidence = result["evidence"]


        relevance = pd.DataFrame(

            {
                "Evidence": [
                    f"Evidence {i}"
                    for i in range(
                        1,
                        len(evidence) + 1,
                    )
                ],

                "Relevance": [
                    item.get(
                        "score",
                        0,
                    )
                    for item in evidence
                ],

            }

        ).set_index(
            "Evidence"
        )


        st.bar_chart(
            relevance
        )


    # -----------------------------------------------------
    # LATEST RESULT
    # -----------------------------------------------------

    st.markdown(
        "### 📌 Latest Result"
    )


    if not result:

        st.info(
            "No policy question has been answered "
            "in this session yet."
        )


    else:

        c1, c2 = st.columns(2)


        c1.metric(
            "Confidence",
            str(
                result.get(
                    "confidence",
                    "low",
                )
            ).title(),
        )


        c2.metric(

            "Evidence Status",

            (
                "Grounded"
                if result.get(
                    "grounded",
                    False,
                )
                else
                "Not Grounded"
            ),

        )


        st.markdown(
            "**Answer**"
        )


        st.info(
            result.get(
                "answer",
                "I don't know.",
            )
        )


    st.caption(
        "Confidence is not treated as proof. "
        "Grounding depends on supporting policy evidence."
    )


# =========================================================
# ABOUT
# =========================================================

def about():

    st.markdown(
        "## 🛡️ About the Citizen Policy Assistant"
    )


    st.caption(
        "An evidence-grounded assistant designed "
        "to reduce unsupported policy answers."
    )


    st.markdown(
        "### 🎯 Purpose"
    )


    st.write(

        "The Citizen Policy Assistant helps citizens "
        "understand benefits policy in simple language "
        "while showing the evidence used to support "
        "the answer."

    )


    st.markdown(
        "### ⚙️ How It Works"
    )


    cols = st.columns(4)


    steps = [

        (
            "1️⃣ Ask",
            "The citizen asks a policy question.",
        ),

        (
            "2️⃣ Retrieve",
            "Relevant policy evidence is retrieved.",
        ),

        (
            "3️⃣ Ground",
            "The response is checked against "
            "the retrieved evidence.",
        ),

        (
            "4️⃣ Explain",
            "The answer is presented in "
            "citizen-friendly language.",
        ),

    ]


    for col, (title, text) in zip(
        cols,
        steps,
    ):

        with col:

            st.markdown(
                f"### {title}"
            )

            st.write(text)


    st.markdown(
        "### 🛡️ Responsible AI"
    )


    st.success(

        "If sufficient policy evidence is unavailable, "
        "the assistant can abstain instead of guessing."

    )


    st.markdown(
        "### 📖 Citizen Transparency"
    )


    features = [

        "Simple-language answer",

        "Confidence level",

        "Grounded / not-grounded status",

        "Exact supporting policy clause",

        "Policy-date reasoning",

        "Retrieved evidence",

        "Safe next step",

    ]


    for feature in features:

        st.markdown(
            f"✓ {feature}"
        )


# =========================================================
# ROUTING
# =========================================================

if st.session_state.page == "Dashboard":

    dashboard()


elif st.session_state.page == "Ask Policy":

    ask_policy()


elif st.session_state.page == "Policy Insights":

    policy_insights()


else:

    about()


# =========================================================
# FOOTER
# =========================================================

st.divider()


st.caption(

    "🏛️ Citizen Policy Assistant  •  "
    "Evidence-grounded policy support  •  "
    "Designed to avoid unsupported answers"

)