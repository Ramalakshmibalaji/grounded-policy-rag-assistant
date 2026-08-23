import streamlit as st
from pathlib import Path
import pandas as pd
from rag.engine import GroundedEngine

st.set_page_config(
    page_title="Citizen Policy Assistant",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
)


def load_css():
    css_path = Path(__file__).parent / "assests" / "style.css"
    if css_path.exists():
        with open(css_path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


load_css()


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
for key, default in {
    "page": "Dashboard",
    "messages": [],
    "last_result": None,
    "guided_question": None,
    "answer_count": 0,
    "grounded_count": 0,
    "abstained_count": 0,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


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
                return len(value) if name in {"sections", "policy_sections", "chunks", "documents"} else value
            except TypeError:
                return value
    return default


# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.markdown("## 🏛️ Citizen Policy Assistant")
    st.caption("A grounded AI assistant that answers from available policy evidence.")
    st.divider()

    st.subheader("🧭 Navigation")
    pages = [
        ("📊 Dashboard", "Dashboard"),
        ("💬 Ask Policy", "Ask Policy"),
        ("📈 Policy Insights", "Policy Insights"),
        ("🛡️ About", "About"),
    ]
    for label, page in pages:
        if st.button(label, use_container_width=True, type="primary" if st.session_state.page == page else "secondary"):
            navigate(page)

    st.divider()
    st.subheader("🛡️ Grounding")
    st.write("The assistant does not intentionally guess when sufficient policy evidence is unavailable.")
    st.divider()
    if st.button("🗑️ Clear conversation", use_container_width=True):
        clear_conversation()
        st.rerun()


# =========================================================
# TOP HEADER / NAVIGATION
# =========================================================
st.markdown("# 🏛️ Citizen Policy Assistant")
st.markdown("**Evidence-grounded policy support for citizens**")

nav = st.columns(4)
for col, label, page in zip(
    nav,
    ["📊 Dashboard", "💬 Ask Policy", "📈 Policy Insights", "🛡️ About"],
    ["Dashboard", "Ask Policy", "Policy Insights", "About"],
):
    with col:
        if st.button(label, use_container_width=True):
            navigate(page)

st.divider()


# =========================================================
# DASHBOARD
# =========================================================
def dashboard():
    st.header("📊 Policy Dashboard")
    st.write("A quick overview of the available benefits policy rules and the assistant's grounding approach.")

    policy_sections = engine_value(["sections", "policy_sections", "chunks", "documents"])
    policy_change = engine_value(["policy_change_date", "effective_date", "amendment_date"], "Available in policy evidence")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("📚 Policy Sections", policy_sections)
    c2.metric("🎯 Answers Generated", st.session_state.answer_count)
    c3.metric("🛡️ Grounded Answers", st.session_state.grounded_count)
    c4.metric("⚠️ Abstained", st.session_state.abstained_count)

    st.markdown("### 📅 Policy Status")
    st.info(f"Current policy change / effective-date information: **{policy_change}**")

    left, right = st.columns([1.45, 1])

    with left:
        st.subheader("📈 Grounding Overview")
        if st.session_state.answer_count:
            df = pd.DataFrame({
                "Status": ["Grounded", "I don't know"],
                "Answers": [st.session_state.grounded_count, st.session_state.abstained_count],
            }).set_index("Status")
            st.bar_chart(df)
            st.caption("Based on actual answers generated during this session.")
        else:
            st.info("Ask a policy question to populate this chart with real results.")

    with right:
        st.subheader("🛡️ Grounded AI Status")
        result = st.session_state.last_result
        if result is None:
            st.success("Ready — ask a policy question to see grounding status.")
            st.write("The assistant retrieves policy evidence before producing an answer.")
        elif result.get("grounded", False):
            st.success("✅ Latest answer is supported by policy evidence.")
            st.write(f"Confidence: **{result.get('confidence', 'low')}**")
        else:
            st.warning("⚠️ Latest question did not have enough supporting evidence.")
            st.write(f"Confidence: **{result.get('confidence', 'low')}**")

    st.subheader("🚀 Quick Actions")
    q1, q2, q3 = st.columns(3)
    with q1:
        if st.button("💬 Ask a Policy Question", use_container_width=True):
            navigate("Ask Policy")
    with q2:
        if st.button("📈 View Policy Insights", use_container_width=True):
            navigate("Policy Insights")
    with q3:
        if st.button("🛡️ How It Works", use_container_width=True):
            navigate("About")


# =========================================================
# ASK POLICY
# =========================================================
def ask_policy():
    st.header("💬 Ask Policy")
    st.write("Ask your benefits policy question in simple language. The assistant answers only from retrieved policy evidence.")

    st.markdown("### 👋 How can I help you?")
    st.caption("Choose a common topic or type your own question.")

    options = {
        "🟢 Check my eligibility": "eligibility",
        "💰 How much can I get?": "amount",
        "📅 Report a change": "report_change",
        "⚠️ Understand a sanction": "sanction",
        "📖 Understand a policy rule": "policy",
        "❓ I don't know what to ask": "help",
    }
    selected = st.selectbox("Choose what you need help with", ["Select an option"] + list(options))
    guided_question = None

    if selected != "Select an option":
        mode = options[selected]

        if mode == "eligibility":
            st.markdown("#### 🟢 Tell us about your situation")
            size = st.number_input("How many people are in your household?", 1, 20, 1, key="elig_size")
            income = st.number_input("Approximate monthly household income", 0, value=0, step=50, key="elig_income")
            dt = st.date_input("Determination date", key="elig_date")
            guided_question = f"What are the eligibility requirements for a household of {size} with a monthly income of ${income:,} for a determination made in {dt.strftime('%B %Y')}?"

        elif mode == "amount":
            st.markdown("#### 💰 Tell us about your household")
            size = st.number_input("Household size", 1, 20, 3, key="amount_size")
            dt = st.date_input("Determination date", key="amount_date")
            guided_question = f"What is the maximum benefit for a household of {size} for a determination made in {dt.strftime('%B %Y')}?"

        elif mode == "report_change":
            st.markdown("#### 📅 What kind of change happened?")
            change = st.radio("Choose one", [
                "💼 I started a job", "💰 My income changed", "🏠 I moved house",
                "👨‍👩‍👧 Someone joined or left my household", "❓ Something else"
            ], key="change_type")
            dt = st.date_input("When did the change happen?", key="change_date")
            desc = {
                "💼 I started a job": "I started a job",
                "💰 My income changed": "My income changed",
                "🏠 I moved house": "I moved house",
                "👨‍👩‍👧 Someone joined or left my household": "Someone joined or left my household",
                "❓ Something else": "Something else changed in my circumstances",
            }[change]
            guided_question = f"{desc}. The change occurred in {dt.strftime('%B %Y')}. How long do I have to report this change of circumstances?"

        elif mode == "sanction":
            st.markdown("#### ⚠️ Tell us about the sanction situation")
            dt = st.date_input("When did the change of circumstances occur?", key="sanction_date")
            increase = st.radio("Would the change have increased the benefit award?", ["Yes", "No", "I don't know"], key="sanction_increase")
            if increase == "Yes":
                guided_question = "What sanction applies when the unreported change would have increased the award?"
            elif increase == "No":
                guided_question = f"What sanction applies to a change of circumstances occurring in {dt.strftime('%B %Y')}?"
            else:
                guided_question = f"What sanction rules apply to a change of circumstances occurring in {dt.strftime('%B %Y')}?"

        elif mode == "policy":
            st.markdown("#### 📖 What policy rule do you want to understand?")
            topic = st.selectbox("Choose a topic", [
                "Income threshold", "Earnings disregard", "Maximum benefit", "Reporting period", "Sanction", "Other policy rule"
            ], key="policy_topic")
            if topic == "Income threshold":
                size = st.number_input("Household size", 1, 20, 3, key="policy_size")
                dt = st.date_input("Determination date", key="policy_income_date")
                guided_question = f"What is the income threshold for a household of {size} for a determination made in {dt.strftime('%B %Y')}?"
            elif topic == "Earnings disregard":
                dt = st.date_input("Determination date", key="policy_earnings_date")
                guided_question = f"What is the earnings disregard for a determination made in {dt.strftime('%B %Y')}?"
            elif topic == "Maximum benefit":
                size = st.number_input("Household size", 1, 20, 3, key="policy_benefit_size")
                dt = st.date_input("Determination date", key="policy_benefit_date")
                guided_question = f"What is the maximum benefit for a household of {size} for a determination made in {dt.strftime('%B %Y')}?"
            elif topic == "Reporting period":
                dt = st.date_input("When did the change occur?", key="policy_report_date")
                guided_question = f"A change of circumstances occurred in {dt.strftime('%B %Y')}. How long do I have to report it?"
            elif topic == "Sanction":
                dt = st.date_input("When did the change occur?", key="policy_sanction_date")
                guided_question = f"What sanction rules apply to a change of circumstances occurring in {dt.strftime('%B %Y')}?"
            else:
                st.info("Type your own policy question below.")

        else:
            st.info("You can ask about eligibility, benefit amounts, income thresholds, earnings disregard, reporting deadlines, sanctions, and policy rules.")

    if guided_question:
        st.markdown("#### 📝 Your question")
        st.info(guided_question)
        if st.button("🔎 Get answer", type="primary", use_container_width=True):
            st.session_state.guided_question = guided_question
            st.rerun()

    st.markdown("### 💬 Example questions")
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
            if st.button(q, key=f"quick_{i}", use_container_width=True):
                selected_question = q

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    question = st.chat_input("Type your policy question here...")
    if st.session_state.guided_question:
        question = st.session_state.guided_question
        st.session_state.guided_question = None
    elif selected_question:
        question = selected_question

    if question and question.strip():
        process_question(question.strip())


def process_question(question):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("🔎 Checking the policy evidence..."):
            try:
                result = engine.ask(question)
            except Exception as e:
                st.error(f"Error while processing the question: {e}")
                return

        st.session_state.last_result = result
        st.session_state.answer_count += 1
        if result.get("grounded", False):
            st.session_state.grounded_count += 1
        else:
            st.session_state.abstained_count += 1

        st.markdown("### 💬 Answer")
        st.markdown('<div class="answer-card">', unsafe_allow_html=True)
        st.write(result.get("answer", "I don't know."))
        st.markdown("</div>", unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        c1.metric("🎯 Confidence", result.get("confidence", "low"))
        c2.metric("📚 Evidence status", "✅ Grounded" if result.get("grounded", False) else "⚠️ Not grounded")

        if result.get("grounded", False):
            st.success("✅ Grounded answer — this response is supported by retrieved policy evidence.")
        else:
            st.warning("⚠️ I don't know — the available policy evidence does not provide enough support for a reliable answer.")

        citation = result.get("citation")
        if citation:
            st.markdown("### 📖 Official supporting clause")
            with st.expander("View the exact policy clause", expanded=True):
                st.code(citation, language="text")

        effective_rule = result.get("effective_rule")
        if effective_rule:
            st.markdown("### 📅 Why this rule applies")
            with st.expander("View policy-date reasoning", expanded=True):
                st.write(effective_rule)

        evidence = result.get("evidence", [])
        if evidence:
            st.markdown("### 🔎 Retrieved evidence")
            for i, item in enumerate(evidence, 1):
                score = item.get("score", 0)
                section = item.get("section", "Policy evidence")
                with st.expander(f"Evidence {i} • {section} • relevance {score:.3f}"):
                    st.write(item.get("text", ""))

        st.markdown("### 👉 What you can do next")
        next_step = result.get("next_step")
        if result.get("grounded", False):
            st.info(next_step or "Review the supporting policy clause and contact the benefits office if needed.")
        else:
            st.warning(next_step or "Contact the appropriate county benefits policy office for further guidance.")

    st.session_state.messages.append({
        "role": "assistant",
        "content": result.get("answer", "I don't know."),
    })


# =========================================================
# POLICY INSIGHTS
# =========================================================
def policy_insights():
    st.header("📈 Policy Insights")
    st.write("Simple comparisons from the policy assistant's current session.")

    st.subheader("⚖️ Grounded vs Abstained")
    df = pd.DataFrame({
        "Status": ["Grounded", "I don't know"],
        "Answers": [st.session_state.grounded_count, st.session_state.abstained_count],
    }).set_index("Status")
    st.bar_chart(df)

    result = st.session_state.last_result
    if result and result.get("evidence"):
        st.subheader("🔎 Latest Evidence Relevance")
        evidence = result["evidence"]
        relevance = pd.DataFrame({
            "Evidence": [f"Evidence {i}" for i in range(1, len(evidence) + 1)],
            "Relevance": [item.get("score", 0) for item in evidence],
        }).set_index("Evidence")
        st.bar_chart(relevance)
        st.caption("These are the actual retrieval relevance scores returned by GroundedEngine.")

    st.subheader("📌 Latest Result")
    if not result:
        st.info("No policy question has been answered in this session yet.")
    else:
        c1, c2 = st.columns(2)
        c1.metric("Confidence", result.get("confidence", "low"))
        c2.metric("Evidence status", "Grounded" if result.get("grounded", False) else "Not grounded")
        st.write("**Answer:**", result.get("answer", "I don't know."))

    st.info("Confidence alone is not treated as proof. A response is grounded only when retrieved policy evidence supports it.")


# =========================================================
# ABOUT
# =========================================================
def about():
    st.header("🛡️ About the Citizen Policy Assistant")
    st.subheader("🎯 Purpose")
    st.write("The Citizen Policy Assistant is an evidence-grounded AI system designed to help citizens understand benefits policy in simple language.")

    st.subheader("⚙️ How It Works")
    cols = st.columns(4)
    steps = [
        ("1️⃣ Ask", "The citizen asks a policy question."),
        ("2️⃣ Retrieve", "Relevant policy evidence is retrieved."),
        ("3️⃣ Ground", "The response is checked against policy evidence."),
        ("4️⃣ Explain", "The result is presented in citizen-friendly language."),
    ]
    for col, (title, text) in zip(cols, steps):
        with col:
            st.markdown(f"### {title}")
            st.write(text)

    st.subheader("🛡️ Responsible AI Approach")
    st.success("When the available policy evidence is insufficient, the assistant can abstain instead of guessing.")

    st.subheader("📖 What the citizen can see")
    st.write("• Simple-language answer\n\n• Confidence level\n\n• Grounded / not-grounded status\n\n• Exact supporting policy clause\n\n• Policy-date reasoning\n\n• Retrieved evidence\n\n• Safe next step")


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

st.divider()
st.caption("🏛️ Citizen Policy Assistant • Evidence-grounded policy support • Designed to avoid unsupported answers")