import streamlit as st
from rag.engine import GroundedEngine

st.set_page_config(page_title="The Grounded Answer", page_icon="📘", layout="wide")

@st.cache_resource
def get_engine():
    return GroundedEngine("data")

st.title("📘 The Grounded Answer")
st.caption("Date-aware, evidence-grounded policy assistant")

st.info(
    "Ask a benefits-policy question. The assistant retrieves policy evidence, "
    "applies the amendment date rules, cites the supporting clause, and refuses to guess."
)

try:
    engine = get_engine()
except Exception as e:
    st.error(f"Startup error: {e}")
    st.stop()

examples = [
    "What is the earnings disregard for a determination made in April 2026?",
    "A change of circumstances occurred in February 2026. How long do I have to report it?",
    "What is the income threshold for a household of 3 for a determination made in April 2026?",
    "What sanction applies when the unreported change would have increased the award?",
]

question = st.text_area(
    "Your question",
    placeholder="Example: What is the earnings disregard for a determination made in April 2026?",
    height=110,
)

if st.button("Get grounded answer", type="primary", use_container_width=True):
    if not question.strip():
        st.warning("Enter a question first.")
    else:
        with st.spinner("Retrieving policy evidence..."):
            result = engine.ask(question)

        st.subheader("Answer")
        st.write(result["answer"])

        c1, c2 = st.columns(2)
        with c1:
            st.metric("Confidence", result["confidence"])
        with c2:
            st.metric("Grounded", "Yes" if result["grounded"] else "No")

        if result["citation"]:
            st.subheader("Exact supporting clause")
            st.code(result["citation"])

        if result["effective_rule"]:
            st.subheader("Why this rule applies")
            st.write(result["effective_rule"])

        st.subheader("Retrieved evidence")
        for i, item in enumerate(result["evidence"], 1):
            with st.expander(f"{i}. {item['section']} — score {item['score']:.3f}"):
                st.write(item["text"])

st.divider()
st.subheader("Demo questions")
for q in examples:
    st.write("• " + q)
