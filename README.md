# The Grounded Answer

Date-aware RAG assistant for the Calder County Household Support Program.

The system helps a county benefits office answer policy questions in plain language while grounding answers in the available policy documents.

## Run

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt

streamlit run app.py

The first run downloads all-MiniLM-L6-v2 through sentence-transformers.

Current Corpus

The data/ folder contains:

policy-manual.md — Original consolidated Calder County Household Support Program policy manual.
amendment_2026_01.md — Amendment No. 2026-01, effective 1 March 2026.

The amendment modifies the consolidated policy manual; it does not replace it.

Design

The system uses a hybrid retrieval and policy-rule architecture.

Sentence-transformer embeddings for semantic retrieval.
BM25-style lexical retrieval.
72/28 hybrid ranking.
Date-aware amendment handling.
Exact policy clause references.
Conservative "I don't know" fallback.
No paid services required for the core demo.
Policy Handling

The assistant separates:

Policy document retrieval.
Effective-date reasoning.
Deterministic policy rules introduced by the amendment.

This allows the Day-2 amendment to be applied without replacing the original policy corpus or rewriting the retrieval layer.

The amendment effective date is 1 March 2026.

The system handles policy changes involving:

Earnings disregard.
Income thresholds.
Reporting periods.
Sanction rates.
Sanction exceptions.
Transitional provisions.
Claim periods spanning the amendment date.
Grounding

The assistant does not intentionally guess.

A response is presented as grounded only when retrieved policy evidence supports the relevant rule.

The interface shows:

Plain-language answer.
Confidence.
Grounded / not-grounded status.
Supporting policy clause.
Retrieved evidence.
Policy-date reasoning.
Suggested next step.

When the available evidence is insufficient, the assistant responds with an "I don't know" message and directs the user to the appropriate county benefits policy team.

Citizen Language

The assistant accepts common citizen-style wording and normalizes it into policy-oriented terminology before retrieval.

Examples include:

"Can I qualify?" → eligibility
"How much can I get?" → maximum benefit amount
"How much can I earn?" → income threshold
"How long do I have?" → reporting period
"Will I be punished?" → sanction
Supported Policy Areas

The current rule-based policy layer supports common questions about:

Eligibility.
Maximum benefit amounts.
Income thresholds.
Earnings disregard.
Reporting changes of circumstances.
Sanctions.
Cross-period policy rules.
Exact Evidence

Grounded answers include the relevant policy section and source.

The application also displays the retrieved policy text so that the user can inspect the evidence used to support the answer.

Conservative Fallback

When sufficient policy evidence is not available, the assistant does not invent an answer.

Instead, it explains that the available policy material is insufficient and recommends contacting the county benefits policy team.

This behavior is intentional and is part of the responsible design of the system.

Project Structure
Cloud_Bride/
├── app.py
├── data/
│   ├── policy-manual.md
│   └── amendment_2026_01.md
├── rag/
│   ├── engine.py
│   ├── chunker.py
│   └── retriever.py
├── assests/
│   └── style.css
├── tests/
├── requirements.txt
├── README.md
├── DECISIONS.md
└── AI-USAGE.md
Known Limitations
No authentication.
No persistent user history.
No external deployment.
No paid API dependency.
The assistant can only answer questions supported by the available policy corpus.
The system does not replace an official benefits determination.
Unusual or unsupported questions may result in an "I don't know" response.
Clean-Environment Check

Before final submission, create a fresh virtual environment and run the exact commands in this README.

The application should start without undocumented setup steps.

The required policy documents must be present in the data/ folder:

policy-manual.md
amendment_2026_01.md
Submission

The repository should contain:

A running application.
The complete policy corpus.
Amendment No. 2026-01.
DECISIONS.md.
AI-USAGE.md.
requirements.txt.
This README.

The repository should also contain real commit history showing development progress.

Core Principle

Retrieve the relevant policy evidence, apply the rule for the relevant date, cite the supporting clause, and say "I don't know" when the available evidence is insufficient.