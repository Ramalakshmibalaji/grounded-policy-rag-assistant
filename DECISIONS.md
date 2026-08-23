# Decisions

## Initial stack
- Reused the proven RAG concepts from the previous semantic-search project: sentence-transformer embeddings, hybrid dense + lexical retrieval, and grounded evidence display.
- Chose Streamlit for the fastest demonstrable interface because Problem 1 does not score interface quality directly.

## Day-2 amendment
- Kept policy retrieval separate from effective-date reasoning.
- Added a policy-rule layer so amendments can be applied without rewriting retrieval.
- Did not replace the policy corpus with the amendment; the amendment is treated as an overlay.

## Cut for time
- No authentication.
- No persistent user history.
- No external deployment.
- No paid API dependency in the core path.

## Original policy corpus

- Added the original consolidated policy manual to `data/policy-manual.md`.
- Kept `amendment_2026_01.md` as a separate document because the amendment modifies the consolidated manual rather than replacing it.
- Kept policy retrieval separate from effective-date reasoning so that the Day-2 change could be incorporated without rewriting the retrieval layer.
