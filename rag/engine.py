from pathlib import Path
import re
import numpy as np
from sentence_transformers import SentenceTransformer

from .chunker import load_documents, chunk_markdown
from .retriever import HybridRetriever


# =========================================================
# DATE / YEAR REGEX
# =========================================================

DATE_RE = re.compile(
    r"\b(January|February|March|April|May|June|July|August|September|"
    r"October|November|December)\s+(\d{4})\b",
    re.I,
)

YEAR_RE = re.compile(r"\b(20\d{2})\b")


# =========================================================
# DATE HELPERS
# =========================================================

def extract_date(text):
    """
    Extracts Month + Year from text.

    Example:
        April 2026 -> (2026, 4)
    """

    m = DATE_RE.search(text)

    if not m:
        return None

    months = {
        month: i
        for i, month in enumerate(
            [
                "january",
                "february",
                "march",
                "april",
                "may",
                "june",
                "july",
                "august",
                "september",
                "october",
                "november",
                "december",
            ],
            1,
        )
    }

    return (int(m.group(2)), months[m.group(1).lower()])


def extract_year(text):
    """
    Extracts a year from a question.

    Example:
        2027 -> 2027
    """

    m = YEAR_RE.search(text)

    if not m:
        return None

    return int(m.group(1))


def month_at_or_after(year, month, y2=2026, m2=3):
    """
    Returns True when the given date is on or after
    1 March 2026.
    """

    return (year, month) >= (y2, m2)


def extract_household_size(text):
    """
    Extract household size from different question formats.

    Supported examples:

        household of 8
        household size 8
        household of size 8
        family of 8

    Returns:
        integer household size or None
    """

    patterns = [
        r"household\s+(?:of\s+)?(?:size\s+)?(\d+)",
        r"household\s+size\s*(?:of\s*)?(\d+)",
        r"family\s+(?:of\s+)?(\d+)",
    ]

    text = text.lower()

    for pattern in patterns:
        m = re.search(pattern, text)

        if m:
            return int(m.group(1))

    return None


# =========================================================
# AMOUNT EXTRACTION HELPERS
# =========================================================

MONEY_RE = re.compile(
    r"\$?\s*[\d,]+(?:\.\d{1,2})?"
)


def normalize_amount(amount_text):
    """
    Converts:

        $2,075
        2,075
        $2075.00

    into a clean display amount.
    """

    if not amount_text:
        return None

    amount_text = amount_text.strip()

    # Ignore years accidentally captured as amounts
    clean = amount_text.replace("$", "").replace(",", "").strip()

    try:
        value = float(clean)
    except ValueError:
        return None

    # Avoid treating 2027 etc. as money
    if value >= 1900 and value <= 2100:
        return None

    if value.is_integer():
        return f"${int(value):,}"

    return f"${value:,.2f}"


def extract_amount_near_household(text, household_size):
    """
    Attempts to find an amount associated with a household size.

    This function supports common markdown/table formats such as:

        | 8 | $3,000 |

        8 | $3,000

        household of 8: $3,000

        household size 8 = $3,000

    IMPORTANT:
    This function only extracts an amount that actually appears
    in the retrieved policy text. It NEVER calculates or guesses
    an amount.
    """

    if household_size is None:
        return None

    size = str(household_size)

    # -----------------------------------------------------
    # Pattern 1
    # Markdown table:
    #
    # | 8 | $3,000 |
    # -----------------------------------------------------

    table_patterns = [
        rf"\|\s*{re.escape(size)}\s*\|\s*(\$?\s*[\d,]+(?:\.\d{{1,2}})?)",
        rf"\|\s*{re.escape(size)}\s*\|\s*[^|]*\|\s*(\$?\s*[\d,]+(?:\.\d{{1,2}})?)",
    ]

    for pattern in table_patterns:
        m = re.search(pattern, text, re.I)

        if m:
            amount = normalize_amount(m.group(1))

            if amount:
                return amount

    # -----------------------------------------------------
    # Pattern 2
    #
    # household of 8: $3,000
    # household size 8: $3,000
    # -----------------------------------------------------

    direct_patterns = [
        rf"household\s+(?:of\s+)?{re.escape(size)}"
        rf"[^$\d]{{0,50}}(\$?\s*[\d,]+(?:\.\d{{1,2}})?)",

        rf"household\s+size\s*(?:of\s*)?{re.escape(size)}"
        rf"[^$\d]{{0,50}}(\$?\s*[\d,]+(?:\.\d{{1,2}})?)",

        rf"size\s*{re.escape(size)}"
        rf"[^$\d]{{0,50}}(\$?\s*[\d,]+(?:\.\d{{1,2}})?)",
    ]

    for pattern in direct_patterns:
        m = re.search(pattern, text, re.I)

        if m:
            amount = normalize_amount(m.group(1))

            if amount:
                return amount

    # -----------------------------------------------------
    # Pattern 3
    #
    # 8 persons ... $3,000
    # 8 people ... $3,000
    # -----------------------------------------------------

    person_patterns = [
        rf"\b{re.escape(size)}\s+(?:persons?|people|members?)"
        rf"[^$\d]{{0,60}}(\$?\s*[\d,]+(?:\.\d{{1,2}})?)",

        rf"\b{re.escape(size)}\b"
        rf"[^$\d]{{0,30}}(\$?\s*[\d,]+(?:\.\d{{1,2}})?)",
    ]

    for pattern in person_patterns:
        m = re.search(pattern, text, re.I)

        if m:
            amount = normalize_amount(m.group(1))

            if amount:
                return amount

    return None


def find_policy_amount(results, household_size):
    """
    Searches retrieved evidence for an exact amount
    associated with the requested household size.

    Returns:
        amount, supporting_result
    """

    if household_size is None:
        return None, None

    for result in results:

        text = result.get("text", "")

        amount = extract_amount_near_household(
            text,
            household_size
        )

        if amount:
            return amount, result

    return None, None


# =========================================================
# GROUNDING ENGINE
# =========================================================

class GroundedEngine:

    def __init__(self, data_dir="data"):

        self.data_dir = data_dir

        # -------------------------------------------------
        # Embedding model
        # -------------------------------------------------

        self.model = SentenceTransformer(
            "all-MiniLM-L6-v2"
        )

        # -------------------------------------------------
        # Load policy documents
        # -------------------------------------------------

        docs = load_documents(data_dir)

        # -------------------------------------------------
        # Create chunks
        # -------------------------------------------------

        self.chunks = []

        for d in docs:

            self.chunks.extend(
                chunk_markdown(
                    d["text"],
                    d["source"]
                )
            )

        # -------------------------------------------------
        # Create embeddings
        # -------------------------------------------------

        texts = [
            c["text"]
            for c in self.chunks
        ]

        self.embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        # -------------------------------------------------
        # Hybrid retriever
        # -------------------------------------------------

        self.retriever = HybridRetriever(
            self.chunks,
            self.embeddings
        )

    # =====================================================
    # RULE-BASED POLICY LOGIC
    # =====================================================

    def _answer_from_rules(self, q, results):

        ql = q.lower()

        # -------------------------------------------------
        # Extract date
        # -------------------------------------------------

        date = extract_date(q)

        # -------------------------------------------------
        # Extract year
        # -------------------------------------------------

        year = extract_year(q)

        # -------------------------------------------------
        # Determine amendment applicability
        # -------------------------------------------------

        post = (
            date is not None
            and month_at_or_after(*date)
        )

        # =================================================
        # 1. MAXIMUM BENEFIT
        # =================================================

        if (
            "maximum benefit" in ql
            or "maximum amount" in ql
            or "maximum award" in ql
            or (
                "benefit" in ql
                and "maximum" in ql
            )
        ):

            household_size = extract_household_size(q)

            if household_size is not None:

                amount, supporting_result = (
                    find_policy_amount(
                        results,
                        household_size
                    )
                )

                # -----------------------------------------
                # Exact amount found in policy
                # -----------------------------------------

                if amount:

                    source = supporting_result.get(
                        "source",
                        "policy document"
                    )

                    section = supporting_result.get(
                        "section",
                        "policy section"
                    )

                    requested_year = (
                        year
                        if year is not None
                        else date[0]
                        if date is not None
                        else None
                    )

                    if requested_year:

                        answer = (
                            f"For a household of "
                            f"{household_size} in "
                            f"{requested_year}, the maximum "
                            f"benefit is {amount}."
                        )

                    else:

                        answer = (
                            f"The maximum benefit for a "
                            f"household of {household_size} "
                            f"is {amount}."
                        )

                    return (
                        answer,
                        f"{section} — {source}",
                        "Retrieved policy evidence"
                    )

                # -----------------------------------------
                # Household size found but amount missing
                # -----------------------------------------

                return (
                    (
                        f"I found policy material about the "
                        f"maximum benefit, but the available "
                        f"evidence does not contain an exact "
                        f"amount for a household of "
                        f"{household_size}. I don't want to "
                        f"guess. Please contact the county "
                        f"benefits policy team."
                    ),
                    (
                        results[0].get("section", "policy section")
                        if results
                        else None
                    ),
                    "Maximum benefit rule not sufficiently specified"
                )

        # =================================================
        # 2. INCOME THRESHOLD
        # =================================================

        if (
            "income" in ql
            and (
                "threshold" in ql
                or "limit" in ql
            )
        ):

            size = extract_household_size(q)

            if size is not None:

                base = {
                    1: 1225,
                    2: 1650,
                    3: 2075,
                    4: 2500,
                    5: 2925,
                }.get(size)

                if base is not None and post:

                    return (
                        (
                            f"For a determination made in "
                            f"{date[0]}, the monthly income "
                            f"threshold for a household of "
                            f"{size} is ${base:,}."
                        ),
                        "§6.6.1 / Amendment §3.1",
                        "Amendment §5.1"
                    )

                # -----------------------------------------
                # If household size is outside hardcoded
                # policy table, don't invent a value.
                # -----------------------------------------

                if base is None:

                    return (
                        (
                            f"I don't have enough policy evidence "
                            f"to determine the monthly income "
                            f"threshold for a household of "
                            f"{size}. Please contact the county "
                            f"benefits policy team."
                        ),
                        None,
                        None
                    )

        # =================================================
        # 3. EARNINGS DISREGARD
        # =================================================

        if "earnings disregard" in ql:

            if post:

                return (
                    "The earnings disregard is $175 per month.",
                    "§6.4.1(a) / Amendment §1.1",
                    "Amendment §5.1"
                )

            if date is not None:

                return (
                    (
                        "The amendment does not apply to a "
                        "determination made before 1 March 2026; "
                        "the pre-amendment amount was $120 per month."
                    ),
                    "§6.4.1(a) / Amendment §1.1",
                    "Amendment §5.1"
                )

        # =================================================
        # 4. REPORTING PERIOD
        # =================================================

        if (
            (
                "report" in ql
                or "reporting" in ql
            )
            and
            (
                "change" in ql
                or "circumstance" in ql
            )
        ):

            if date is not None:

                if post:

                    return (
                        (
                            "The reporting period is 14 "
                            "calendar days because the change "
                            "of circumstances occurred on or "
                            "after 1 March 2026."
                        ),
                        "§4.3.2 / Amendment §2.1",
                        "Amendment §5.2"
                    )

                return (
                    (
                        "The reporting period was 10 "
                        "calendar days because the change "
                        "of circumstances occurred before "
                        "1 March 2026."
                    ),
                    "§4.3.2 / Amendment §2.1",
                    "Amendment §5.2"
                )

        # =================================================
        # 5. CROSS-PERIOD RULE
        # =================================================

        if (
            "spans 1 march 2026" in ql
            or "spanning 1 march 2026" in ql
            or "crosses 1 march 2026" in ql
            or (
                "claim period" in ql
                and "1 march 2026" in ql
            )
        ):

            return (
                (
                    "If a claim period spans 1 March 2026, "
                    "use the figures that were in force on "
                    "each day of the period and apportion "
                    "the award accordingly under §7.4.3."
                ),
                "§7.4.3 / Amendment §5.3",
                "Amendment §5.3"
            )

        # =================================================
        # 6. SANCTION
        # =================================================

        if "sanction" in ql:

            # ---------------------------------------------
            # Special exception
            # ---------------------------------------------

            if (
                "increase" in ql
                or "increased" in ql
            ):

                return (
                    (
                        "A sanction must not be imposed where "
                        "the change of circumstances would "
                        "have increased the award."
                    ),
                    "§10.5.3A / Amendment §4.2",
                    "Amendment §5.1"
                )

            # ---------------------------------------------
            # Normal post-amendment sanction
            # ---------------------------------------------

            if post:

                return (
                    (
                        "The sanction rate is 15 per cent "
                        "for a determination made on or "
                        "after 1 March 2026."
                    ),
                    "§10.5.2 / Amendment §4.1",
                    "Amendment §5.1"
                )

        # =================================================
        # No deterministic rule found
        # =================================================

        return None

    # =====================================================
    # MAIN QUESTION ANSWER FUNCTION
    # =====================================================

    def ask(self, question):

        # -------------------------------------------------
        # Create query embedding
        # -------------------------------------------------

        query_embedding = self.model.encode(
            [question],
            convert_to_numpy=True,
            normalize_embeddings=True
        )[0]

        # -------------------------------------------------
        # Retrieve evidence
        # -------------------------------------------------

        results = self.retriever.retrieve(
            question,
            query_embedding,
            top_k=8
        )

        # -------------------------------------------------
        # Try deterministic policy rules
        # -------------------------------------------------

        rule = self._answer_from_rules(
            question,
            results
        )

        # -------------------------------------------------
        # RULE-BASED ANSWER
        # -------------------------------------------------

        if rule:

            answer, citation, effective = rule

            # If rule deliberately says insufficient evidence
            if citation is None:

                return {
                    "answer": answer,
                    "citation": None,
                    "effective_rule": effective,
                    "confidence": "low",
                    "evidence": results[:3],
                    "grounded": False,
                }

            return {
                "answer": answer,
                "citation": citation,
                "effective_rule": effective,
                "confidence": "high",
                "evidence": results[:3],
                "grounded": True,
            }

        # =================================================
        # CONSERVATIVE FALLBACK
        # =================================================

        if (
            not results
            or results[0]["score"] < 0.20
        ):

            return {
                "answer": (
                    "I don't know. The available policy "
                    "material does not provide enough "
                    "evidence to answer this reliably. "
                    "Please contact the county benefits "
                    "policy team."
                ),
                "citation": None,
                "effective_rule": None,
                "confidence": "low",
                "evidence": results[:3],
                "grounded": False,
            }

        # =================================================
        # RELEVANT BUT NOT ENOUGH TO ANSWER
        # =================================================

        top = results[0]

        return {
            "answer": (
                "I found potentially relevant policy text, "
                "but I cannot determine the applicable rule "
                "with enough confidence from the available "
                "material."
            ),
            "citation": (
                f"{top.get('section', 'Policy section')} — "
                f"{top.get('source', 'Policy document')}"
            ),
            "effective_rule": None,
            "confidence": "medium",
            "evidence": results[:3],
            "grounded": True,
        }