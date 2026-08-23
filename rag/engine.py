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
    Extract Month + Year.

    Example:
        April 2026 -> (2026, 4)
    """

    m = DATE_RE.search(text)

    if not m:
        return None

    months = {
        "january": 1,
        "february": 2,
        "march": 3,
        "april": 4,
        "may": 5,
        "june": 6,
        "july": 7,
        "august": 8,
        "september": 9,
        "october": 10,
        "november": 11,
        "december": 12,
    }

    return (
        int(m.group(2)),
        months[m.group(1).lower()],
    )


def extract_year(text):
    """
    Extract a year.

    Example:
        2027 -> 2027
    """

    m = YEAR_RE.search(text)

    if not m:
        return None

    return int(m.group(1))


def month_at_or_after(year, month, y2=2026, m2=3):
    """
    Checks whether a date is on or after
    1 March 2026.
    """

    return (year, month) >= (y2, m2)


# =========================================================
# HOUSEHOLD SIZE
# =========================================================

def extract_household_size(text):
    """
    Extract household/family size.

    Supported examples:

        household of 3
        household size 3
        family of 3
        3 people
        3 persons
        family has 3 members
    """

    text = text.lower()

    patterns = [
        r"household\s+(?:of\s+)?(?:size\s+)?(\d+)",
        r"household\s+size\s*(?:of\s*)?(\d+)",
        r"family\s+(?:of\s+)?(\d+)",
        r"family\s+(?:has|with)\s+(\d+)",
        r"\b(\d+)\s+(?:people|persons|members)\b",
    ]

    for pattern in patterns:

        match = re.search(pattern, text)

        if match:
            return int(match.group(1))

    return None


# =========================================================
# MONEY
# =========================================================

MONEY_RE = re.compile(
    r"\$?\s*[\d,]+(?:\.\d{1,2})?"
)


def normalize_amount(amount_text):
    """
    Convert policy amount to clean display format.
    """

    if not amount_text:
        return None

    clean = (
        amount_text
        .replace("$", "")
        .replace(",", "")
        .strip()
    )

    try:
        value = float(clean)
    except ValueError:
        return None

    # Do not accidentally treat a year as money.
    if 1900 <= value <= 2100:
        return None

    if value.is_integer():
        return f"${int(value):,}"

    return f"${value:,.2f}"


def extract_amount_near_household(text, household_size):
    """
    Extract an exact amount that appears near
    the requested household size.

    IMPORTANT:
    This function never calculates or invents
    a policy amount.
    """

    if household_size is None:
        return None

    size = str(household_size)

    # -----------------------------------------------------
    # Markdown table
    # -----------------------------------------------------

    table_patterns = [
        rf"\|\s*{re.escape(size)}\s*\|\s*"
        rf"(\$?\s*[\d,]+(?:\.\d{{1,2}})?)",

        rf"\|\s*{re.escape(size)}\s*\|\s*"
        rf"[^|]*\|\s*"
        rf"(\$?\s*[\d,]+(?:\.\d{{1,2}})?)",
    ]

    for pattern in table_patterns:

        match = re.search(
            pattern,
            text,
            re.I,
        )

        if match:

            amount = normalize_amount(
                match.group(1)
            )

            if amount:
                return amount

    # -----------------------------------------------------
    # Direct household wording
    # -----------------------------------------------------

    direct_patterns = [

        rf"household\s+(?:of\s+)?{re.escape(size)}"
        rf"[^$\d]{{0,50}}"
        rf"(\$?\s*[\d,]+(?:\.\d{{1,2}})?)",

        rf"household\s+size\s*(?:of\s*)?"
        rf"{re.escape(size)}"
        rf"[^$\d]{{0,50}}"
        rf"(\$?\s*[\d,]+(?:\.\d{{1,2}})?)",

        rf"size\s*{re.escape(size)}"
        rf"[^$\d]{{0,50}}"
        rf"(\$?\s*[\d,]+(?:\.\d{{1,2}})?)",
    ]

    for pattern in direct_patterns:

        match = re.search(
            pattern,
            text,
            re.I,
        )

        if match:

            amount = normalize_amount(
                match.group(1)
            )

            if amount:
                return amount

    # -----------------------------------------------------
    # People / persons
    # -----------------------------------------------------

    person_patterns = [

        rf"\b{re.escape(size)}\s+"
        rf"(?:persons?|people|members?)"
        rf"[^$\d]{{0,60}}"
        rf"(\$?\s*[\d,]+(?:\.\d{{1,2}})?)",

        rf"\b{re.escape(size)}\b"
        rf"[^$\d]{{0,30}}"
        rf"(\$?\s*[\d,]+(?:\.\d{{1,2}})?)",
    ]

    for pattern in person_patterns:

        match = re.search(
            pattern,
            text,
            re.I,
        )

        if match:

            amount = normalize_amount(
                match.group(1)
            )

            if amount:
                return amount

    return None


def find_policy_amount(results, household_size):

    if household_size is None:
        return None, None

    for result in results:

        text = result.get(
            "text",
            "",
        )

        amount = extract_amount_near_household(
            text,
            household_size,
        )

        if amount:
            return amount, result

    return None, None


# =========================================================
# CITIZEN LANGUAGE NORMALIZATION
# =========================================================

def normalize_citizen_question(question):
    """
    Converts common citizen-style wording into
    policy-friendly terminology.

    This is NOT a generative translation system.
    It only normalizes known phrases.
    """

    q = question.lower().strip()

    replacements = {

        # -------------------------------------------------
        # Benefit
        # -------------------------------------------------

        "how much can i get": "maximum benefit amount",
        "how much money can i get": "maximum benefit amount",
        "how much will i get": "maximum benefit amount",
        "what can i get": "maximum benefit amount",
        "how much benefit": "maximum benefit amount",

        # -------------------------------------------------
        # Income
        # -------------------------------------------------

        "how much can i earn": "income threshold",
        "how much income can i have": "income threshold",
        "income limit": "income threshold",
        "earning limit": "income threshold",
        "earnings limit": "income threshold",

        # -------------------------------------------------
        # Reporting
        # -------------------------------------------------

        "how long do i have": "reporting period",
        "how many days do i have": "reporting period",
        "when do i need to tell you": "reporting period",
        "when should i report": "reporting period",
        "do i need to report": "report change circumstance",

        # -------------------------------------------------
        # Sanction
        # -------------------------------------------------

        "will i be punished": "sanction",
        "will there be a penalty": "sanction",
        "penalty": "sanction",
        "punishment": "sanction",

        # -------------------------------------------------
        # Earnings disregard
        # -------------------------------------------------

        "money i can earn before it affects": "earnings disregard",
        "earn before benefit changes": "earnings disregard",
    }

    normalized = q

    for phrase, replacement in replacements.items():
        normalized = normalized.replace(
            phrase,
            replacement,
        )

    return normalized


# =========================================================
# INTENT DETECTION
# =========================================================

def detect_intent(question):
    """
    Detect the citizen's main policy intent.

    Returns a human-readable intent.
    """

    q = question.lower()

    if (
        "maximum benefit" in q
        or "maximum amount" in q
        or "maximum award" in q
        or "how much can i get" in q
        or "benefit amount" in q
    ):
        return "Maximum Benefit"

    if (
        "income" in q
        and (
            "threshold" in q
            or "limit" in q
        )
    ):
        return "Income Threshold"

    if "earnings disregard" in q:
        return "Earnings Disregard"

    if (
        "report" in q
        and (
            "change" in q
            or "circumstance" in q
        )
    ):
        return "Change Reporting"

    if (
        "sanction" in q
        or "penalty" in q
        or "punishment" in q
    ):
        return "Sanction"

    if (
        "claim period" in q
        or "spans 1 march 2026" in q
        or "crosses 1 march 2026" in q
    ):
        return "Cross-Period Rule"

    return "General Policy Question"


# =========================================================
# SIMPLE NEXT STEP
# =========================================================

def get_next_step(intent, grounded):

    if not grounded:

        return (
            "The available policy does not give enough "
            "information. Please contact the county "
            "benefits policy team before making a decision."
        )

    steps = {

        "Maximum Benefit":
            "Check your household size and compare it with "
            "the policy amount shown above.",

        "Income Threshold":
            "Check your household size and monthly income "
            "against the threshold shown above.",

        "Earnings Disregard":
            "Compare your monthly earnings with the "
            "earnings disregard amount shown above.",

        "Change Reporting":
            "Report the change within the required number "
            "of calendar days.",

        "Sanction":
            "Review the sanction rule and confirm whether "
            "the exception applies to your situation.",

        "Cross-Period Rule":
            "Check which policy figures were in force during "
            "each part of the claim period.",

        "General Policy Question":
            "Review the supporting policy clause and contact "
            "the benefits office if your situation is different.",
    }

    return steps.get(
        intent,
        steps["General Policy Question"],
    )


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
        # Load documents
        # -------------------------------------------------

        docs = load_documents(
            data_dir
        )

        # -------------------------------------------------
        # Chunk documents
        # -------------------------------------------------

        self.chunks = []

        for document in docs:

            self.chunks.extend(
                chunk_markdown(
                    document["text"],
                    document["source"],
                )
            )

        # -------------------------------------------------
        # Create embeddings
        # -------------------------------------------------

        texts = [
            chunk["text"]
            for chunk in self.chunks
        ]

        self.embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        # -------------------------------------------------
        # Hybrid retriever
        # -------------------------------------------------

        self.retriever = HybridRetriever(
            self.chunks,
            self.embeddings,
        )

    # =====================================================
    # RULE-BASED POLICY LOGIC
    # =====================================================

    def _answer_from_rules(self, q, results):

        ql = q.lower()

        # -------------------------------------------------
        # Extract entities
        # -------------------------------------------------

        date = extract_date(q)

        year = extract_year(q)

        household_size = extract_household_size(q)

        # -------------------------------------------------
        # Amendment applicability
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
            or "benefit amount" in ql
        ):

            if household_size is not None:

                amount, supporting_result = (
                    find_policy_amount(
                        results,
                        household_size,
                    )
                )

                if amount:

                    source = (
                        supporting_result.get(
                            "source",
                            "policy document",
                        )
                    )

                    section = (
                        supporting_result.get(
                            "section",
                            "policy section",
                        )
                    )

                    requested_year = (
                        year
                        if year is not None
                        else (
                            date[0]
                            if date is not None
                            else None
                        )
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
                            f"household of "
                            f"{household_size} is "
                            f"{amount}."
                        )

                    return (
                        answer,
                        f"{section} — {source}",
                        "Retrieved policy evidence",
                    )

                return (
                    (
                        f"I found policy information about "
                        f"the maximum benefit, but I could "
                        f"not find an exact amount for a "
                        f"household of {household_size}. "
                        f"I don't want to guess. Please "
                        f"contact the county benefits "
                        f"policy team."
                    ),
                    None,
                    "Maximum benefit amount not found "
                    "in the retrieved evidence",
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

            size = household_size

            if size is not None:

                threshold_table = {
                    1: 1225,
                    2: 1650,
                    3: 2075,
                    4: 2500,
                    5: 2925,
                }

                base = threshold_table.get(
                    size
                )

                if base is not None and post:

                    return (
                        (
                            f"For a determination made "
                            f"in {date[0]}, the monthly "
                            f"income threshold for a "
                            f"household of {size} is "
                            f"${base:,}."
                        ),
                        "§6.6.1 / Amendment §3.1",
                        "Amendment §5.1",
                    )

                if base is None:

                    return (
                        (
                            f"I don't have enough policy "
                            f"evidence to determine the "
                            f"monthly income threshold for "
                            f"a household of {size}. "
                            f"Please contact the county "
                            f"benefits policy team."
                        ),
                        None,
                        "Household size is outside "
                        "the available policy table",
                    )

        # =================================================
        # 3. EARNINGS DISREGARD
        # =================================================

        if "earnings disregard" in ql:

            if post:

                return (
                    "The earnings disregard is "
                    "$175 per month.",
                    "§6.4.1(a) / Amendment §1.1",
                    "Amendment §5.1",
                )

            if date is not None:

                return (
                    (
                        "The amendment does not apply "
                        "to a determination made before "
                        "1 March 2026. The pre-amendment "
                        "amount was $120 per month."
                    ),
                    "§6.4.1(a) / Amendment §1.1",
                    "Amendment §5.1",
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
                or "period" in ql
            )
        ):

            if date is not None:

                if post:

                    return (
                        (
                            "You have 14 calendar days "
                            "to report the change because "
                            "the change occurred on or "
                            "after 1 March 2026."
                        ),
                        "§4.3.2 / Amendment §2.1",
                        "Amendment §5.2",
                    )

                return (
                    (
                        "You had 10 calendar days to "
                        "report the change because "
                        "the change occurred before "
                        "1 March 2026."
                    ),
                    "§4.3.2 / Amendment §2.1",
                    "Amendment §5.2",
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
                    "If a claim period spans "
                    "1 March 2026, use the figures "
                    "that were in force on each day "
                    "of the period and apportion "
                    "the award accordingly under "
                    "§7.4.3."
                ),
                "§7.4.3 / Amendment §5.3",
                "Amendment §5.3",
            )

        # =================================================
        # 6. SANCTION
        # =================================================

        if (
            "sanction" in ql
            or "penalty" in ql
            or "punishment" in ql
        ):

            # -------------------------------------------------
            # Exception:
            # Change would increase award.
            # -------------------------------------------------

            if (
                "increase" in ql
                or "increased" in ql
            ):

                return (
                    (
                        "A sanction must not be imposed "
                        "where the change of circumstances "
                        "would have increased the award."
                    ),
                    "§10.5.3A / Amendment §4.2",
                    "Amendment §5.1",
                )

            # -------------------------------------------------
            # Post-amendment rule
            # -------------------------------------------------

            if post:

                return (
                    (
                        "The sanction rate is 15 per cent "
                        "for a determination made on or "
                        "after 1 March 2026."
                    ),
                    "§10.5.2 / Amendment §4.1",
                    "Amendment §5.1",
                )

        return None

    # =====================================================
    # MAIN ASK FUNCTION
    # =====================================================

    def ask(self, question):

        original_question = question.strip()

        # -------------------------------------------------
        # Normalize citizen wording
        # -------------------------------------------------

        normalized_question = (
            normalize_citizen_question(
                original_question
            )
        )

        # -------------------------------------------------
        # Detect intent
        # -------------------------------------------------

        intent = detect_intent(
            normalized_question
        )

        # -------------------------------------------------
        # Query embedding
        # -------------------------------------------------

        query_embedding = self.model.encode(
            [normalized_question],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )[0]

        # -------------------------------------------------
        # Retrieve policy evidence
        # -------------------------------------------------

        results = self.retriever.retrieve(
            normalized_question,
            query_embedding,
            top_k=8,
        )

        # -------------------------------------------------
        # Deterministic rules
        # -------------------------------------------------

        rule = self._answer_from_rules(
            normalized_question,
            results,
        )

        # =================================================
        # RULE ANSWER
        # =================================================

        if rule:

            answer, citation, effective = rule

            # -------------------------------------------------
            # Unsupported / insufficient evidence
            # -------------------------------------------------

            if citation is None:

                return {
                    "answer": answer,
                    "citation": None,
                    "effective_rule": effective,
                    "confidence": "low",
                    "evidence": results[:3],
                    "grounded": False,

                    # New citizen-friendly fields
                    "intent": intent,
                    "next_step": get_next_step(
                        intent,
                        False,
                    ),
                    "normalized_question":
                        normalized_question,
                }

            # -------------------------------------------------
            # Grounded answer
            # -------------------------------------------------

            return {
                "answer": answer,
                "citation": citation,
                "effective_rule": effective,
                "confidence": "high",
                "evidence": results[:3],
                "grounded": True,

                # New citizen-friendly fields
                "intent": intent,
                "next_step": get_next_step(
                    intent,
                    True,
                ),
                "normalized_question":
                    normalized_question,
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
                    "I don't know. The available "
                    "policy material does not provide "
                    "enough evidence to answer this "
                    "reliably. Please contact the "
                    "county benefits policy team."
                ),
                "citation": None,
                "effective_rule": None,
                "confidence": "low",
                "evidence": results[:3],
                "grounded": False,

                "intent": intent,
                "next_step": get_next_step(
                    intent,
                    False,
                ),
                "normalized_question":
                    normalized_question,
            }

        # =================================================
        # RELEVANT BUT NOT ENOUGH
        # =================================================

        return {
    "answer": (
        "I don't know. I found some related policy material, "
        "but it does not contain enough evidence to answer "
        "your specific question reliably. "
        "I don't want to guess or give you incorrect information. "
        "Please contact the appropriate county benefits policy team."
    ),
    "citation": None,
    "effective_rule": (
        "The retrieved policy evidence was not sufficient "
        "to support a specific answer."
    ),
    "confidence": "low",
    "evidence": results[:3],
    "grounded": False,
}