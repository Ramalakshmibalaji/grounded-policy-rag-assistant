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
# POLICY CONSTANTS
# =========================================================

AMENDMENT_DATE = (2026, 3)

# These values represent the policy rules already defined
# in the project. They are NEVER treated as sufficient
# evidence by themselves. Retrieved policy evidence must
# support the answer before it is marked as grounded.

INCOME_THRESHOLDS = {
    1: 1225,
    2: 1650,
    3: 2075,
    4: 2500,
    5: 2925,
}

PRE_AMENDMENT_EARNINGS_DISREGARD = 120
POST_AMENDMENT_EARNINGS_DISREGARD = 175

PRE_AMENDMENT_REPORTING_DAYS = 10
POST_AMENDMENT_REPORTING_DAYS = 14

POST_AMENDMENT_SANCTION_RATE = 15


# =========================================================
# DATE HELPERS
# =========================================================

def extract_date(text):
    """
    Extract Month + Year.

    Example:
        April 2026 -> (2026, 4)
    """

    match = DATE_RE.search(text)

    if not match:
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
        int(match.group(2)),
        months[match.group(1).lower()],
    )


def extract_year(text):
    """
    Extract a year.

    Example:
        2027 -> 2027
    """

    match = YEAR_RE.search(text)

    if not match:
        return None

    return int(match.group(1))


def month_at_or_after(
    year,
    month,
    y2=2026,
    m2=3,
):
    """
    Check whether date is on or after
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

        match = re.search(
            pattern,
            text,
        )

        if match:
            return int(match.group(1))

    return None


# =========================================================
# MONTHLY INCOME
# =========================================================

def extract_monthly_income(text):
    """
    Extract monthly household income.

    Examples:

        monthly income of $2,000
        monthly household income $2000
        income of $2,000 per month
        monthly income: $2,000
    """

    patterns = [

        r"monthly\s+(?:household\s+)?income\s+"
        r"(?:of\s+|is\s+|:?\s*)"
        r"\$?\s*([\d,]+(?:\.\d{1,2})?)",

        r"income\s+"
        r"(?:of\s+|is\s+|:?\s*)"
        r"\$?\s*([\d,]+(?:\.\d{1,2})?)"
        r"\s*(?:per\s+month|monthly)",

        r"\$?\s*([\d,]+(?:\.\d{1,2})?)"
        r"\s*(?:per\s+month|monthly)",
    ]

    for pattern in patterns:

        match = re.search(
            pattern,
            text,
            re.I,
        )

        if match:

            try:
                return float(
                    match.group(1).replace(",", "")
                )

            except ValueError:
                return None

    return None


# =========================================================
# MONEY HELPERS
# =========================================================

MONEY_RE = re.compile(
    r"\$?\s*[\d,]+(?:\.\d{1,2})?"
)


def normalize_amount(amount_text):

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

    # Do not treat a year as money.
    if 1900 <= value <= 2100:
        return None

    if value.is_integer():
        return f"${int(value):,}"

    return f"${value:,.2f}"


def extract_amount_near_household(
    text,
    household_size,
):
    """
    Extract exact policy amount appearing near
    the requested household size.

    IMPORTANT:
    This function does not calculate policy values.
    """

    if household_size is None:
        return None

    size = str(household_size)

    # -----------------------------------------------------
    # Markdown tables
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


def find_policy_amount(
    results,
    household_size,
):

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
# EVIDENCE HELPERS
# =========================================================

def clean_text(text):
    """
    Normalize whitespace for evidence processing.
    """

    if not text:
        return ""

    return re.sub(
        r"\s+",
        " ",
        str(text),
    ).strip()


def evidence_supports(
    result,
    required_terms=None,
    required_any=None,
):
    """
    Check whether retrieved evidence contains
    enough textual support.

    required_terms:
        Every term must appear.

    required_any:
        At least one term must appear.
    """

    if not result:
        return False

    text = clean_text(
        result.get("text", "")
    ).lower()

    if not text:
        return False

    if required_terms:

        for term in required_terms:

            if str(term).lower() not in text:
                return False

    if required_any:

        if not any(
            str(term).lower() in text
            for term in required_any
        ):
            return False

    return True


def find_supporting_evidence(
    results,
    required_terms=None,
    required_any=None,
):
    """
    Find the first retrieved evidence chunk
    that actually supports the required rule.
    """

    for result in results:

        if evidence_supports(
            result,
            required_terms=required_terms,
            required_any=required_any,
        ):
            return result

    return None


def extract_clause_from_result(
    result,
    section_hint=None,
):
    """
    Extract a useful policy excerpt from retrieved
    evidence.

    The actual retrieved text is preserved.
    """

    if not result:
        return None

    text = str(
        result.get("text", "")
    ).strip()

    if not text:
        return None

    # -----------------------------------------------------
    # If a section hint exists, try to locate it.
    # -----------------------------------------------------

    if section_hint:

        match = re.search(
            re.escape(section_hint),
            text,
            re.I,
        )

        if match:

            start = max(
                0,
                match.start() - 80,
            )

            end = min(
                len(text),
                match.end() + 500,
            )

            return text[start:end].strip()

    # -----------------------------------------------------
    # Otherwise return retrieved evidence itself.
    # -----------------------------------------------------

    return text


def build_citation(
    result,
    section_hint=None,
):
    """
    Build citation using the actual retrieved
    policy evidence.
    """

    if not result:
        return None

    source = result.get(
        "source",
        "policy document",
    )

    section = result.get(
        "section",
        "policy section",
    )

    clause = extract_clause_from_result(
        result,
        section_hint,
    )

    if not clause:
        return None

    return (
        f"{section} — {source}\n\n"
        f"{clause}"
    )


def result_score(result):
    """
    Safely read retriever score.
    """

    try:
        return float(
            result.get("score", 0)
        )

    except (
        TypeError,
        ValueError,
    ):
        return 0.0


def best_result(results):
    """
    Return highest-scoring result.
    """

    if not results:
        return None

    return max(
        results,
        key=result_score,
    )


# =========================================================
# CITIZEN LANGUAGE NORMALIZATION
# =========================================================

def normalize_citizen_question(question):

    q = question.lower().strip()

    replacements = {

        # Eligibility
        "can i qualify": "eligibility",
        "do i qualify": "eligibility",
        "am i eligible": "eligibility",
        "can i get the benefit": "eligibility",

        # Benefit
        "how much can i get":
            "maximum benefit amount",

        "how much money can i get":
            "maximum benefit amount",

        "how much will i get":
            "maximum benefit amount",

        "what can i get":
            "maximum benefit amount",

        "how much benefit":
            "maximum benefit amount",

        # Income
        "how much can i earn":
            "income threshold",

        "how much income can i have":
            "income threshold",

        "income limit":
            "income threshold",

        "earning limit":
            "income threshold",

        "earnings limit":
            "income threshold",

        # Reporting
        "how long do i have":
            "reporting period",

        "how many days do i have":
            "reporting period",

        "when do i need to tell you":
            "reporting period",

        "when should i report":
            "reporting period",

        "do i need to report":
            "report change circumstance",

        # Sanction
        "will i be punished":
            "sanction",

        "will there be a penalty":
            "sanction",

        "penalty":
            "sanction",

        "punishment":
            "sanction",

        # Earnings disregard
        "money i can earn before it affects":
            "earnings disregard",

        "earn before benefit changes":
            "earnings disregard",
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

    q = question.lower()

    if (
        "eligibility" in q
        or "eligible" in q
        or "qualify" in q
        or "qualification" in q
    ):
        return "Eligibility"

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
# NEXT STEP
# =========================================================

def get_next_step(
    intent,
    grounded,
):

    if not grounded:

        return (
            "The available policy does not give enough "
            "information. Please contact the county "
            "benefits policy team before making a decision."
        )

    steps = {

        "Eligibility":
            "Compare your monthly household income with "
            "the applicable income threshold for your "
            "household size.",

        "Maximum Benefit":
            "Check your household size and compare it "
            "with the policy amount shown above.",

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
            "Check which policy figures were in force "
            "during each part of the claim period.",

        "General Policy Question":
            "Review the supporting policy clause and "
            "contact the benefits office if your situation "
            "is different.",
    }

    return steps.get(
        intent,
        steps["General Policy Question"],
    )


# =========================================================
# GROUNDING ENGINE
# =========================================================

class GroundedEngine:

    def __init__(
        self,
        data_dir="data",
    ):

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

        if not docs:
            raise ValueError(
                f"No policy documents found in '{data_dir}'."
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

        if not self.chunks:
            raise ValueError(
                "Policy documents were loaded, "
                "but no chunks were created."
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
    # STANDARD RESULT BUILDERS
    # =====================================================

    def _grounded_result(
        self,
        answer,
        result,
        section_hint,
        effective_rule,
        intent,
        results,
        confidence="high",
    ):
        """
        Create a grounded response only when
        actual evidence exists.
        """

        citation = build_citation(
            result,
            section_hint,
        )

        if not citation:

            return self._abstain_result(
                (
                    "I don't know. I found related policy "
                    "material, but I could not identify a "
                    "specific supporting clause for this "
                    "answer. I don't want to guess."
                ),
                intent,
                results,
                "A specific supporting policy clause "
                "could not be identified.",
            )

        return {
            "answer": answer,
            "citation": citation,
            "effective_rule": effective_rule,
            "confidence": confidence,
            "evidence": results[:3],
            "grounded": True,
            "intent": intent,
            "next_step": get_next_step(
                intent,
                True,
            ),
            "normalized_question": None,
        }

    def _abstain_result(
        self,
        answer,
        intent,
        results,
        reason,
    ):

        return {
            "answer": answer,
            "citation": None,
            "effective_rule": reason,
            "confidence": "low",
            "evidence": results[:3],
            "grounded": False,
            "intent": intent,
            "next_step": get_next_step(
                intent,
                False,
            ),
            "normalized_question": None,
        }

    # =====================================================
    # RULE-BASED POLICY LOGIC
    # =====================================================

    def _answer_from_rules(
        self,
        q,
        results,
    ):

        ql = q.lower()

        date = extract_date(q)

        year = extract_year(q)

        household_size = (
            extract_household_size(q)
        )

        monthly_income = (
            extract_monthly_income(q)
        )

        # -------------------------------------------------
        # Amendment applicability
        # -------------------------------------------------

        post = (
            date is not None
            and month_at_or_after(*date)
        )

        # =================================================
        # 1. ELIGIBILITY
        # =================================================

        if (
            "eligibility" in ql
            or "eligible" in ql
            or "qualify" in ql
            or "qualification" in ql
        ):

            if household_size is None:

                return self._abstain_result(
                    (
                        "I need the household size to "
                        "determine eligibility from the "
                        "available policy."
                    ),
                    "Eligibility",
                    results,
                    "Household size was not provided.",
                )

            if monthly_income is None:

                return self._abstain_result(
                    (
                        "I need the monthly household income "
                        "to determine eligibility from the "
                        "available policy."
                    ),
                    "Eligibility",
                    results,
                    "Monthly household income was not provided.",
                )

            threshold = INCOME_THRESHOLDS.get(
                household_size
            )

            if threshold is None:

                return self._abstain_result(
                    (
                        f"I don't have enough policy evidence "
                        f"to determine the eligibility threshold "
                        f"for a household of "
                        f"{household_size}. Please contact "
                        f"the county benefits policy team."
                    ),
                    "Eligibility",
                    results,
                    "Household size is outside the "
                    "available policy threshold table.",
                )

            if date is not None and not post:

                return self._abstain_result(
                    (
                        "I cannot determine eligibility using "
                        "the post-amendment threshold because "
                        "the determination date is before "
                        "1 March 2026."
                    ),
                    "Eligibility",
                    results,
                    "The available threshold applies to "
                    "determinations on or after 1 March 2026.",
                )

            # -------------------------------------------------
            # IMPORTANT:
            # Require retrieved evidence to support the
            # policy amount before answering.
            # -------------------------------------------------

            threshold_text = (
                f"{threshold:,}"
            )

            supporting_result = (
                find_supporting_evidence(
                    results,
                    required_any=[
                        threshold_text,
                        f"${threshold_text}",
                    ],
                )
            )

            if supporting_result is None:

                return self._abstain_result(
                    (
                        "I don't know. I found policy material "
                        "related to eligibility, but the "
                        "retrieved evidence does not clearly "
                        "support the required income threshold "
                        "for this household size."
                    ),
                    "Eligibility",
                    results,
                    "The retrieved evidence did not support "
                    f"the threshold ${threshold:,}.",
                )

            if monthly_income <= threshold:

                answer = (
                    f"Yes. Based on the available policy, "
                    f"a household of {household_size} with "
                    f"a monthly income of "
                    f"${monthly_income:,.0f} is within the "
                    f"monthly income threshold of "
                    f"${threshold:,} for the applicable "
                    f"determination period."
                )

            else:

                answer = (
                    f"Based on the available policy, "
                    f"a household of {household_size} with "
                    f"a monthly income of "
                    f"${monthly_income:,.0f} is above the "
                    f"monthly income threshold of "
                    f"${threshold:,} for the applicable "
                    f"determination period."
                )

            return self._grounded_result(
                answer,
                supporting_result,
                "§6.6.1",
                "The retrieved policy evidence supports "
                "the applicable income threshold.",
                "Eligibility",
                results,
            )

        # =================================================
        # 2. MAXIMUM BENEFIT
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

            if household_size is None:

                return self._abstain_result(
                    (
                        "I need the household size to "
                        "identify the maximum benefit "
                        "from the available policy."
                    ),
                    "Maximum Benefit",
                    results,
                    "Household size was not provided.",
                )

            amount, supporting_result = (
                find_policy_amount(
                    results,
                    household_size,
                )
            )

            if amount and supporting_result:

                source = supporting_result.get(
                    "source",
                    "policy document",
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

                return self._grounded_result(
                    answer,
                    supporting_result,
                    supporting_result.get(
                        "section",
                        None,
                    ),
                    (
                        f"The amount was retrieved from "
                        f"the supporting policy evidence "
                        f"({source})."
                    ),
                    "Maximum Benefit",
                    results,
                )

            return self._abstain_result(
                (
                    f"I found policy information about "
                    f"the maximum benefit, but I could "
                    f"not find an exact amount for a "
                    f"household of {household_size}. "
                    f"I don't want to guess. Please "
                    f"contact the county benefits "
                    f"policy team."
                ),
                "Maximum Benefit",
                results,
                "Maximum benefit amount was not found "
                "in the retrieved evidence.",
            )

        # =================================================
        # 3. INCOME THRESHOLD
        # =================================================

        if (
            "income" in ql
            and (
                "threshold" in ql
                or "limit" in ql
            )
        ):

            size = household_size

            if size is None:

                return self._abstain_result(
                    (
                        "I need the household size to "
                        "identify the applicable income "
                        "threshold."
                    ),
                    "Income Threshold",
                    results,
                    "Household size was not provided.",
                )

            base = INCOME_THRESHOLDS.get(
                size
            )

            if base is None:

                return self._abstain_result(
                    (
                        f"I don't have enough policy "
                        f"evidence to determine the "
                        f"monthly income threshold for "
                        f"a household of {size}."
                    ),
                    "Income Threshold",
                    results,
                    "Household size is outside the "
                    "available policy table.",
                )

            if date is not None and not post:

                return self._abstain_result(
                    (
                        "I don't have enough evidence to "
                        "apply the post-amendment threshold "
                        "to a determination made before "
                        "1 March 2026."
                    ),
                    "Income Threshold",
                    results,
                    "Requested date is before the "
                    "amendment effective date.",
                )

            supporting_result = (
                find_supporting_evidence(
                    results,
                    required_any=[
                        f"{base:,}",
                        f"${base:,}",
                    ],
                )
            )

            if supporting_result is None:

                return self._abstain_result(
                    (
                        "I don't know. The retrieved policy "
                        "evidence does not clearly support "
                        "the exact income threshold for "
                        f"a household of {size}."
                    ),
                    "Income Threshold",
                    results,
                    "Retrieved evidence did not support "
                    f"${base:,}.",
                )

            answer = (
                f"For a determination made in "
                f"{date[0] if date else 'the applicable period'}, "
                f"the monthly income threshold for a "
                f"household of {size} is "
                f"${base:,}."
            )

            return self._grounded_result(
                answer,
                supporting_result,
                "§6.6.1",
                "The retrieved evidence supports "
                "the applicable income threshold.",
                "Income Threshold",
                results,
            )

        # =================================================
        # 4. EARNINGS DISREGARD
        # =================================================

        if "earnings disregard" in ql:

            if date is None:

                return self._abstain_result(
                    (
                        "I need the determination date to "
                        "identify which earnings disregard "
                        "rule applies."
                    ),
                    "Earnings Disregard",
                    results,
                    "Determination date was not provided.",
                )

            if post:

                amount = (
                    POST_AMENDMENT_EARNINGS_DISREGARD
                )

            else:

                amount = (
                    PRE_AMENDMENT_EARNINGS_DISREGARD
                )

            supporting_result = (
                find_supporting_evidence(
                    results,
                    required_any=[
                        f"{amount}",
                        f"${amount}",
                    ],
                )
            )

            if supporting_result is None:

                return self._abstain_result(
                    (
                        "I don't know. The retrieved "
                        "policy evidence does not clearly "
                        "support the earnings disregard "
                        "amount for the requested period."
                    ),
                    "Earnings Disregard",
                    results,
                    "Retrieved evidence did not support "
                    f"${amount}.",
                )

            if post:

                answer = (
                    "The earnings disregard is "
                    f"${amount} per month."
                )

            else:

                answer = (
                    "The amendment does not apply to a "
                    "determination made before 1 March 2026. "
                    f"The pre-amendment amount was "
                    f"${amount} per month."
                )

            return self._grounded_result(
                answer,
                supporting_result,
                "§6.4.1(a)",
                "The requested date was matched against "
                "the amendment effective date.",
                "Earnings Disregard",
                results,
            )

        # =================================================
        # 5. REPORTING PERIOD
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

            if date is None:

                return self._abstain_result(
                    (
                        "I need the date when the change "
                        "occurred to determine which "
                        "reporting period applies."
                    ),
                    "Change Reporting",
                    results,
                    "Change date was not provided.",
                )

            if post:

                days = (
                    POST_AMENDMENT_REPORTING_DAYS
                )

            else:

                days = (
                    PRE_AMENDMENT_REPORTING_DAYS
                )

            supporting_result = (
                find_supporting_evidence(
                    results,
                    required_any=[
                        f"{days} calendar days",
                        f"{days} days",
                        f"{days} calendar",
                    ],
                )
            )

            if supporting_result is None:

                return self._abstain_result(
                    (
                        "I don't know. The retrieved policy "
                        "evidence does not clearly support "
                        f"the {days}-day reporting period "
                        "for the requested date."
                    ),
                    "Change Reporting",
                    results,
                    "Retrieved evidence did not support "
                    f"the {days}-day reporting period.",
                )

            if post:

                answer = (
                    "You have 14 calendar days to report "
                    "the change because the change occurred "
                    "on or after 1 March 2026."
                )

            else:

                answer = (
                    "You had 10 calendar days to report "
                    "the change because the change occurred "
                    "before 1 March 2026."
                )

            return self._grounded_result(
                answer,
                supporting_result,
                "§4.3.2",
                "The reporting period was selected based "
                "on the amendment effective date.",
                "Change Reporting",
                results,
            )

        # =================================================
        # 6. CROSS-PERIOD RULE
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

            supporting_result = (
                find_supporting_evidence(
                    results,
                    required_any=[
                        "§7.4.3",
                        "apportion",
                        "1 March 2026",
                    ],
                )
            )

            if supporting_result is None:

                return self._abstain_result(
                    (
                        "I don't know. The retrieved policy "
                        "evidence does not clearly support "
                        "the cross-period rule."
                    ),
                    "Cross-Period Rule",
                    results,
                    "Cross-period policy evidence "
                    "was not sufficient.",
                )

            answer = (
                "If a claim period spans 1 March 2026, "
                "use the figures that were in force on "
                "each day of the period and apportion "
                "the award accordingly."
            )

            return self._grounded_result(
                answer,
                supporting_result,
                "§7.4.3",
                "The claim period crosses the "
                "amendment effective date.",
                "Cross-Period Rule",
                results,
            )

        # =================================================
        # 7. SANCTION
        # =================================================

        if (
            "sanction" in ql
            or "penalty" in ql
            or "punishment" in ql
        ):

            # -------------------------------------------------
            # Exception: change would increase award
            # -------------------------------------------------

            if (
                "increase" in ql
                or "increased" in ql
            ):

                supporting_result = (
                    find_supporting_evidence(
                        results,
                        required_any=[
                            "increased the award",
                            "increase the award",
                            "must not be imposed",
                            "sanction",
                        ],
                    )
                )

                if supporting_result is None:

                    return self._abstain_result(
                        (
                            "I don't know. The retrieved "
                            "policy evidence does not clearly "
                            "support the sanction exception "
                            "for an increased award."
                        ),
                        "Sanction",
                        results,
                        "Sanction exception was not "
                        "supported by retrieved evidence.",
                    )

                answer = (
                    "A sanction must not be imposed "
                    "where the change of circumstances "
                    "would have increased the award."
                )

                return self._grounded_result(
                    answer,
                    supporting_result,
                    "§10.5.3A",
                    "The retrieved evidence supports "
                    "the increased-award exception.",
                    "Sanction",
                    results,
                )

            # -------------------------------------------------
            # Post amendment sanction
            # -------------------------------------------------

            if post:

                supporting_result = (
                    find_supporting_evidence(
                        results,
                        required_any=[
                            "15 per cent",
                            "15%",
                            "15 percent",
                        ],
                    )
                )

                if supporting_result is None:

                    return self._abstain_result(
                        (
                            "I don't know. The retrieved "
                            "policy evidence does not clearly "
                            "support the sanction rate for "
                            "the requested period."
                        ),
                        "Sanction",
                        results,
                        "Retrieved evidence did not support "
                        "the 15 percent sanction rate.",
                    )

                answer = (
                    "The sanction rate is 15 per cent "
                    "for a determination made on or "
                    "after 1 March 2026."
                )

                return self._grounded_result(
                    answer,
                    supporting_result,
                    "§10.5.2",
                    "The sanction rule was selected "
                    "using the amendment effective date.",
                    "Sanction",
                    results,
                )

        return None

    # =====================================================
    # MAIN ASK FUNCTION
    # =====================================================

    def ask(self, question):

        original_question = (
            question.strip()
        )

        if not original_question:

            return {
                "answer": (
                    "Please enter a policy question."
                ),
                "citation": None,
                "effective_rule": None,
                "confidence": "low",
                "evidence": [],
                "grounded": False,
                "intent": "General Policy Question",
                "next_step": (
                    "Enter a specific question about "
                    "the benefits policy."
                ),
                "normalized_question": "",
            }

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
        # Retrieve evidence
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

            rule["normalized_question"] = (
                normalized_question
            )

            return rule

        # =================================================
        # CONSERVATIVE FALLBACK
        # =================================================

        if (
            not results
            or result_score(results[0]) < 0.20
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
                "I don't know. I found some related "
                "policy material, but it does not contain "
                "enough evidence to answer your specific "
                "question reliably. I don't want to guess "
                "or give you incorrect information. "
                "Please contact the appropriate county "
                "benefits policy team."
            ),
            "citation": None,
            "effective_rule": (
                "The retrieved policy evidence was not "
                "sufficient to support a specific answer."
            ),
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