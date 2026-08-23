import re
import math
from collections import Counter

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


# =========================================================
# TOKENIZATION
# =========================================================

TOKEN_RE = re.compile(
    r"[a-z0-9$§.%+\-/]+"
)


def tokenize(text):
    """
    Tokenize policy/query text while preserving
    useful policy tokens such as:
        $175
        §6.4.1(a)
        15%
        14
        1 march 2026
    """

    return [
        token
        for token in TOKEN_RE.findall(
            str(text).lower()
        )
        if len(token) > 1
    ]


# =========================================================
# BM25
# =========================================================

class BM25:

    def __init__(
        self,
        docs,
        k1=1.5,
        b=0.75,
    ):

        self.docs = [
            tokenize(doc)
            for doc in docs
        ]

        self.n = len(
            self.docs
        )

        self.k1 = k1
        self.b = b

        self.lengths = np.array(
            [
                max(
                    1,
                    len(doc)
                )
                for doc in self.docs
            ],
            dtype=float,
        )

        self.avgdl = (
            float(
                self.lengths.mean()
            )
            if self.n
            else 1.0
        )

        self.tf = [
            Counter(doc)
            for doc in self.docs
        ]

        df = Counter()

        for doc in self.docs:

            df.update(
                set(doc)
            )

        self.idf = {

            token: math.log(
                1
                + (
                    self.n
                    - frequency
                    + 0.5
                )
                / (
                    frequency
                    + 0.5
                )
            )

            for token, frequency
            in df.items()
        }


    def scores(self, query):

        scores = np.zeros(
            self.n,
            dtype=float,
        )

        query_tokens = tokenize(
            query
        )

        for term in query_tokens:

            idf = self.idf.get(
                term
            )

            if idf is None:
                continue

            for index, tf in enumerate(
                self.tf
            ):

                frequency = tf.get(
                    term,
                    0,
                )

                if not frequency:
                    continue

                denominator = (
                    frequency
                    + self.k1
                    * (
                        1
                        - self.b
                        + self.b
                        * self.lengths[index]
                        / self.avgdl
                    )
                )

                scores[index] += (
                    idf
                    * (
                        frequency
                        * (self.k1 + 1)
                    )
                    / denominator
                )

        return scores


# =========================================================
# NORMALIZATION
# =========================================================

def minmax(values):

    values = np.asarray(
        values,
        dtype=float,
    )

    if (
        len(values) == 0
        or np.ptp(values) == 0
    ):

        return np.zeros_like(
            values
        )

    return (
        values - values.min()
    ) / np.ptp(values)


# =========================================================
# EXACT POLICY SIGNALS
# =========================================================

def exact_policy_boost(
    query,
    chunk,
):
    """
    Add a small deterministic retrieval boost when
    policy-critical terms from the query are explicitly
    present in a chunk.

    This does NOT generate or calculate policy values.

    It only improves retrieval ranking so that exact
    policy evidence can be found by the grounding layer.
    """

    query_lower = str(
        query
    ).lower()

    text_lower = str(
        chunk.get(
            "text",
            ""
        )
    ).lower()

    section = str(
        chunk.get(
            "section",
            ""
        )
    ).lower()

    source = str(
        chunk.get(
            "source",
            ""
        )
    ).lower()

    boost = 0.0

    # -----------------------------------------------------
    # Policy section references
    # -----------------------------------------------------

    section_patterns = [
        r"§\s*6\.4\.1",
        r"§\s*6\.6\.1",
        r"§\s*4\.3\.2",
        r"§\s*10\.5\.2",
        r"§\s*10\.5\.3a",
        r"§\s*7\.4\.3",
    ]

    for pattern in section_patterns:

        if re.search(
            pattern,
            query_lower,
            re.I,
        ):

            if (
                re.search(
                    pattern,
                    text_lower,
                    re.I,
                )
                or re.search(
                    pattern,
                    section,
                    re.I,
                )
            ):

                boost += 0.18


    # -----------------------------------------------------
    # Policy-specific terms
    # -----------------------------------------------------

    term_groups = {

        "earnings disregard": [
            "earnings disregard",
            "§6.4.1",
        ],

        "income threshold": [
            "income threshold",
            "§6.6.1",
        ],

        "reporting period": [
            "reporting period",
            "change of circumstances",
            "§4.3.2",
        ],

        "sanction": [
            "sanction",
            "§10.5.2",
            "§10.5.3a",
        ],

        "claim period": [
            "claim period",
            "apportion",
            "§7.4.3",
        ],
    }


    for intent, terms in term_groups.items():

        if intent in query_lower:

            if any(
                term in text_lower
                or term in section
                or term in source
                for term in terms
            ):

                boost += 0.20


    # -----------------------------------------------------
    # Exact dates
    # -----------------------------------------------------

    dates = re.findall(
        r"\b(?:january|february|march|april|may|june|"
        r"july|august|september|october|november|december)"
        r"\s+20\d{2}\b",
        query_lower,
    )

    for date_text in dates:

        if date_text in text_lower:

            boost += 0.10


    # -----------------------------------------------------
    # Exact policy amounts
    # -----------------------------------------------------

    amounts = re.findall(
        r"\$?\s*\d[\d,]*(?:\.\d{1,2})?",
        query_lower,
    )

    for amount in amounts:

        normalized = (
            amount
            .replace(
                "$",
                "",
            )
            .replace(
                ",",
                "",
            )
            .strip()
        )

        if not normalized:
            continue

        if normalized in {
            "2026",
            "2025",
        }:
            continue

        # Match either "$175" or "175"
        amount_forms = [
            f"${normalized}",
            normalized,
        ]

        if any(
            form in text_lower
            for form in amount_forms
        ):

            boost += 0.12


    # -----------------------------------------------------
    # Amendment file signal
    # -----------------------------------------------------

    amendment_terms = [
        "amendment",
        "amendment_2026_01",
        "amendment 2026-01",
    ]

    # When query contains amendment-sensitive wording,
    # give amendment chunks a small ranking advantage.
    if (
        "march 2026" in query_lower
        or "april 2026" in query_lower
        or "may 2026" in query_lower
        or "june 2026" in query_lower
        or "july 2026" in query_lower
        or "august 2026" in query_lower
        or "september 2026" in query_lower
        or "october 2026" in query_lower
        or "november 2026" in query_lower
        or "december 2026" in query_lower
    ):

        if any(
            term in source
            for term in amendment_terms
        ):

            boost += 0.15

        if (
            "amendment"
            in text_lower
        ):

            boost += 0.05


    return boost


# =========================================================
# HYBRID RETRIEVER
# =========================================================

class HybridRetriever:

    def __init__(
        self,
        chunks,
        embeddings,
    ):

        self.chunks = chunks

        self.embeddings = np.asarray(
            embeddings,
            dtype=np.float32,
        )

        self.bm25 = BM25(
            [
                chunk["text"]
                for chunk in chunks
            ]
        )


    def retrieve(
        self,
        query,
        query_embedding,
        top_k=20,
    ):

        if not self.chunks:
            return []


        # -------------------------------------------------
        # Dense semantic similarity
        # -------------------------------------------------

        dense = cosine_similarity(
            np.asarray(
                query_embedding
            ).reshape(
                1,
                -1,
            ),
            self.embeddings,
        )[0]


        # -------------------------------------------------
        # BM25 lexical similarity
        # -------------------------------------------------

        lexical = self.bm25.scores(
            query
        )


        # -------------------------------------------------
        # Normalize retrieval signals
        # -------------------------------------------------

        dense_norm = minmax(
            dense
        )

        lexical_norm = minmax(
            lexical
        )


        # -------------------------------------------------
        # Base hybrid score
        # -------------------------------------------------

        score = (
            0.72 * dense_norm
            + 0.28 * lexical_norm
        )


        # -------------------------------------------------
        # Exact policy evidence boost
        # -------------------------------------------------

        exact_boost = np.array(
            [
                exact_policy_boost(
                    query,
                    chunk,
                )
                for chunk in self.chunks
            ],
            dtype=float,
        )


        score = (
            score
            + exact_boost
        )


        # -------------------------------------------------
        # Rank
        # -------------------------------------------------

        order = np.argsort(
            -score
        )[:top_k]


        results = []

        for index in order:

            result = {

                **self.chunks[index],

                "score": float(
                    score[index]
                ),

                "dense_score": float(
                    dense[index]
                ),

                "lexical_score": float(
                    lexical[index]
                ),

                "exact_policy_boost": float(
                    exact_boost[index]
                ),
            }

            results.append(
                result
            )


        return results