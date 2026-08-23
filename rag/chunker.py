from pathlib import Path
import re


# =========================================================
# MARKDOWN HEADING / POLICY CLAUSE DETECTION
# =========================================================

SECTION_RE = re.compile(
    r"^(?:\*\*)?#{1,6}\s*(.+?)\s*(?:\*\*)?\s*$",
    re.M,
)

CLAUSE_RE = re.compile(
    r"^(?:\*\*)?\s*"
    r"(\d+(?:\.\d+)*[A-Za-z]?)"
    r"\s*(?:\*\*)?"
    r"\s*(.*)"
    r"\s*$",
    re.M,
)


# =========================================================
# DOCUMENT LOADER
# =========================================================

def load_documents(data_dir: str):

    docs = []

    for path in sorted(
        Path(data_dir).glob("*.md")
    ):

        text = path.read_text(
            encoding="utf-8"
        )

        docs.append(
            {
                "source": path.name,
                "text": text,
            }
        )

    return docs


# =========================================================
# MARKDOWN CHUNKER
# =========================================================

def chunk_markdown(
    text: str,
    source: str,
):

    lines = text.splitlines()

    chunks = []

    current = []

    current_section = "Document"

    current_clause = None


    def flush():

        nonlocal current

        body = "\n".join(
            current
        ).strip()

        if not body:
            current = []
            return


        parts = re.split(
            r"\n\s*\n",
            body,
        )


        for part in parts:

            part = part.strip()

            if not part:
                continue


            # -------------------------------------------------
            # Build a meaningful section label
            # -------------------------------------------------

            if current_clause:

                section = (
                    f"{current_section} "
                    f"§{current_clause}"
                )

            else:

                section = current_section


            chunks.append(
                {
                    "chunk_id": (
                        f"{source}:"
                        f"{len(chunks) + 1}"
                    ),

                    "source": source,

                    "section": section,

                    "text": part,
                }
            )


        current = []


    for line in lines:

        stripped = line.strip()


        # =====================================================
        # Markdown headings
        # =====================================================

        heading_match = SECTION_RE.match(
            stripped
        )

        if heading_match:

            flush()

            current_section = (
                heading_match.group(1)
                .strip()
            )

            current_clause = None

            continue


        # =====================================================
        # Numbered policy clauses
        #
        # Examples:
        # **1.1** In §6.4.1(a)...
        # **2.1** In §4.3.2...
        # **5.1** The amendments...
        # =====================================================

        clause_match = CLAUSE_RE.match(
            stripped
        )

        if (
            clause_match
            and re.match(
                r"^\d+(?:\.\d+)+[A-Za-z]?$",
                clause_match.group(1),
            )
        ):

            # Flush previous clause before starting
            # a new one.
            flush()

            current_clause = (
                clause_match.group(1)
            )

            remainder = (
                clause_match.group(2)
                .strip()
            )

            if remainder:
                current.append(
                    remainder
                )

            continue


        # =====================================================
        # Normal content
        # =====================================================

        current.append(line)


    flush()

    return chunks