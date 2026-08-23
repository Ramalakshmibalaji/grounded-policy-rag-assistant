from pathlib import Path
import re

SECTION_RE = re.compile(r"^#{1,6}\s*(.+?)\s*$", re.M)

def load_documents(data_dir: str):
    docs = []
    for path in sorted(Path(data_dir).glob("*.md")):
        text = path.read_text(encoding="utf-8")
        docs.append({"source": path.name, "text": text})
    return docs

def chunk_markdown(text: str, source: str):
    lines = text.splitlines()
    chunks = []
    current = []
    headings = []

    def flush():
        body = "\n".join(current).strip()
        if not body:
            return
        section = headings[-1] if headings else "Document"
        # Keep small passages manageable for retrieval.
        parts = re.split(r"\n\s*\n", body)
        for i, part in enumerate(parts):
            part = part.strip()
            if not part:
                continue
            chunks.append({
                "chunk_id": f"{source}:{len(chunks)+1}",
                "source": source,
                "section": section,
                "text": part,
            })

    for line in lines:
        m = SECTION_RE.match(line)
        if m:
            flush()
            current = []
            headings.append(m.group(1).strip())
        else:
            current.append(line)
    flush()
    return chunks
