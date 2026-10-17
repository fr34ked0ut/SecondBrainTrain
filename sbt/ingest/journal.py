import re
from datetime import date
from pathlib import Path

from .obsidian import Note

DATE_NAME = re.compile(r"(\d{4})-(\d{2})-(\d{2})")
SECTION = re.compile(r"^##\s+(.+)$", re.MULTILINE)


def is_journal(note: Note) -> bool:
    return bool(DATE_NAME.fullmatch(note.path.stem)) or "journal" in note.path.parts


def entry_date(note: Note) -> date | None:
    m = DATE_NAME.search(note.path.stem)
    if not m:
        return None
    return date(int(m[1]), int(m[2]), int(m[3]))


def split_sections(note: Note) -> dict[str, str]:
    parts = SECTION.split(note.body)
    if len(parts) < 3:
        return {"body": note.body}
    out = {}
    if parts[0].strip():
        out["preamble"] = parts[0].strip()
    for heading, content in zip(parts[1::2], parts[2::2]):
        out[heading.strip().lower()] = content.strip()
    return out


def to_qa_pairs(note: Note) -> list[dict]:
    d = entry_date(note)
    if d is None:
        return []
    sections = split_sections(note)
    pairs = []
    for key, text in sections.items():
        if len(text) < 40:
            continue
        pairs.append(
            {
                "instruction": f"What was on my mind regarding '{key}' on {d.isoformat()}?",
                "response": text,
                "source": str(note.path),
            }
        )
    return pairs