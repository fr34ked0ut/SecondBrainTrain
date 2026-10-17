import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import frontmatter

WIKILINK = re.compile(r"\[\[([^\]|#]+)(?:#[^\]|]*)?(?:\|[^\]]*)?\]\]")
TAG = re.compile(r"(?<!\w)#([A-Za-z0-9_/\-]+)")


@dataclass
class Note:
    path: Path
    title: str
    body: str
    tags: list[str]
    links: list[str]
    modified: datetime


def _matches(path: Path, patterns: list[str]) -> bool:
    return any(path.match(p) for p in patterns)


def load_vault(root: Path, include: list[str], exclude: list[str]) -> list[Note]:
    notes = []
    for pattern in include:
        for p in root.glob(pattern):
            rel = p.relative_to(root)
            if _matches(rel, exclude):
                continue
            post = frontmatter.load(p)
            body = post.content.strip()
            if not body:
                continue
            fm_tags = post.get("tags") or []
            if isinstance(fm_tags, str):
                fm_tags = [t.strip() for t in fm_tags.split(",")]
            tags = sorted(set(fm_tags) | set(TAG.findall(body)))
            notes.append(
                Note(
                    path=rel,
                    title=post.get("title", p.stem),
                    body=body,
                    tags=tags,
                    links=WIKILINK.findall(body),
                    modified=datetime.fromtimestamp(p.stat().st_mtime),
                )
            )
    return notes


def filter_private(notes: list[Note], blocked: list[str]) -> list[Note]:
    return [n for n in notes if not any(t.startswith(b) for t in n.tags for b in blocked)]