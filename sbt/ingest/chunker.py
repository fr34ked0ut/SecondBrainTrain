from dataclasses import dataclass

from transformers import AutoTokenizer

from .obsidian import Note


@dataclass
class Chunk:
    note_path: str
    title: str
    idx: int
    text: str
    tags: list[str]


class Chunker:
    def __init__(self, tokenizer_name: str, max_tokens: int = 384, overlap: int = 48):
        self.tok = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_tokens = max_tokens
        self.overlap = overlap

    def split(self, note: Note) -> list[Chunk]:
        ids = self.tok.encode(note.body, add_special_tokens=False)
        if not ids:
            return []
        step = max(1, self.max_tokens - self.overlap)
        chunks = []
        for i, start in enumerate(range(0, len(ids), step)):
            window = ids[start : start + self.max_tokens]
            text = self.tok.decode(window, skip_special_tokens=True).strip()
            if text:
                chunks.append(Chunk(str(note.path), note.title, i, f"# {note.title}\n{text}", note.tags))
            if start + self.max_tokens >= len(ids):
                break
        return chunks