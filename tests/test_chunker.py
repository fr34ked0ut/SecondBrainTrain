from datetime import datetime
from pathlib import Path

from sbt.ingest.chunker import Chunker
from sbt.ingest.obsidian import Note


def _note(body):
    return Note(Path("x.md"), "x", body, [], [], datetime.now())


def test_short_note_single_chunk():
    c = Chunker("gpt2", max_tokens=64, overlap=8)
    assert len(c.split(_note("hello there general kenobi"))) == 1


def test_long_note_overlaps():
    c = Chunker("gpt2", max_tokens=32, overlap=8)
    chunks = c.split(_note(" ".join(f"word{i}" for i in range(200))))
    assert len(chunks) > 3
    assert all(ch.text.startswith("# x") for ch in chunks)


def test_empty_note():
    c = Chunker("gpt2")
    assert c.split(_note("")) == []