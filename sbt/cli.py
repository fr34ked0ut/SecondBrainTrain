from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from . import config
from .ingest.chunker import Chunker
from .ingest.journal import is_journal, to_qa_pairs
from .ingest.obsidian import filter_private, load_vault
from .memory.embedder import Embedder
from .memory.vector_store import VectorStore

app = typer.Typer(add_completion=False)
console = Console()
DATASET = config.ROOT / "data/processed/train.jsonl"


def _notes(cfg):
    v = cfg["vault"]
    notes = load_vault(cfg.vault_path, v["include"], v["exclude"])
    blocked = [b["tag"] for b in cfg.raw.get("persona", {}).get("do_not_surface", [])]
    return filter_private(notes, blocked)


@app.command()
def ingest():
    from .train.dataset import build_examples, save_jsonl

    cfg = config.load()
    notes = _notes(cfg)
    chunker = Chunker(cfg["training"]["base_model"], **cfg["chunking"])
    journals = [n for n in notes if is_journal(n)]
    pairs = [p for j in journals for p in to_qa_pairs(j)]
    chunks = [c for n in notes if not is_journal(n) for c in chunker.split(n)]
    examples = build_examples(pairs, chunks, cfg.raw.get("persona", {}))
    save_jsonl(examples, DATASET)
    console.print(f"[green]{len(notes)} notes, {len(journals)} journals, {len(examples)} examples[/]")


@app.command()
def embed():
    cfg = config.load()
    notes = _notes(cfg)
    chunker = Chunker(cfg["training"]["base_model"], **cfg["chunking"])
    e = cfg["embedding"]
    embedder = Embedder(e["model"], cfg.device, e["batch_size"])
    store = VectorStore(cfg.db_path, embedder.dim)
    chunks = [c for n in notes for c in chunker.split(n)]
    vecs = embedder.encode([c.text for c in chunks])
    store.upsert(chunks, vecs)
    console.print(f"[green]{store.count()} chunks in store[/]")


@app.command()
def train(epochs: int = typer.Option(None)):
    from .train.trainer import run

    cfg = config.load()
    if epochs:
        cfg.raw["training"]["epochs"] = epochs
    out = run(cfg.raw, DATASET, cfg.device)
    console.print(f"[green]adapter saved to {out}[/]")


@app.command()
def ask(question: str, adapter: Path = typer.Option(None), show_sources: bool = True):
    from .agent.recall import SecondBrain

    cfg = config.load()
    adapter = adapter or (Path(cfg["training"]["output_dir"]) / "latest" / "adapter")
    e = cfg["embedding"]
    embedder = Embedder(e["model"], cfg.device, e["batch_size"])
    store = VectorStore(cfg.db_path, embedder.dim)
    brain = SecondBrain(cfg, embedder, store, adapter if adapter.exists() else None)
    answer, hits = brain.ask(question)
    console.print(answer)
    if show_sources and hits:
        t = Table("score", "note")
        for h in hits:
            t.add_row(f"{h['score']:.2f}", h["note_path"])
        console.print(t)


def main():
    app()


if __name__ == "__main__":
    main()