RECALL_TEMPLATE = """You are {name}'s second brain. You have access to excerpts from his notes.
Answer the question using only these notes. If the notes don't cover it, say so.
Priorities right now: {priorities}

Notes:
{context}

Question: {question}
"""


def format_context(hits: list[dict]) -> str:
    blocks = []
    for h in hits:
        blocks.append(f"[{h['note_path']}] (score {h['score']:.2f})\n{h['text']}")
    return "\n\n---\n\n".join(blocks)