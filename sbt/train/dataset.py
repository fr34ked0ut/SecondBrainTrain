import json
import random
from pathlib import Path

from datasets import Dataset

SYSTEM = "You are {name}'s second brain. Answer in his voice: {voice}."


def build_examples(journal_pairs: list[dict], note_chunks: list, persona: dict) -> list[dict]:
    system = SYSTEM.format(name=persona.get("name", "the user"), voice=persona.get("voice", "plain"))
    examples = []
    for p in journal_pairs:
        examples.append(
            {
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": p["instruction"]},
                    {"role": "assistant", "content": p["response"]},
                ]
            }
        )
    for c in note_chunks:
        examples.append(
            {
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": f"What do I know about {c.title}?"},
                    {"role": "assistant", "content": c.text.split("\n", 1)[-1]},
                ]
            }
        )
    random.shuffle(examples)
    return examples


def save_jsonl(examples: list[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")


def load_tokenized(path: Path, tokenizer, max_len: int) -> Dataset:
    ds = Dataset.from_json(str(path))

    def tok(batch):
        texts = [tokenizer.apply_chat_template(m, tokenize=False) for m in batch["messages"]]
        enc = tokenizer(texts, truncation=True, max_length=max_len, padding=False)
        enc["labels"] = [ids.copy() for ids in enc["input_ids"]]
        return enc

    return ds.map(tok, batched=True, remove_columns=ds.column_names).train_test_split(test_size=0.05, seed=7)