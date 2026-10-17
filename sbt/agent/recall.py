from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..memory.embedder import Embedder
from ..memory.vector_store import VectorStore
from .prompts import RECALL_TEMPLATE, format_context


class SecondBrain:
    def __init__(self, cfg, embedder: Embedder, store: VectorStore, adapter: Path | None = None):
        self.cfg = cfg
        self.embedder = embedder
        self.store = store
        self.persona = cfg.raw.get("persona", {})
        base = cfg["training"]["base_model"]
        self.tok = AutoTokenizer.from_pretrained(adapter or base)
        model = AutoModelForCausalLM.from_pretrained(
            base,
            torch_dtype=torch.bfloat16 if cfg.device == "cuda" else torch.float32,
            device_map="auto" if cfg.device == "cuda" else None,
        )
        self.model = PeftModel.from_pretrained(model, adapter) if adapter else model
        self.model.eval()

    def retrieve(self, question: str) -> list[dict]:
        r = self.cfg["recall"]
        qv = self.embedder.encode([question])[0]
        blocked = [b["tag"] for b in self.persona.get("do_not_surface", [])]
        hits = self.store.search(qv, k=r["top_k"] * 2, min_score=r["min_score"])
        hits = [h for h in hits if not any(t.startswith(b) for t in h["tags"] for b in blocked)]
        return hits[: r["top_k"]]

    @torch.inference_mode()
    def ask(self, question: str, max_new_tokens: int = 400) -> tuple[str, list[dict]]:
        hits = self.retrieve(question)
        prompt = RECALL_TEMPLATE.format(
            name=self.persona.get("name", "the user"),
            priorities=", ".join(self.persona.get("priorities", [])),
            context=format_context(hits) if hits else "(none)",
            question=question,
        )
        msgs = [{"role": "user", "content": prompt}]
        ids = self.tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to(self.model.device)
        out = self.model.generate(ids, max_new_tokens=max_new_tokens, temperature=0.6, top_p=0.9, do_sample=True)
        answer = self.tok.decode(out[0, ids.shape[1] :], skip_special_tokens=True).strip()
        return answer, hits