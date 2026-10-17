import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Settings:
    vault_path: Path = field(default_factory=lambda: Path(os.getenv("SBT_VAULT_PATH", "~/Vault")).expanduser())
    db_path: Path = field(default_factory=lambda: Path(os.getenv("SBT_DB_PATH", ROOT / "data/processed/memory.db")))
    device: str = field(default_factory=lambda: os.getenv("SBT_DEVICE", "cpu"))
    raw: dict = field(default_factory=dict)

    def __getitem__(self, key):
        return self.raw[key]


def load(path: Path | None = None) -> Settings:
    path = path or ROOT / "config/default.yaml"
    with open(path) as f:
        raw = yaml.safe_load(f)
    persona_path = ROOT / "config/persona.yaml"
    if persona_path.exists():
        with open(persona_path) as f:
            raw["persona"] = yaml.safe_load(f)
    return Settings(raw=raw)