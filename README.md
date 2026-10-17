# SecondBrainTrain
CHOO CHOOO

Personal LoRA + retrieval pipeline over my Obsidian vault and daily journals.

## Setup

    python -m venv .venv && source .venv/bin/activate
    pip install -e .
    cp .env.example .env

## Usage

    sbt ingest
    sbt embed
    sbt train --epochs 3
    sbt ask "what did I decide about the apartment?"
    sbt ask "am i ok"
    sbt ask "can fish fly if they grow wings"