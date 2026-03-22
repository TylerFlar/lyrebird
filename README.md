# Lyrebird

Cross-game card research project. Extracts card data from Balatro and Slay the Spire into a unified JSON corpus for downstream encoder experiments.

## Setup

```bash
uv venv && source .venv/bin/activate
```

No external dependencies required — stdlib only.

## Usage

Run the extraction scripts in order:

```bash
python scripts/extract_balatro_cards.py   # → data/balatro_cards.json
python scripts/extract_sts_cards.py       # → data/sts_cards.json
python scripts/build_corpus.py            # → data/card_corpus.json
python scripts/validate_corpus.py         # checks schema validity
```

## Repo layout

Expects sibling directories:
- `../jackdaw-balatro/` — Balatro engine with `jackdaw/engine/data/centers.json`
- `../decapitate-the-spire/` — StS engine with `decapitate_the_spire/game.py`

## Schema

Each card in `data/card_corpus.json` follows this structure:

```json
{
  "id": "balatro:j_joker",
  "game": "balatro",
  "name": "Joker",
  "card_type": "modifier",
  "card_subtype": "joker",
  "cost": 2,
  "rarity": "common",
  "numeric_effects": {"mult_flat": 4},
  "keywords": ["copyable", "can_be_eternal", "can_perish", "mult"],
  "rules_text": "+4 Mult",
  "raw": { ... }
}
```
