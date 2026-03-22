"""Validate card_corpus.json against the shared schema requirements."""

import json
import sys
from pathlib import Path

CORPUS_PATH = Path(__file__).resolve().parent.parent / "data" / "card_corpus.json"


def main():
    with open(CORPUS_PATH, "r") as f:
        corpus = json.load(f)

    errors = []
    ids_seen = set()

    balatro_count = 0
    sts_count = 0

    for i, card in enumerate(corpus):
        prefix = f"card[{i}] ({card.get('id', '???')})"

        # Required non-null fields
        for field in ("id", "game", "name", "card_type", "cost"):
            if card.get(field) is None:
                errors.append(f"{prefix}: missing or null '{field}'")

        # Duplicate ID check
        card_id = card.get("id")
        if card_id in ids_seen:
            errors.append(f"{prefix}: duplicate ID")
        ids_seen.add(card_id)

        # Type checks
        if not isinstance(card.get("numeric_effects", None), dict):
            errors.append(f"{prefix}: numeric_effects is not a dict")
        if not isinstance(card.get("keywords", None), list):
            errors.append(f"{prefix}: keywords is not a list")

        rules = card.get("rules_text")
        if not rules or not isinstance(rules, str):
            errors.append(f"{prefix}: rules_text is empty or not a string")

        # Count by game
        game = card.get("game")
        if game == "balatro":
            balatro_count += 1
        elif game == "sts":
            sts_count += 1

    # Count checks
    if balatro_count < 150:
        errors.append(f"Expected 150+ Balatro cards, got {balatro_count}")
    if sts_count < 41:
        errors.append(f"Expected 41+ StS cards, got {sts_count}")

    # Report
    print(f"Validated {len(corpus)} cards ({balatro_count} Balatro, {sts_count} StS)")

    if errors:
        print(f"\n{len(errors)} error(s):")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)
    else:
        print("All checks passed!")
        sys.exit(0)


if __name__ == "__main__":
    main()
