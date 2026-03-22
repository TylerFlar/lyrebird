"""Combine Balatro and StS card data into a single card_corpus.json."""

import json
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
BALATRO_PATH = DATA_DIR / "balatro_cards.json"
STS_PATH = DATA_DIR / "sts_cards.json"
OUTPUT = DATA_DIR / "card_corpus.json"


def main():
    with open(BALATRO_PATH, "r") as f:
        balatro_cards = json.load(f)

    with open(STS_PATH, "r") as f:
        sts_cards = json.load(f)

    corpus = balatro_cards + sts_cards

    with open(OUTPUT, "w") as f:
        json.dump(corpus, f, indent=2)

    # Summary stats
    balatro_subtypes = {}
    for c in balatro_cards:
        st = c["card_subtype"]
        balatro_subtypes[st] = balatro_subtypes.get(st, 0) + 1

    sts_subtypes = {}
    for c in sts_cards:
        st = c["card_subtype"]
        sts_subtypes[st] = sts_subtypes.get(st, 0) + 1

    print("=== Card Corpus Summary ===")
    print(f"\nBalatro: {len(balatro_cards)} cards")
    for st, count in sorted(balatro_subtypes.items()):
        print(f"  {st}: {count}")

    print(f"\nStS: {len(sts_cards)} cards")
    for st, count in sorted(sts_subtypes.items()):
        print(f"  {st}: {count}")

    print(f"\nTotal corpus: {len(corpus)} cards")
    print(f"Output: {OUTPUT}")


if __name__ == "__main__":
    main()
