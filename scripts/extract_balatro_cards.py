"""Extract card data from Balatro's centers.json and cards.json into a unified format."""

import json
from pathlib import Path

BALATRO_DATA = Path(__file__).resolve().parent.parent.parent / "jackdaw-balatro" / "jackdaw" / "engine" / "data"
OUTPUT = Path(__file__).resolve().parent.parent / "data" / "balatro_cards.json"

RARITY_MAP = {1: "common", 2: "uncommon", 3: "rare", 4: "legendary"}

# card_type / card_subtype by set
SET_TYPE_MAP = {
    "Joker": ("modifier", "joker"),
    "Tarot": ("utility", "tarot"),
    "Planet": ("scaling", "planet"),
    "Spectral": ("utility", "spectral"),
    "Voucher": ("power", "voucher"),
}


def extract_numeric_effects(config):
    """Flatten config into numeric_effects dict."""
    if not config or isinstance(config, list):
        return {}

    effects = {}

    # Top-level known keys
    if "mult" in config:
        effects["mult_flat"] = config["mult"]
    if "Xmult" in config:
        effects["mult_x"] = config["Xmult"]

    extra = config.get("extra")
    if extra is None:
        pass
    elif isinstance(extra, dict):
        # Known mappings from extra
        key_map = {
            "s_mult": "mult_flat",
            "Xmult": "mult_x",
            "chips": "chips",
            "chip_mod": "chip_mod",
            "dollars": "money",
            "money": "money",
            "h_size": "hand_size",
            "d_size": "discard",
        }
        for k, v in extra.items():
            mapped = key_map.get(k, k)
            # Don't overwrite a top-level mapping
            if mapped not in effects:
                effects[mapped] = v
    elif isinstance(extra, (int, float)):
        effects["extra"] = extra

    # Other top-level config keys worth capturing
    for k in ("hand_type", "max_highlighted", "mod_conv"):
        if k in config:
            effects[k] = config[k]

    return effects


def extract_keywords(entry):
    """Extract keywords from card properties."""
    keywords = []
    if entry.get("blueprint_compat"):
        keywords.append("copyable")
    if entry.get("eternal_compat"):
        keywords.append("can_be_eternal")
    if entry.get("perishable_compat"):
        keywords.append("can_perish")
    effect = entry.get("effect")
    if effect and isinstance(effect, str):
        keywords.append(effect.lower().replace(" ", "_"))
    return keywords


def build_rules_text(entry, numeric_effects):
    """Build a simple templated rules_text."""
    name = entry["name"]
    card_set = entry.get("set", "")

    # Jokers
    if card_set == "Joker":
        parts = []
        if "mult_flat" in numeric_effects:
            parts.append(f"+{numeric_effects['mult_flat']} Mult")
        if "mult_x" in numeric_effects:
            parts.append(f"x{numeric_effects['mult_x']} Mult")
        if "chips" in numeric_effects:
            parts.append(f"+{numeric_effects['chips']} Chips")
        if "money" in numeric_effects:
            parts.append(f"+${numeric_effects['money']}")
        if "hand_size" in numeric_effects:
            parts.append(f"+{numeric_effects['hand_size']} hand size")

        # Add suit condition if present
        config = entry.get("config", {})
        extra = config.get("extra", {}) if isinstance(config, dict) else {}
        if isinstance(extra, dict) and "suit" in extra:
            condition = f" if played hand contains a {extra['suit']} card"
            if parts:
                parts[-1] += condition

        if parts:
            return "; ".join(parts)
        effect = entry.get("effect", "")
        return f"{name}: {effect}" if effect else name

    # Planets
    if card_set == "Planet":
        hand_type = numeric_effects.get("hand_type", "a poker hand")
        return f"Levels up {hand_type}"

    # Tarots
    if card_set == "Tarot":
        if "mod_conv" in numeric_effects:
            max_h = numeric_effects.get("max_highlighted", "selected")
            return f"Enhances up to {max_h} selected cards"
        effect = entry.get("effect", "")
        return f"{name}: {effect}" if effect else name

    # Spectrals
    if card_set == "Spectral":
        effect = entry.get("effect", "")
        return f"{name}: {effect}" if effect else name

    # Vouchers
    if card_set == "Voucher":
        effect = entry.get("effect", "")
        return f"{name}: {effect}" if effect else name

    return name


def process_center(key, entry):
    """Process a single centers.json entry into the shared schema."""
    card_set = entry.get("set", "")
    if card_set not in SET_TYPE_MAP:
        return None

    card_type, card_subtype = SET_TYPE_MAP[card_set]
    config = entry.get("config", {})
    numeric_effects = extract_numeric_effects(config if not isinstance(config, list) else {})

    rarity_num = entry.get("rarity")
    rarity = RARITY_MAP.get(rarity_num, "common")

    return {
        "id": f"balatro:{key}",
        "game": "balatro",
        "name": entry["name"],
        "card_type": card_type,
        "card_subtype": card_subtype,
        "cost": entry.get("cost", 0),
        "rarity": rarity,
        "numeric_effects": numeric_effects,
        "keywords": extract_keywords(entry),
        "rules_text": build_rules_text(entry, numeric_effects),
        "raw": entry,
    }


def process_playing_card(key, entry):
    """Process a playing card from cards.json."""
    return {
        "id": f"balatro:{key}",
        "game": "balatro",
        "name": entry["name"],
        "card_type": "playing_card",
        "card_subtype": "playing_card",
        "cost": 0,
        "rarity": "basic",
        "numeric_effects": {
            "suit": entry.get("suit", ""),
            "value": entry.get("value", ""),
        },
        "keywords": [],
        "rules_text": entry["name"],
        "raw": entry,
    }


def main():
    # Process centers.json
    centers_path = BALATRO_DATA / "centers.json"
    with open(centers_path, "r") as f:
        centers = json.load(f)

    cards = []
    prefixes = ("j_", "c_", "v_")
    for key, entry in centers.items():
        if not any(key.startswith(p) for p in prefixes):
            continue
        card = process_center(key, entry)
        if card:
            cards.append(card)

    # Process cards.json (playing cards)
    cards_path = BALATRO_DATA / "cards.json"
    if cards_path.exists():
        with open(cards_path, "r") as f:
            playing_cards = json.load(f)
        for key, entry in playing_cards.items():
            if "suit" in entry and "value" in entry:
                cards.append(process_playing_card(key, entry))

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w") as f:
        json.dump(cards, f, indent=2)

    # Summary
    by_subtype = {}
    for c in cards:
        st = c["card_subtype"]
        by_subtype[st] = by_subtype.get(st, 0) + 1
    print(f"Extracted {len(cards)} Balatro cards:")
    for st, count in sorted(by_subtype.items()):
        print(f"  {st}: {count}")
    print(f"Output: {OUTPUT}")


if __name__ == "__main__":
    main()
