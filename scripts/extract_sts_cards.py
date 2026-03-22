"""Extract card data from Slay the Spire's game.py via AST introspection."""

import ast
import json
import re
from pathlib import Path

STS_GAME = Path(__file__).resolve().parent.parent.parent / "decapitate-the-spire" / "decapitate_the_spire" / "game.py"
OUTPUT = Path(__file__).resolve().parent.parent / "data" / "sts_cards.json"

# Maps from AST enum references to string values
CARD_TYPE_MAP = {
    "ATTACK": "attack",
    "SKILL": "skill",
    "POWER": "power",
    "STATUS": "status",
    "CURSE": "curse",
}

CARD_TARGET_MAP = {
    "ENEMY": "targets_enemy",
    "ALL_ENEMY": "targets_all_enemy",
    "SELF": "targets_self",
    "NONE": "targets_none",
    "SELF_AND_ENEMY": "targets_self_and_enemy",
    "ALL": "targets_all",
}

RARITY_MAP = {
    "BASIC": "basic",
    "SPECIAL": "special",
    "COMMON": "common",
    "UNCOMMON": "uncommon",
    "RARE": "rare",
    "CURSE": "curse",
}

COLOR_MAP = {
    "RED": "red",
    "GREEN": "green",
    "BLUE": "blue",
    "PURPLE": "purple",
    "COLORLESS": "colorless",
    "CURSE": "curse",
}

# Class attributes we want to read
CLASS_ATTRS = [
    "base_damage_master",
    "base_block_master",
    "base_magic_number_master",
    "damage_upgrade_amount",
    "block_upgrade_amount",
    "magic_number_upgrade_amount",
    "upgraded_base_cost",
    "rarity",
    "color",
]

# Boolean keyword args in super().__init__()
BOOL_KWARGS = ["exhaust", "is_innate", "is_multi_damage", "self_retain", "is_ethereal"]


def resolve_enum_attr(node):
    """Resolve CardType.ATTACK style references to the enum member name."""
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def resolve_constant(node):
    """Get a constant value from an AST node."""
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub) and isinstance(node.operand, ast.Constant):
        return -node.operand.value
    return None


def extract_super_init_args(init_node):
    """Parse super().__init__(ctx, cost, card_type, card_target, ...) call."""
    result = {"cost": 0, "card_type": None, "card_target": None}
    for kw in BOOL_KWARGS:
        result[kw] = False

    # Find the super().__init__() call
    for node in ast.walk(init_node):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        # Match super().__init__ or super(ClassName, self).__init__
        if isinstance(func, ast.Attribute) and func.attr == "__init__":
            inner = func.value
            if isinstance(inner, ast.Call) and isinstance(inner.func, ast.Name) and inner.func.id == "super":
                # Positional args: ctx, cost, card_type, card_target
                args = node.args
                if len(args) >= 4:
                    cost_val = resolve_constant(args[1])
                    if cost_val is not None:
                        result["cost"] = cost_val

                    ct = resolve_enum_attr(args[2])
                    if ct:
                        result["card_type"] = ct

                    tgt = resolve_enum_attr(args[3])
                    if tgt:
                        result["card_target"] = tgt

                # Keyword args
                for kw in node.keywords:
                    if kw.arg in BOOL_KWARGS:
                        val = resolve_constant(kw.value)
                        if val is None:
                            # Assume True for NameConstant-style True
                            val = True
                        result[kw.arg] = val

                break

    return result


def extract_instance_assignments(init_node):
    """Extract self.base_damage = X and self.base_magic_number = X set in __init__ body."""
    assignments = {}
    for node in ast.walk(init_node):
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name) and target.value.id == "self":
                attr_name = target.attr
                if attr_name in ("base_damage", "base_block", "base_magic_number"):
                    val = resolve_constant(node.value)
                    if val is not None:
                        assignments[attr_name] = val
    return assignments


def camel_to_snake(name):
    """Convert CamelCase to snake_case."""
    s1 = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s1).lower()


def determine_shared_card_type(card_type_str, has_block):
    """Map StS CardType to shared card_type taxonomy."""
    if card_type_str == "ATTACK":
        return "attack"
    if card_type_str == "SKILL":
        return "defense" if has_block else "utility"
    if card_type_str == "POWER":
        return "power"
    if card_type_str == "STATUS":
        return "status"
    if card_type_str == "CURSE":
        return "curse"
    return "utility"


def build_rules_text(name, card_type_str, numeric_effects, keywords):
    """Build simple templated rules_text."""
    parts = []

    # Prefix keywords
    for kw in ("innate", "ethereal", "retain"):
        if kw in keywords:
            parts.append(kw.capitalize() + ".")

    damage = numeric_effects.get("damage")
    block = numeric_effects.get("block")
    magic = numeric_effects.get("magic_number")

    if card_type_str == "ATTACK" and damage is not None:
        parts.append(f"Deal {damage} damage.")
    elif block is not None:
        parts.append(f"Gain {block} Block.")

    if magic is not None and card_type_str == "POWER":
        # Infer effect from card name
        parts.append(f"Gain {magic} {name} effect.")

    if "exhaust" in keywords:
        parts.append("Exhaust.")

    if parts:
        return " ".join(parts)

    # Fallback
    type_label = CARD_TYPE_MAP.get(card_type_str, card_type_str)
    return f"{name} ({type_label})."


def main():
    source = STS_GAME.read_text(encoding="utf-8")
    tree = ast.parse(source)

    # Find the Card base class node
    card_base_node = None
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "Card":
            card_base_node = node
            break

    if not card_base_node:
        print("ERROR: Could not find Card class")
        return

    card_base_lineno = card_base_node.lineno

    # Find all Card subclasses
    card_classes = []
    for node in ast.iter_child_nodes(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        for base in node.bases:
            base_name = None
            if isinstance(base, ast.Name):
                base_name = base.id
            if base_name == "Card":
                card_classes.append(node)

    cards = []
    for cls_node in card_classes:
        name = cls_node.name

        # Extract class-level attributes
        class_attrs = {}
        for item in cls_node.body:
            if isinstance(item, ast.Assign):
                for target in item.targets:
                    if isinstance(target, ast.Name) and target.id in CLASS_ATTRS:
                        val = resolve_constant(item.value)
                        if val is not None:
                            class_attrs[target.id] = val
                        else:
                            # Could be an enum like CardRarity.BASIC
                            enum_val = resolve_enum_attr(item.value)
                            if enum_val:
                                class_attrs[target.id] = enum_val
            elif isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                if item.target.id in CLASS_ATTRS and item.value:
                    val = resolve_constant(item.value)
                    if val is not None:
                        class_attrs[item.target.id] = val
                    else:
                        enum_val = resolve_enum_attr(item.value)
                        if enum_val:
                            class_attrs[item.target.id] = enum_val

        # Find __init__ method
        init_node = None
        for item in cls_node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == "__init__":
                init_node = item
                break

        if not init_node:
            continue

        super_args = extract_super_init_args(init_node)
        instance_assignments = extract_instance_assignments(init_node)

        # Determine values
        card_type_str = super_args["card_type"]
        card_target_str = super_args["card_target"]
        cost = super_args["cost"]

        # Numeric effects - prefer class attrs, fall back to instance assignments
        base_damage = class_attrs.get("base_damage_master") or instance_assignments.get("base_damage")
        base_block = class_attrs.get("base_block_master") or instance_assignments.get("base_block")
        base_magic = class_attrs.get("base_magic_number_master") or instance_assignments.get("base_magic_number")

        numeric_effects = {}
        if base_damage is not None:
            numeric_effects["damage"] = base_damage
        if base_block is not None:
            numeric_effects["block"] = base_block
        if base_magic is not None:
            numeric_effects["magic_number"] = base_magic
        if class_attrs.get("damage_upgrade_amount") is not None:
            numeric_effects["damage_upgrade"] = class_attrs["damage_upgrade_amount"]
        if class_attrs.get("block_upgrade_amount") is not None:
            numeric_effects["block_upgrade"] = class_attrs["block_upgrade_amount"]
        if class_attrs.get("magic_number_upgrade_amount") is not None:
            numeric_effects["magic_number_upgrade"] = class_attrs["magic_number_upgrade_amount"]

        # Keywords
        keywords = []
        if super_args.get("exhaust"):
            keywords.append("exhaust")
        if super_args.get("is_innate"):
            keywords.append("innate")
        if super_args.get("is_ethereal"):
            keywords.append("ethereal")
        if super_args.get("self_retain"):
            keywords.append("retain")
        if super_args.get("is_multi_damage"):
            keywords.append("multi_target")
        if card_target_str:
            keywords.append(CARD_TARGET_MAP.get(card_target_str, f"targets_{card_target_str.lower()}"))

        # Rarity and color
        rarity_str = class_attrs.get("rarity", "COMMON")
        color_str = class_attrs.get("color", "COLORLESS")
        rarity = RARITY_MAP.get(rarity_str, "common")
        color = COLOR_MAP.get(color_str, "colorless")

        # Shared card_type
        has_block = base_block is not None
        shared_card_type = determine_shared_card_type(card_type_str, has_block)
        sts_subtype = CARD_TYPE_MAP.get(card_type_str, "skill")

        rules_text = build_rules_text(name, card_type_str, numeric_effects, keywords)

        card_id = f"sts:{camel_to_snake(name)}"

        raw = {
            "class_name": name,
            "card_type": card_type_str,
            "card_target": card_target_str,
            "color": color_str,
            "rarity": rarity_str,
            "cost": cost,
            "class_attrs": {k: v for k, v in class_attrs.items() if k not in ("rarity", "color")},
            "init_flags": {k: super_args[k] for k in BOOL_KWARGS if super_args.get(k)},
            "instance_overrides": instance_assignments,
        }

        cards.append({
            "id": card_id,
            "game": "sts",
            "name": name,
            "card_type": shared_card_type,
            "card_subtype": sts_subtype,
            "cost": cost,
            "rarity": rarity,
            "numeric_effects": numeric_effects,
            "keywords": keywords,
            "rules_text": rules_text,
            "raw": raw,
        })

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w") as f:
        json.dump(cards, f, indent=2)

    # Summary
    by_type = {}
    for c in cards:
        t = c["card_subtype"]
        by_type[t] = by_type.get(t, 0) + 1
    print(f"Extracted {len(cards)} StS cards:")
    for t, count in sorted(by_type.items()):
        print(f"  {t}: {count}")
    print(f"Output: {OUTPUT}")


if __name__ == "__main__":
    main()
