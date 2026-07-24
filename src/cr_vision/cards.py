from __future__ import annotations

CARD_COSTS: dict[str, int] = {
    "archers": 3,
    "arrows": 3,
    "baby_dragon": 4,
    "balloon": 5,
    "bats": 2,
    "cannon": 3,
    "fireball": 4,
    "goblin_barrel": 3,
    "hog_rider": 4,
    "knight": 3,
    "mini_pekka": 4,
    "musketeer": 4,
    "skeletons": 1,
    "the_log": 2,
    "valkyrie": 4,
    "zap": 2,
}


def card_cost(card_name: str) -> int:
    try:
        return CARD_COSTS[card_name]
    except KeyError as exc:
        known = ", ".join(sorted(CARD_COSTS))
        raise ValueError(f"Unknown card '{card_name}'. Known cards: {known}") from exc

