"""Archetype catalog. IDs match human_archetype_t in core/human.h."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Archetype:
    id: int
    slug: str
    age: str
    sex: str
    target_height_m: float
    target_tris: int


CATALOG = (
    Archetype(0, "child_f", "child", "F", 1.30, 30000),
    Archetype(1, "child_m", "child", "M", 1.32, 30000),
    Archetype(2, "adult_f", "adult", "F", 1.65, 40000),
    Archetype(3, "adult_m", "adult", "M", 1.78, 40000),
    Archetype(4, "old_f",   "old",   "F", 1.60, 40000),
    Archetype(5, "old_m",   "old",   "M", 1.72, 40000),
)


def by_slug(slug: str) -> Archetype:
    for a in CATALOG:
        if a.slug == slug:
            return a
    raise KeyError(slug)
