"""Stage 1 — generate the 12 reference images via Nano Banana Pro.

Per archetype: one front T-pose and one back T-pose. Anatomical reference
style, T-pose with horizontal arms, nude, bald, neutral expression, neutral
lighting, white background. Outputs go to assets/basemesh/reference/<slug>_{front,back}.png.

The Nano Banana Pro call is intentionally not made from this script — the
operator runs the `/nano-banana-pro` skill in Claude with each PROMPT below
and saves the result to the indicated path. This keeps API spend explicit
and human-supervised. Once images exist, run `gen_mesh.py`.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from .archetypes import CATALOG, Archetype


def prompt(a: Archetype, view: str) -> str:
    sex_word = {"F": "female", "M": "male"}[a.sex]
    age_clause = {
        "child":  "approximately 8 years old, child proportions",
        "adult":  "approximately 30 years old",
        "old":    "approximately 70 years old, age-appropriate skin texture and proportions",
    }[a.age]
    return (
        f"Anatomical reference photograph of a nude bald {sex_word}, {age_clause}, "
        f"{view} view, T-pose with arms held perfectly horizontal at shoulder height, "
        f"feet shoulder-width apart, neutral facial expression, no hair, no facial hair, "
        f"no jewelry, no clothing, no accessories. "
        f"Even diffuse studio lighting, plain white seamless background, "
        f"sharp focus from head to feet, full body visible, "
        f"medical anatomy reference style. Photorealistic, neutral skin tone."
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, default=Path("assets/basemesh/reference"))
    ap.add_argument("--archetype", help="run only this slug (default: all)")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    targets = [a for a in CATALOG if not args.archetype or a.slug == args.archetype]

    print(f"# {len(targets) * 2} reference renders to produce")
    for a in targets:
        for view in ("front", "back"):
            path = args.out_dir / f"{a.slug}_{view}.png"
            print()
            print(f"## {a.slug} / {view}  ->  {path}")
            print(f"PROMPT: {prompt(a, view)}")
    print()
    print("Run `/nano-banana-pro` in Claude with each prompt, save to the listed path.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
