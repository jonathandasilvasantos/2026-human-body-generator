"""Stage 1 — generate the 12 reference images via a self-hosted text-to-image
model on the `bender` GPU host. No paid APIs.

Per archetype: one front T-pose and one back T-pose. Anatomical reference
style, T-pose with horizontal arms, nude, bald, neutral expression, neutral
lighting, white background. Outputs go to
`assets/basemesh/reference/<slug>_{front,back}.png`.

The remote model is configurable. Default assumes FLUX.1 dev installed at
`~/human/flux-env` on bender; override via --remote-cmd. The remote command
must accept `--prompt PROMPT --out PATH` and write a PNG.

Usage:
  python -m tools.basemesh.gen_reference                        # all archetypes
  python -m tools.basemesh.gen_reference --archetype adult_m    # one
  python -m tools.basemesh.gen_reference --print                # prompts only
"""
from __future__ import annotations

import argparse
import shlex
import subprocess
from pathlib import Path

from .archetypes import CATALOG, Archetype

BENDER_HOST = "bender"
DEFAULT_REMOTE_CMD = (
    "~/human/flux-env/bin/python -m human_t2i.cli "
    "--steps 28 --guidance 3.5 --width 768 --height 1280"
)


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


def run_one(a: Archetype, view: str, out_dir: Path, remote_cmd: str, host: str) -> int:
    out_path = out_dir / f"{a.slug}_{view}.png"
    remote_out = f"/tmp/human_ref_{a.slug}_{view}.png"
    pr = prompt(a, view)
    out_dir.mkdir(parents=True, exist_ok=True)

    full_remote = f"{remote_cmd} --prompt {shlex.quote(pr)} --out {shlex.quote(remote_out)}"
    cmds = [
        ["ssh", host, full_remote],
        ["scp", f"{host}:{remote_out}", str(out_path)],
    ]
    for c in cmds:
        print("$", " ".join(shlex.quote(x) for x in c))
        rc = subprocess.call(c)
        if rc != 0:
            print(f"FAIL {a.slug}/{view}: rc={rc}")
            return rc
    print(f"OK {a.slug}/{view} -> {out_path}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, default=Path("assets/basemesh/reference"))
    ap.add_argument("--archetype", help="run only this slug (default: all)")
    ap.add_argument("--host", default=BENDER_HOST, help="SSH host alias")
    ap.add_argument("--remote-cmd", default=DEFAULT_REMOTE_CMD,
                    help="remote command, will receive --prompt PROMPT --out PATH")
    ap.add_argument("--print", dest="print_only", action="store_true",
                    help="print prompts and remote commands without running")
    args = ap.parse_args()

    targets = [a for a in CATALOG if not args.archetype or a.slug == args.archetype]
    fails = 0
    for a in targets:
        for view in ("front", "back"):
            if args.print_only:
                pr = prompt(a, view)
                out_path = args.out_dir / f"{a.slug}_{view}.png"
                print(f"\n## {a.slug} / {view}  ->  {out_path}")
                print(f"PROMPT: {pr}")
                continue
            rc = run_one(a, view, args.out_dir, args.remote_cmd, args.host)
            fails += (rc != 0)
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
