"""human v2 CLI.

Subcommands:
  proto        write a synthetic .hmesh to a path (engine smoke test)
  render       render a .hmesh to a PNG with optional --param NAME=VALUE
  triview      render front/side/iso into a 3-up grid
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from bindings import Human


def _parse_params(items: list[str]) -> list[tuple[str, float]]:
    out = []
    for it in items or []:
        if "=" not in it:
            raise SystemExit(f"bad --param '{it}', expected NAME=VALUE")
        k, v = it.split("=", 1)
        out.append((k.strip(), float(v)))
    return out


def _load(args) -> Human:
    if args.input == "proto":
        return Human.proto()
    return Human.load(args.input)


def _apply_params(h: Human, params: list[tuple[str, float]]) -> None:
    for k, v in params:
        h.set_param(k, v)
    if params:
        h.evaluate()


def cmd_proto(args) -> int:
    h = Human.proto()
    h.save(args.out)
    print(f"wrote {args.out}: {h.vertex_count} verts, {h.bone_count} bones, {len(h.params)} params")
    return 0


def cmd_render(args) -> int:
    from app.render import render

    h = _load(args)
    _apply_params(h, _parse_params(args.param))
    img = render(h, view=args.view, size=(args.width, args.height))
    img.save(args.out)
    print(f"wrote {args.out}  ({args.width}x{args.height}, view={args.view})")
    return 0


def cmd_triview(args) -> int:
    from PIL import Image

    from app.render import render

    h = _load(args)
    _apply_params(h, _parse_params(args.param))
    w, hh = args.width, args.height
    panels = [render(h, view=v, size=(w, hh)) for v in ("front", "side", "iso")]
    grid = Image.new("RGB", (w * 3, hh))
    for i, p in enumerate(panels):
        grid.paste(p, (i * w, 0))
    grid.save(args.out)
    print(f"wrote {args.out}  (3x {w}x{hh})")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="human")
    sub = p.add_subparsers(dest="cmd", required=True)

    pp = sub.add_parser("proto", help="write a synthetic .hmesh")
    pp.add_argument("out", type=Path)
    pp.set_defaults(func=cmd_proto)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("input", help="path to .hmesh, or 'proto' for the synthetic model")
    common.add_argument("--param", "-p", action="append", default=[], metavar="NAME=VALUE")

    pr = sub.add_parser("render", parents=[common])
    pr.add_argument("--view", choices=("front", "side", "iso"), default="front")
    pr.add_argument("--width", type=int, default=512)
    pr.add_argument("--height", type=int, default=768)
    pr.add_argument("--out", "-o", type=Path, required=True)
    pr.set_defaults(func=cmd_render)

    pt = sub.add_parser("triview", parents=[common])
    pt.add_argument("--width", type=int, default=384)
    pt.add_argument("--height", type=int, default=576)
    pt.add_argument("--out", "-o", type=Path, required=True)
    pt.set_defaults(func=cmd_triview)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
