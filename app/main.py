from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    print("human v2 — C core + Cython + Python app")
    print("scaffold ready. next step: implement .hmesh I/O in core/io.c")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
