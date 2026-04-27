#!/usr/bin/env bash
# Convenience launcher: open the viewer on a sensible default mesh.
#
# Usage:
#   ./run.sh                       # opens c10 placeholder (bakes if missing)
#   ./run.sh proto                 # opens the synthetic proto mesh
#   ./run.sh path/to/foo.hmesh     # opens an arbitrary .hmesh
#   ./run.sh --self-test           # forwarded to viewer (writes a PNG, exits)
#
# Anything after the first positional arg (or any --flag) is forwarded to
# `python -m app.viewer` verbatim.

set -euo pipefail
cd "$(dirname "$0")"

if [[ ! -d env ]]; then
    echo "error: env/ not found. bootstrap with:" >&2
    echo "    python3 -m venv env && source env/bin/activate && pip install -e ." >&2
    exit 1
fi

# shellcheck disable=SC1091
source env/bin/activate

# Build / refresh the Cython extension if the source is newer than the
# compiled .so. `pip install -e .` is idempotent and fast on a no-op.
ext_so=$(find bindings -name 'human*.so' -print -quit 2>/dev/null || true)
if [[ -z "$ext_so" ]] || [[ bindings/human.pyx -nt "$ext_so" ]] \
        || [[ core/human.c -nt "$ext_so" ]] || [[ core/human.h -nt "$ext_so" ]]; then
    echo "[run.sh] (re)building extension..."
    pip install -e . --quiet
fi

# Default mesh: bake c10.glb on first run so there is always something to show.
# Inject the default if no positional (non-flag) argument was provided — flags
# alone like `./run.sh --self-test` should still get the default mesh.
default_mesh="/tmp/c10.hmesh"
# If no args at all, or the first arg starts with `-` (i.e. only flags
# were passed), the user did not specify an input mesh -- inject the default.
# Don't try to walk the full arg list because some flags take values that
# look like positionals (e.g. `--out /tmp/foo.png`).
if [[ $# -eq 0 || "${1}" == -* ]]; then
    if [[ ! -f "$default_mesh" ]]; then
        echo "[run.sh] baking placeholder $default_mesh from screenshots/triview/c10.glb..."
        python -m tools.basemesh.bake_hmesh \
            --in screenshots/triview/c10.glb \
            --out "$default_mesh"
    fi
    set -- "$default_mesh" "$@"
fi

exec python -m app.viewer "$@"
