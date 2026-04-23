#!/usr/bin/env bash
# Start the procedural human generator playing a BVH walk cycle.
set -euo pipefail
cd "$(dirname "$0")"
exec env/bin/python human_generator.py --bvh animations/walk2.bvh "$@"
