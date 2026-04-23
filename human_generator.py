"""Procedural 3D human generator: skeleton + skin + face + hair + clothing + walk.

Inspired by:
- SMPL / SMPL-X        - parametric body (gendered); blend skinning
- FLAME                - parametric 3D face (eyes, jaw, expressions)
- MakeHuman            - procedural humanoid mesh pipeline
- SKEL                 - skin-to-skeleton biomechanical re-rig of SMPL
- CAPE                 - clothing as displacement over SMPL
- GarmentCode          - programmatic parametric sewing patterns
- Reallusion Smart Hair / hair-card systems
- Johansen (2009) thesis, "Bipedal Walk Cycles" (ASEE 2009) - procedural locomotion

Pipeline:
    shape (gender-aware)  ->  skeleton.apply_shape
    mesh.build            ->  body + compound head + bust + glutes
    accessories           ->  eyes/iris/hair/eyebrows/facial hair
    garments              ->  top/bottom/dress/skirt/shoes (capsule + cone shells)
    skeleton.walk_pose(t) ->  bone matrices  ->  LBS in vertex shader
    procedural skin/fabric/hair shading in fragment shader

Controls:
    drag / scroll     - orbit / zoom
    SPACE / R         - regenerate everything (random gender)
    M / F             - regenerate forcing male / female
    S                 - new random shape (keep gender)
    C                 - new random clothes/hair/colors (keep shape)
    W                 - toggle walk animation
    [ / ]             - slower / faster walk
    P                 - random static pose        T - T-pose
    B                 - toggle skeleton overlay
    Esc / Q           - quit
"""

import argparse

from human.viewer import Viewer


def main():
    print(__doc__)
    ap = argparse.ArgumentParser()
    ap.add_argument("--bvh", default=None,
                    help="Path to a BVH animation file to play on startup")
    args = ap.parse_args()
    Viewer(bvh_path=args.bvh).run()


if __name__ == "__main__":
    main()
