"""Fetch optional BVH animation packs into ./animations/.

The viewer auto-discovers any ``*.bvh`` under ``animations/`` at startup
(``human/viewer.py`` uses ``rglob``), so downloaded packs appear in the
animation cycler with no code changes.

Packs are NOT committed to git (they are several hundred MB and have
attribution-required licenses — see each pack's SOURCE_README.md after
download). Run this script on demand:

    python tools/download_animations.py --all
    python tools/download_animations.py --bandai2
    python tools/download_animations.py --100style

Missing or failed packs are reported; the script keeps going and the
viewer tolerates an empty ``animations/`` directory.
"""

from __future__ import annotations

import argparse
import io
import shutil
import sys
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ANIM_DIR = ROOT / "animations"


def _download(url: str, label: str) -> bytes | None:
    print(f"[{label}] downloading {url}")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "human-mocap-fetch/1.0"})
        with urllib.request.urlopen(req, timeout=120) as resp:
            total = resp.headers.get("Content-Length")
            total_mb = f"{int(total) / 1e6:.1f} MB" if total else "unknown size"
            print(f"[{label}] {total_mb}, reading ...")
            return resp.read()
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        print(f"[{label}] download failed: {e}")
        return None


def _extract_zip(data: bytes, dest: Path, filter_prefix: str | None, label: str) -> int:
    """Extract ``*.bvh`` files from ``data`` into ``dest``.

    If ``filter_prefix`` is given, only entries whose archive path contains
    that substring are kept. Returns the number of BVH files written.
    """
    dest.mkdir(parents=True, exist_ok=True)
    count = 0
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            for info in zf.infolist():
                if info.is_dir():
                    continue
                name = info.filename
                if not name.lower().endswith(".bvh"):
                    continue
                if filter_prefix and filter_prefix not in name:
                    continue
                target = dest / Path(name).name
                with zf.open(info) as src, open(target, "wb") as out:
                    shutil.copyfileobj(src, out)
                count += 1
    except zipfile.BadZipFile as e:
        print(f"[{label}] bad zip: {e}")
        return 0
    return count


def fetch_bandai2(force: bool) -> None:
    label = "bandai2"
    dest = ANIM_DIR / "bandai2"
    if dest.exists() and any(dest.glob("*.bvh")) and not force:
        print(f"[{label}] already present at {dest} (use --force to re-download)")
        return
    url = (
        "https://github.com/BandaiNamcoResearchInc/"
        "Bandai-Namco-Research-Motiondataset/archive/refs/heads/master.zip"
    )
    data = _download(url, label)
    if data is None:
        return
    n = _extract_zip(
        data,
        dest,
        filter_prefix="Bandai-Namco-Research-Motiondataset-2/",
        label=label,
    )
    (dest / "SOURCE_README.md").write_text(
        "# Bandai-Namco-Research-Motiondataset-2\n\n"
        "Source: https://github.com/BandaiNamcoResearchInc/"
        "Bandai-Namco-Research-Motiondataset\n\n"
        "License: CC BY-NC 4.0 (non-commercial, attribution required).\n"
        "Citation: Kobayashi et al., arXiv:2306.08861 (2023).\n",
        encoding="utf-8",
    )
    print(f"[{label}] wrote {n} BVH files to {dest}")


def fetch_100style(force: bool) -> None:
    label = "100style"
    dest = ANIM_DIR / "100style"
    if dest.exists() and any(dest.glob("*.bvh")) and not force:
        print(f"[{label}] already present at {dest} (use --force to re-download)")
        return
    # orangeduck's retargeted BVH zip (compatible common skeleton).
    url = "https://theorangeduck.com/media/uploads/Geno/100style-retarget/bvh.zip"
    data = _download(url, label)
    if data is None:
        return
    n = _extract_zip(data, dest, filter_prefix=None, label=label)
    (dest / "SOURCE_README.md").write_text(
        "# 100STYLE (retargeted)\n\n"
        "Source: https://github.com/orangeduck/100style-retarget\n"
        "Original: https://www.ianxmason.com/100style/\n\n"
        "License: CC BY 4.0 (attribution required).\n"
        "Citation: Mason et al., 'Real-Time Style Modelling of Human Locomotion', "
        "PACMCGIT 2022.\n\n"
        "Note: animations are missing finger motion (known limitation of the "
        "retargeted release).\n",
        encoding="utf-8",
    )
    print(f"[{label}] wrote {n} BVH files to {dest}")


def note_cmu() -> None:
    """CMU CGSpeed is hosted on MediaFire across many zips and isn't reliably
    scriptable. Leave a placeholder with instructions."""
    dest = ANIM_DIR / "cmu"
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "SOURCE_README.md").write_text(
        "# CMU CGSpeed BVH (manual download)\n\n"
        "The cgspeed MotionBuilder-friendly release is distributed as many\n"
        "MediaFire zips that change over time; automated download is fragile.\n\n"
        "Download from:\n"
        "https://sites.google.com/a/cgspeed.com/cgspeed/motion-capture/"
        "the-motionbuilder-friendly-bvh-conversion-release-of-cmus-motion-capture-database\n\n"
        "Extract any ``*.bvh`` files into this directory — the viewer will\n"
        "discover them automatically.\n\n"
        "License: CMU free-for-any-use (see their database page for terms).\n",
        encoding="utf-8",
    )
    print(f"[cmu] wrote placeholder README at {dest}")


PACKS = {
    "bandai2": fetch_bandai2,
    "100style": fetch_100style,
}


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bandai2", action="store_true", help="Bandai-Namco dataset-2 (~200 MB, CC BY-NC 4.0)")
    p.add_argument("--100style", dest="hundred_style", action="store_true",
                   help="100STYLE retargeted (~350 MB, CC BY 4.0)")
    p.add_argument("--cmu", action="store_true",
                   help="CMU CGSpeed (writes instructions only — manual download)")
    p.add_argument("--all", action="store_true", help="All automatable packs")
    p.add_argument("--force", action="store_true", help="Re-download even if already present")
    args = p.parse_args(argv)

    wanted = {
        "bandai2": args.bandai2 or args.all,
        "100style": args.hundred_style or args.all,
        "cmu": args.cmu or args.all,
    }
    if not any(wanted.values()):
        p.print_help()
        return 0

    ANIM_DIR.mkdir(parents=True, exist_ok=True)
    if wanted["bandai2"]:
        fetch_bandai2(args.force)
    if wanted["100style"]:
        fetch_100style(args.force)
    if wanted["cmu"]:
        note_cmu()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
