"""WAV-driven lip sync.

Takes a mono/stereo PCM WAV file and produces a time-indexed stream of
ARKit-52 blendshape weights suitable for feeding the procedural face
rig in ``human/face_anim.py``.

The design is deliberately phoneme-recogniser-free: no CMUSphinx, no
forced alignment, no neural model. Two signals drive the mouth:

1. **Short-time energy** (RMS) -> jaw open / lip separation amount.
2. **Band-energy ratios + spectral centroid** -> pick one of the
   Preston-Blair visemes from ``face_anim.VISEMES``.

This is the same family of techniques used by Oculus LipSync's
"simple" mode, Annosoft's energy+formant fallback, and Rhubarb's
dumb analyser. It is not as precise as a phoneme recogniser but it
runs in a few ms on a long clip, has no external dependencies, and
produces visibly-plausible speech motion for arbitrary audio.

References
----------
- Lewis (1991), "Automated lip-sync: background and techniques"
- Ezzat, Geiger, Poggio (2002), "Trainable videorealistic speech animation"
- Edwards, Landreth, Fiume, Singh (2016), "JALI: An Animator-Centric
  Viseme Model"  -- inspired the jaw-scales-with-energy rule.
- Cohen & Massaro (1993), "Modeling coarticulation in synthetic visual
  speech" -- dominance blending (implemented in face_anim.VisemeTrack).
- Blair (1946), "Advanced Animation" -- 12-viseme reduction.
"""

from __future__ import annotations

import math
import wave
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from . import face_anim


# Analysis parameters. 25ms window / 10ms hop is the classic speech-
# processing front-end (Davis & Mermelstein 1980).
WIN_MS = 25.0
HOP_MS = 10.0
# Silence floor (dBFS). Anything quieter collapses to the "sil" viseme.
SIL_DBFS = -42.0
# Fricative decision: fraction of spectral energy above 3 kHz.
HF_RATIO_FRIC = 0.38


# --- WAV loading ------------------------------------------------------------

def load_wav(path: str) -> Tuple[np.ndarray, int]:
    """Return ``(mono_float32 in [-1,1], sample_rate)``."""
    with wave.open(path, "rb") as w:
        sr = w.getframerate()
        n = w.getnframes()
        ch = w.getnchannels()
        sw = w.getsampwidth()
        raw = w.readframes(n)
    if sw == 2:
        data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sw == 4:
        data = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    elif sw == 1:
        data = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    else:
        raise ValueError(f"unsupported sample width: {sw} bytes")
    if ch > 1:
        data = data.reshape(-1, ch).mean(axis=1)
    return data, sr


# --- frame-level features ---------------------------------------------------

@dataclass
class Frame:
    t: float           # centre time (s)
    rms: float         # linear [0,1]
    dbfs: float        # dB, <=0
    centroid: float    # Hz
    hf_ratio: float    # energy > 3kHz / total
    lf_ratio: float    # energy 0..1kHz / total
    mf_ratio: float    # energy 1..3kHz / total
    zcr: float         # zero crossings / sample


def _frames(samples: np.ndarray, sr: int) -> List[Frame]:
    win = max(2, int(sr * WIN_MS / 1000.0))
    hop = max(1, int(sr * HOP_MS / 1000.0))
    # Hann window reduces spectral leakage.
    window = 0.5 - 0.5 * np.cos(2.0 * math.pi * np.arange(win) / max(1, win - 1))
    freqs = np.fft.rfftfreq(win, d=1.0 / sr)
    lf = freqs < 1000.0
    mf = (freqs >= 1000.0) & (freqs < 3000.0)
    hf = freqs >= 3000.0

    out: List[Frame] = []
    # Pre-emphasis (classic speech front-end) boosts high frequencies so
    # fricatives are not buried under voiced energy.
    pre = np.empty_like(samples)
    pre[0] = samples[0]
    pre[1:] = samples[1:] - 0.97 * samples[:-1]

    n = len(samples)
    i = 0
    while i + win <= n:
        seg = pre[i:i + win]
        # RMS on the *original* signal (pre-emphasis inflates HF RMS).
        raw = samples[i:i + win]
        rms = float(np.sqrt(np.mean(raw * raw) + 1e-12))
        dbfs = 20.0 * math.log10(max(rms, 1e-6))
        spec = np.abs(np.fft.rfft(seg * window)) + 1e-9
        total = float(spec.sum())
        lf_e = float(spec[lf].sum()) / total
        mf_e = float(spec[mf].sum()) / total
        hf_e = float(spec[hf].sum()) / total
        centroid = float((freqs * spec).sum() / total)
        # Zero crossings on the raw segment -- fricatives have high ZCR.
        zc = int(np.sum(np.abs(np.diff(np.signbit(raw).astype(np.int8)))))
        zcr = zc / float(len(raw))
        out.append(Frame(
            t=(i + win * 0.5) / sr,
            rms=rms, dbfs=dbfs,
            centroid=centroid,
            hf_ratio=hf_e, lf_ratio=lf_e, mf_ratio=mf_e,
            zcr=zcr,
        ))
        i += hop
    return out


# --- viseme classification --------------------------------------------------

def _classify(f: Frame) -> str:
    """Map a single analysis frame to one of the Preston-Blair visemes.

    Decision tree tuned for English speech at conversational loudness.
    Coarticulation is left to ``face_anim.VisemeTrack`` (Cohen-Massaro).
    """
    if f.dbfs < SIL_DBFS:
        return "sil"

    # Unvoiced / fricative family: lots of HF energy, high ZCR, low-ish RMS.
    if f.hf_ratio > HF_RATIO_FRIC and f.zcr > 0.12:
        # Sibilants (/s z ʃ ʒ tʃ dʒ/) sit high; /f v θ ð/ sit lower.
        if f.centroid > 4200.0:
            return "S"
        if f.centroid > 2800.0:
            return "TH"
        return "FV"

    # Voiced region: pick a vowel by where the spectral mass sits.
    # Acoustic phonetics (Peterson & Barney 1952): low F1 = close vowel;
    # high F2 = front vowel. We approximate F1 by LF ratio and F2 by
    # centroid, which is crude but enough to separate the four corners.
    if f.lf_ratio > 0.78 and f.centroid < 900.0:
        # Very low spectrum: rounded close back vowel.
        return "U" if f.dbfs < -22.0 else "O"
    if f.lf_ratio > 0.62 and f.centroid < 1400.0:
        # Mostly low with some mid: open/near-open back.
        return "O" if f.dbfs > -25.0 else "AI"
    if f.mf_ratio > 0.28 and f.centroid > 1600.0:
        # Strong mid / high centroid: front vowel.
        return "E"
    # Default open vowel -- carries most of the energy of casual speech.
    return "AI"


def _compact(frames: List[Frame], labels: List[str],
             min_dur: float = 0.045) -> List[face_anim.VisemeSegment]:
    """Collapse consecutive equal-labelled frames into one segment.

    Segments shorter than ``min_dur`` are merged into their longer
    neighbour; this removes single-frame flicker without smoothing away
    real plosives.
    """
    if not frames:
        return []
    # First pass: runs.
    runs: List[Tuple[str, float, float]] = []  # (label, t0, t1)
    cur_lab = labels[0]
    cur_t0 = frames[0].t
    cur_t1 = frames[0].t
    for fr, lab in zip(frames[1:], labels[1:]):
        if lab == cur_lab:
            cur_t1 = fr.t
        else:
            runs.append((cur_lab, cur_t0, cur_t1))
            cur_lab, cur_t0, cur_t1 = lab, fr.t, fr.t
    runs.append((cur_lab, cur_t0, cur_t1))

    # Second pass: merge tiny runs into neighbour.
    merged: List[List[float | str]] = []  # type: ignore[type-arg]
    for lab, t0, t1 in runs:
        dur = t1 - t0
        if merged and dur < min_dur and lab != "sil":
            # Extend previous run rather than keep the flicker.
            merged[-1][2] = t1  # type: ignore[index]
            continue
        merged.append([lab, t0, t1])

    segs: List[face_anim.VisemeSegment] = []
    for lab, t0, t1 in merged:  # type: ignore[misc]
        mid = 0.5 * (t0 + t1)  # type: ignore[operator]
        dur = max(0.06, (t1 - t0))  # type: ignore[operator]
        # Dominance width scales with segment duration; clamped so brief
        # plosives still blend, long vowels don't bleed forever.
        width = min(0.22, max(0.08, dur * 1.2))
        segs.append(face_anim.VisemeSegment(
            time=float(mid), name=str(lab), width=float(width),
        ))
    return segs


# --- public API -------------------------------------------------------------

@dataclass
class LipsyncTrack:
    """Time-indexed ARKit-52 weights from a WAV file.

    ``sample(t)`` returns a ``Dict[str, float]`` compatible with
    ``Appearance.blendshapes``.
    """
    viseme_track: face_anim.VisemeTrack
    energy: np.ndarray       # per-frame RMS, normalised to [0,1]
    frame_times: np.ndarray  # (N,) seconds
    duration: float

    def _energy_at(self, t: float) -> float:
        if len(self.frame_times) == 0:
            return 0.0
        # linear interp
        if t <= self.frame_times[0]:
            return float(self.energy[0])
        if t >= self.frame_times[-1]:
            return float(self.energy[-1])
        idx = np.searchsorted(self.frame_times, t)
        t0, t1 = self.frame_times[idx - 1], self.frame_times[idx]
        e0, e1 = self.energy[idx - 1], self.energy[idx]
        a = (t - t0) / max(1e-6, (t1 - t0))
        return float(e0 + (e1 - e0) * a)

    def sample(self, t: float) -> Dict[str, float]:
        w = self.viseme_track.sample(t)
        # Scale jaw and lip parting with instantaneous loudness. The
        # viseme table gives a *target* shape; real speech modulates the
        # aperture from word to word. JALI (Edwards et al. 2016) argues
        # for independent jaw + lip controls driven by prosody -- this
        # is the minimal version: energy multiplies the open-mouth
        # channels, caps the closed-lip channels.
        e = self._energy_at(t)
        # Smooth low end so silences still close the mouth.
        loud = max(0.0, min(1.0, (e - 0.05) / 0.55))
        jaw = w.get("jawOpen", 0.0)
        # Blend: 55% viseme + 45% loudness-driven, take the max so
        # strong vowels stay open on quiet tails.
        w["jawOpen"] = max(jaw, 0.35 * jaw + 0.65 * loud * max(jaw, 0.25))
        # A tiny bit of lower-lip drop with loudness adds liveliness.
        if loud > 0.15:
            w["mouthLowerDownLeft"] = max(w.get("mouthLowerDownLeft", 0.0),
                                          0.25 * loud)
            w["mouthLowerDownRight"] = max(w.get("mouthLowerDownRight", 0.0),
                                           0.25 * loud)
        # During silence, collapse everything toward rest.
        if e < 0.02:
            for k in list(w.keys()):
                w[k] *= 0.15
        return w


def from_wav(path: str) -> LipsyncTrack:
    """Analyse ``path`` and return a sampler over the full clip."""
    samples, sr = load_wav(path)
    if samples.size == 0:
        raise ValueError(f"empty audio: {path}")
    frames = _frames(samples, sr)
    labels = [_classify(f) for f in frames]
    segs = _compact(frames, labels)
    track = face_anim.VisemeTrack(segs)
    times = np.array([f.t for f in frames], dtype=np.float32)
    rms = np.array([f.rms for f in frames], dtype=np.float32)
    # Normalise energy to a robust peak (95th percentile) so a single
    # clipped sample doesn't crush the rest of the clip.
    peak = float(np.percentile(rms, 95.0)) if rms.size else 1.0
    energy = rms / max(peak, 1e-4)
    energy = np.clip(energy, 0.0, 1.0)
    duration = float(len(samples)) / sr
    return LipsyncTrack(
        viseme_track=track, energy=energy, frame_times=times,
        duration=duration,
    )
