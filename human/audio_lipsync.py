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
    # Vowel band split chosen against English F1/F2 ranges (Peterson &
    # Barney 1952): F1 typically 250-800 Hz (open vs close), F2 typically
    # 800-2700 Hz (back vs front). Above 3 kHz is essentially fricative
    # territory.
    lf = freqs < 800.0
    mf = (freqs >= 800.0) & (freqs < 3000.0)
    hf = freqs >= 3000.0

    out: List[Frame] = []
    # Spectral features are computed on the RAW signal. Pre-emphasis
    # (0.97 first-order HP) is a useful robustification for MFCC-style
    # pipelines but here it biases lf/mf/hf ratios toward HF, distorting
    # the vowel classifier. Fricative detection leans on ZCR instead.
    n = len(samples)
    i = 0
    while i + win <= n:
        raw = samples[i:i + win]
        rms = float(np.sqrt(np.mean(raw * raw) + 1e-12))
        dbfs = 20.0 * math.log10(max(rms, 1e-6))
        spec = np.abs(np.fft.rfft(raw * window)) + 1e-9
        total = float(spec.sum())
        lf_e = float(spec[lf].sum()) / total
        mf_e = float(spec[mf].sum()) / total
        hf_e = float(spec[hf].sum()) / total
        centroid = float((freqs * spec).sum() / total)
        # Zero crossings on the raw segment -- fricatives have high ZCR.
        # Normalised by sample rate so the feature is sample-rate-
        # independent (zc-per-second / sr).
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

@dataclass
class _ClipStats:
    """Per-clip reference percentiles used for adaptive thresholds.

    Absolute lf/mf/hf ratios vary wildly with microphone, room tone,
    speaker gender and noise. A vowel that would show lf=0.8 on a lo-fi
    telephone recording might only reach lf=0.4 on a crisp condenser.
    We anchor thresholds to within-clip quantiles of the active frames
    so the classifier adapts to each recording's spectral baseline.
    """
    centroid_lo: float
    centroid_hi: float
    lf_hi: float
    hf_hi: float
    zcr_hi: float


def _clip_stats(frames: List[Frame]) -> _ClipStats:
    active = [f for f in frames if f.dbfs >= SIL_DBFS]
    if not active:
        return _ClipStats(1500.0, 3000.0, 0.4, 0.5, 0.10)
    c = np.array([f.centroid for f in active])
    lf = np.array([f.lf_ratio for f in active])
    hf = np.array([f.hf_ratio for f in active])
    zc = np.array([f.zcr for f in active])
    return _ClipStats(
        centroid_lo=float(np.percentile(c, 30)),
        centroid_hi=float(np.percentile(c, 75)),
        lf_hi=float(np.percentile(lf, 70)),
        hf_hi=float(np.percentile(hf, 70)),
        zcr_hi=float(np.percentile(zc, 75)),
    )


def _classify(f: Frame, st: _ClipStats) -> str:
    """Map one analysis frame to a Preston-Blair viseme using per-clip
    adaptive thresholds.

    Decision order (most discriminative first):
      1. silence
      2. fricative family (high ZCR relative to clip baseline)
      3. rounded back vowels (low centroid + high LF concentration)
      4. open front vowel (mid-high centroid + decent loudness)
      5. mid front vowel E (falls out of the remaining mid band)
      6. catch-all 'etc' (unvoiced release / K/G/H/schwa)

    Coarticulation smoothing is left to ``face_anim.VisemeTrack``.
    """
    if f.dbfs < SIL_DBFS:
        return "sil"

    # Fricative family: ZCR spikes for unvoiced turbulence. Use the
    # clip's own ZCR distribution so we stay calibrated across mic/room.
    if f.zcr > max(0.05, st.zcr_hi) and f.hf_ratio > st.hf_hi * 0.85:
        if f.centroid > 5500.0 or f.zcr > max(0.12, st.zcr_hi * 1.6):
            return "S"       # /s z ʃ ʒ tʃ dʒ/
        if f.centroid > 3500.0:
            return "TH"      # /θ ð/
        return "FV"          # /f v/

    # Rounded close back: LF mass concentrated, centroid very low.
    if f.lf_ratio > st.lf_hi and f.centroid < st.centroid_lo:
        return "U" if f.dbfs < -22.0 else "O"

    # Open back (AI / "father"): moderate LF, moderate centroid, strong
    # loudness. High-energy vowels with centroid in the lower half of
    # the clip's range read as open.
    if f.centroid < st.centroid_lo * 1.1 and f.dbfs > SIL_DBFS + 8.0:
        return "AI"

    # Front vowel E: centroid in upper band, LF is subdued.
    if f.centroid > st.centroid_hi * 0.9 and f.lf_ratio < st.lf_hi:
        return "E"

    # Mid / near-open / schwa / consonant release.
    if f.dbfs < SIL_DBFS + 10.0:
        # Quiet but not silent: closing plosives / mumbles -- read as MBP
        # transit (lips together) if surrounded by speech.
        return "etc"
    return "AI" if f.lf_ratio > st.lf_hi * 0.8 else "E"


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
    stats = _clip_stats(frames)
    labels = [_classify(f, stats) for f in frames]
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
