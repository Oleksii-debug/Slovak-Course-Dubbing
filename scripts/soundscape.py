#!/usr/bin/env python3
"""Retain a lesson's real background and match its estimated room decay.

Requires numpy, scipy, soundfile, ffmpeg. No network access or model downloads.
Supply speech intervals from a speech detector; these are NOT an ASR transcript.

prepare: preserve guarded nonspeech audio, and recover only brief, strongly
template-matched percussive attacks during speech. Original speech is never
mixed wholesale underneath the replacement voice. The extracted stem must be
audited for missed speech before release. Percussive classification and the
room-decay estimate are heuristic, not proof that every keyboard key survived.

mix: remap source transient centers with a dubbing timeline while keeping
their waveform duration unchanged; copy the original pauses without stretching.
Add a quiet, deterministic room approximation to the synthesized speech.
The approximation is explicitly NOT the original room impulse response.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
from scipy import ndimage, signal
import soundfile as sf


SR = 24_000
HOP = 96
BLOCK = 30.0


def write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


class Reader:
    """Bounded, arithmetic-mean downmix; avoid FFmpeg's +3 dB mono sum."""

    def __init__(self, path: Path):
        self.file = sf.SoundFile(str(path))
        self.sr = self.file.samplerate
        self.duration = len(self.file) / self.sr
        self.channels = self.file.channels

    def read(self, start: float, end: float) -> np.ndarray:
        start, end = max(0., start), min(self.duration, end)
        if end <= start:
            return np.empty(0, dtype=np.float32)
        self.file.seek(round(start * self.sr))
        x = self.file.read(round((end - start) * self.sr), dtype="float32", always_2d=True).mean(axis=1)
        if self.sr != SR:
            g = math.gcd(self.sr, SR)
            x = signal.resample_poly(x, SR // g, self.sr // g).astype(np.float32)
        return x[:round((end - start) * SR)]

    def close(self):
        self.file.close()


def intervals(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    data = data.get("segments", data) if isinstance(data, dict) else data
    out = sorted([{"start": float(s["start"]), "end": float(s["end"])} for s in data], key=lambda x: x["start"])
    if not out or any(not (math.isfinite(s["start"]) and math.isfinite(s["end"]) and 0 <= s["start"] < s["end"]) for s in out):
        raise ValueError("Speech intervals must contain finite 0 <= start < end values")
    return out


def expanded(spans: list[dict], duration: float, before: float, after: float) -> list[dict]:
    merged = []
    for s in spans:
        a, b = max(0., s["start"] - before), min(duration, s["end"] + after)
        if merged and a <= merged[-1]["end"]:
            merged[-1]["end"] = max(b, merged[-1]["end"])
        else:
            merged.append({"start": a, "end": b})
    return merged


def overlaps(spans: list[dict], a: float, b: float) -> bool:
    return any(s["start"] < b and s["end"] > a for s in spans)


def quiet_ranges(spans: list[dict], duration: float) -> list[dict]:
    result, cursor = [], 0.
    for s in spans:
        if s["start"] > cursor:
            result.append({"start": cursor, "end": s["start"]})
        cursor = max(cursor, s["end"])
    if cursor < duration:
        result.append({"start": cursor, "end": duration})
    return result


def source_mask(start: float, count: int, blocked: list[dict], fade: float = .025) -> np.ndarray:
    """Cosine transitions lie in the already guarded voice-free region."""
    mask = np.ones(count, np.float32)
    end = start + count / SR
    for s in blocked:
        if s["end"] + fade <= start or s["start"] - fade >= end:
            continue
        a, b = round((s["start"] - start) * SR), round((s["end"] - start) * SR)
        lo, hi = max(0, a), min(count, b)
        if hi > lo:
            mask[lo:hi] = 0
        n = max(1, round(fade * SR))
        lo, hi = max(0, a - n), min(count, a)
        if hi > lo:
            z = (np.arange(lo, hi) - (a - n)) / n
            mask[lo:hi] = np.minimum(mask[lo:hi], .5 + .5 * np.cos(np.pi * z))
        lo, hi = max(0, b), min(count, b + n)
        if hi > lo:
            z = (np.arange(lo, hi) - b) / n
            mask[lo:hi] = np.minimum(mask[lo:hi], .5 - .5 * np.cos(np.pi * z))
    return mask


def detect_candidates(reader: Reader, blocked: list[dict]) -> list[dict]:
    sos = signal.butter(4, [1800, 10_500], btype="bandpass", fs=SR, output="sos")
    events = []
    for base in np.arange(0, reader.duration, BLOCK):
        a, b = max(0., base - .5), min(reader.duration, base + BLOCK + .5)
        x = reader.read(a, b)
        hp = signal.sosfilt(sos, x).astype(np.float32)
        frames = hp[:len(hp) // HOP * HOP].reshape(-1, HOP)
        rms = np.sqrt(np.mean(frames * frames, axis=1) + 1e-14)
        baseline = ndimage.median_filter(rms, size=51)
        score = 20 * np.log10((rms + 1e-7) / (baseline + 1e-7))
        peaks, _ = signal.find_peaks(score, height=8, prominence=6, distance=10)
        for i in peaks:
            t = a + i * HOP / SR
            if t < base or t >= min(reader.duration, base + BLOCK) or rms[i] < .0006:
                continue
            z = x[max(0, i * HOP - 240):min(len(x), i * HOP + 1440)]
            spec = np.abs(np.fft.rfft(z * np.hanning(len(z)), n=2048)) ** 2
            hf = float(spec[np.fft.rfftfreq(2048, 1 / SR) > 1800].sum() / max(spec.sum(), 1e-15))
            events.append({"time": round(float(t), 6), "score_db": round(float(score[i]), 3),
                           "high_frequency_fraction": round(hf, 4), "highband_rms": float(rms[i]),
                           "overlaps_guarded_speech": overlaps(blocked, t - .01, t + .08)})
    return events


def fingerprint(reader: Reader, t: float, duration: float = .04) -> np.ndarray:
    x = reader.read(t - .025, t + .12)
    sos = signal.butter(3, [1800, 10_000], btype="bandpass", fs=SR, output="sos")
    x = signal.sosfiltfilt(sos, x).astype(np.float32)
    peak = round(.015 * SR) + int(np.argmax(np.abs(x[round(.015 * SR):round(.04 * SR)])))
    a = max(0, peak - 120)
    return x[a:a + round(duration * SR)]


def template_similarity(x: np.ndarray, templates: list[np.ndarray]) -> float:
    best = 0.
    for ref in templates:
        if len(x) < len(ref):
            continue
        xy = signal.correlate(x, ref, mode="valid", method="direct")
        xx = signal.correlate(x * x, np.ones(len(ref), np.float32), mode="valid", method="direct")
        score = np.max(np.abs(xy) / np.sqrt(np.maximum(xx, 1e-12)))
        best = max(best, float(score))
    return min(best, 1.)


def transient_excerpt(reader: Reader, time: float) -> tuple[float, np.ndarray]:
    """Short spectral residual, never a whole word-length original voice clip."""
    a = max(0., time - .1)
    x = reader.read(a, time + .2)
    f, t, stft = signal.stft(x, fs=SR, nperseg=256, noverlap=208, boundary="zeros")
    mag = np.abs(stft)
    centre = time - a
    stationary = (t < centre - .025) | (t > centre + .105)
    baseline = np.median(mag[:, stationary], axis=1)[:, None]
    # Preserve original phase; reject sustained voice and low-frequency voicing.
    ratio = np.maximum(mag - 1.6 * baseline, 0) / np.maximum(mag, 1e-9)
    ratio *= np.clip((f[:, None] - 650) / 850, 0, 1)
    _, y = signal.istft(stft * ratio, fs=SR, nperseg=256, noverlap=208)
    left = max(0, round((centre - .012) * SR))
    right = min(len(y), round((centre + .105) * SR))
    y = y[left:right].astype(np.float32)
    fade = min(round(.006 * SR), len(y) // 2)
    y[:fade] *= np.linspace(0, 1, fade, dtype=np.float32)
    y[-fade:] *= np.linspace(1, 0, fade, dtype=np.float32)
    return a + left / SR, y


def estimate_decay(reader: Reader, spans: list[dict], events: list[dict]) -> dict:
    fits = []
    for e in events:
        t = e["time"]
        if e["overlaps_guarded_speech"] or e["score_db"] < 18 or e["high_frequency_fraction"] < .2 or t < 20 or t + .8 > reader.duration:
            continue
        if overlaps(spans, t - .15, t + .7):
            continue
        if any(t + .05 < q["time"] < t + .5 and q["highband_rms"] > e["highband_rms"] * .2 for q in events):
            continue
        z = reader.read(t - .02, t + .7)
        z = signal.sosfilt(signal.butter(3, [500, 5000], fs=SR, btype="bandpass", output="sos"), z)
        energy = np.mean(z[:len(z) // 240 * 240].reshape(-1, 240) ** 2, axis=1)
        noise = float(np.median(energy[-20:]))
        db = 10 * np.log10(np.maximum(energy - noise, 1e-12))
        db -= max(db[:7])
        tt = np.arange(len(db)) * .01 - .02
        use = (tt >= .02) & (tt < .5) & (db < -5) & (db > -25) & (energy > noise * 3)
        if np.sum(use) < 8:
            continue
        slope, intercept = np.polyfit(tt[use], db[use], 1)
        r2 = 1 - np.sum((db[use] - (slope * tt[use] + intercept)) ** 2) / max(np.sum((db[use] - db[use].mean()) ** 2), 1e-9)
        rt60 = -60 / slope if slope < 0 else 0
        if r2 >= .65 and .15 <= rt60 <= 1.5:
            fits.append({"source_time": t, "apparent_rt60_seconds": float(rt60), "fit_r_squared": float(r2), "samples": int(use.sum())})
    values = [x["apparent_rt60_seconds"] for x in fits]
    return {"method": "noise-subtracted 500–5000 Hz transient energy decay, extrapolated to 60 dB",
            "original_impulse_response_recovered": False,
            "limitation": "Keyboard/body resonance and incidental sounds can affect this estimate; it is not an acoustic room measurement.",
            "usable_fits": fits, "median_seconds": float(np.median(values)) if values else None,
            "range_seconds": [min(values), max(values)] if values else None}


def choose_room_tone(reader: Reader, gaps: list[dict], events: list[dict]) -> tuple[float, np.ndarray]:
    best = None
    for gap in gaps:
        a, b = max(20., gap["start"] + .08), min(reader.duration - 10, gap["end"] - .08)
        for start in np.arange(a, b - .8, .4):
            if any(start - .15 < e["time"] < start + .95 and e["highband_rms"] > .005 for e in events):
                continue
            x = reader.read(float(start), float(start + .8))
            rms = float(np.sqrt(np.mean(x * x)))
            if 1e-5 < rms and (best is None or rms < best[0]):
                best = (rms, float(start), x)
    if best is None:
        return 0., np.zeros(round(.8 * SR), np.float32)
    return best[1], best[2]


def prepare(args) -> dict:
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    reader = Reader(args.source)
    spans = intervals(args.speech_spans)
    vad_blocked = expanded(spans, reader.duration, args.guard_before, args.guard_after)
    reviewed = intervals(args.reviewed_source_spans) if args.reviewed_source_spans else []
    combined_spans = sorted(spans + reviewed, key=lambda s: s["start"])
    blocked = expanded(combined_spans, reader.duration, args.guard_before, args.guard_after)
    gaps = quiet_ranges(blocked, reader.duration)
    candidates = detect_candidates(reader, blocked)
    # Templates come from VAD-confirmed quiet intervals even when a reviewed
    # sentence interval spans intervening keyboard sounds and pauses.
    refs = sorted([e for e in candidates if not overlaps(vad_blocked, e["time"] - .01, e["time"] + .08) and e["time"] > 20 and e["score_db"] >= 18 and e["high_frequency_fraction"] >= .25 and e["highband_rms"] > .015], key=lambda e: e["highband_rms"], reverse=True)[:24]
    templates = [fingerprint(reader, e["time"]) for e in refs]
    templates = [x / max(float(np.linalg.norm(x)), 1e-9) for x in templates]
    extracted = []
    for e in candidates:
        if not e["overlaps_guarded_speech"] or e["score_db"] < 14 or e["high_frequency_fraction"] < .2 or e["highband_rms"] < .01:
            continue
        match = template_similarity(fingerprint(reader, e["time"], .06), templates)
        if match < args.template_threshold:
            continue
        a, y = transient_excerpt(reader, e["time"])
        record = dict(e, template_correlation=match, start=a, end=a + len(y) / SR, samples=y)
        extracted.append(record)
    decay = estimate_decay(reader, spans, candidates)
    tone_start, tone = choose_room_tone(reader, gaps, candidates)
    sf.write(str(out / "room_tone.flac"), tone, SR, subtype="PCM_16")
    clipped, peak, energy, frames = 0, 0., 0., 0
    bed_path = out / "source_soundscape.flac"
    with sf.SoundFile(str(bed_path), mode="w", samplerate=SR, channels=1, format="FLAC", subtype="PCM_16") as writer:
        for base in np.arange(0, reader.duration, BLOCK):
            x = reader.read(float(base), min(reader.duration, float(base + BLOCK)))
            mask = source_mask(float(base), len(x), blocked)
            y = x * mask
            for e in extracted:
                offset = round((e["start"] - base) * SR)
                lo, hi = max(0, offset), min(len(y), offset + len(e["samples"]))
                if hi > lo:
                    y[lo:hi] += e["samples"][lo - offset:hi - offset] * (1 - mask[lo:hi])
            clipped += int(np.sum(np.abs(y) >= 1))
            peak = max(peak, float(np.max(np.abs(y))))
            energy += float(np.sum(y.astype(np.float64) ** 2))
            frames += len(y)
            writer.write(np.clip(y, -.9999, .9999))
    # Nonoverlapping source excerpts prevent a rapid sequence's neighboring
    # click from being copied twice when speech slots are greatly expanded.
    rendered_events = []
    for e in candidates:
        if not e["overlaps_guarded_speech"]:
            rendered_events.append(dict(time=e["time"], start=max(0., e["time"] - .012),
                                        end=min(reader.duration, e["time"] + .24), kind="source_nonspeech"))
    rendered_events.extend(dict(time=e["time"], start=e["start"], end=e["end"], kind="isolated_transient") for e in extracted)
    rendered_events.sort(key=lambda e: e["time"])
    for index, e in enumerate(rendered_events):
        if index:
            e["start"] = max(e["start"], (rendered_events[index - 1]["time"] + e["time"]) / 2)
        if index + 1 < len(rendered_events):
            e["end"] = min(e["end"], (e["time"] + rendered_events[index + 1]["time"]) / 2)
    report = {"schema_version": 1, "sample_rate": SR, "source_duration": reader.duration,
              "source_sha256": sha256(args.source), "source_channels": reader.channels,
              "source_downmix": "arithmetic mean of channels", "speech_regions": len(spans),
              "reviewed_source_intervals": len(reviewed),
              "speech_guard_before": args.guard_before, "speech_guard_after": args.guard_after,
              "retained_nonspeech_seconds": sum(g["end"] - g["start"] for g in gaps),
              "guarded_speech_intervals": blocked, "retained_nonspeech_intervals": gaps,
              "transient_candidates": len(candidates),
              "nonspeech_transient_candidates_retained": sum(not e["overlaps_guarded_speech"] for e in candidates),
              "template_reference_times": [e["time"] for e in refs],
              "overlapping_transients_extracted": len(extracted), "template_threshold": args.template_threshold,
              "overlap_events": [{k: v for k, v in e.items() if k != "samples"} for e in extracted],
              "rendered_transient_events": rendered_events,
              "room_decay": decay, "room_tone_source_start": tone_start,
              "room_tone_duration": len(tone) / SR, "room_tone_rms_dbfs": 20 * math.log10(max(float(np.sqrt(np.mean(tone * tone))), 1e-9)),
              "stem_peak": peak, "stem_rms_dbfs": 10 * math.log10(max(energy / frames, 1e-18)),
              "clipped_input_samples": clipped, "soundscape_file": bed_path.name,
              "limitations": ["Automatic VAD can miss speech; audit the isolated stem before release.",
                              "Overlapping quiet keys can remain unrecoverable; only confident transient matches are retained.",
                              "Nonspeech sound is source audio; replacement-voice room reverberation is an estimate."]}
    write_json(out / "soundscape_report.json", report)
    write_json(out / "transient_candidates_reproducible.json", candidates)
    reader.close()
    return report


def place_transients(reader: Reader, span: dict, events: list[dict], count: int) -> tuple[np.ndarray, int]:
    """Map centers; copy real click waveforms at 1x speed and original duration."""
    y = np.zeros(count, np.float32)
    duration = span["source_end"] - span["source_start"]
    target_duration = span["output_end"] - span["output_start"]
    placed = 0
    for e in events:
        if not (span["source_start"] <= e["time"] < span["source_end"]):
            continue
        x = reader.read(e["start"], e["end"])
        if not len(x):
            continue
        # The short boundary fades only affect the excerpt edges. Natural
        # onset, pitch, decay speed, and duration of the transient stay intact.
        fade = min(round(.003 * SR), len(x) // 2)
        x[:fade] *= np.linspace(0, 1, fade, dtype=np.float32)
        x[-fade:] *= np.linspace(1, 0, fade, dtype=np.float32)
        center = (e["time"] - span["source_start"]) / duration * target_duration
        offset = round((center - (e["time"] - e["start"])) * SR)
        lo, hi = max(0, offset), min(count, offset + len(x))
        if hi > lo:
            y[lo:hi] += x[lo - offset:hi - offset]
            placed += 1
    return y, placed


def excerpt_envelope(count: int) -> np.ndarray:
    envelope = np.ones(count, np.float32)
    fade = min(round(.003 * SR), count // 2)
    if fade:
        envelope[:fade] = np.linspace(0, 1, fade, dtype=np.float32)
        envelope[-fade:] = np.linspace(1, 0, fade, dtype=np.float32)
    return envelope


def remove_excerpt_windows(x: np.ndarray, source_start: float, events: list[dict]) -> None:
    """Remove precisely the windows later overlaid once on the global timeline."""
    for e in events:
        offset = round(e["start"] * SR) - round(source_start * SR)
        count = round((e["end"] - e["start"]) * SR)
        lo, hi = max(0, offset), min(len(x), offset + count)
        if hi > lo:
            envelope = excerpt_envelope(count)
            x[lo:hi] *= 1 - envelope[lo - offset:hi - offset]


def overlay_global_transients(path: Path, reader: Reader, mapped: list[dict], events: list[dict]) -> dict:
    """One complete source excerpt per event, including across slot boundaries."""
    counts = {"speech_slot": 0, "gap": 0, "trailing_gap": 0, "clipped_at_complete_audio_boundary": 0}
    with sf.SoundFile(str(path), "r+") as target:
        for e in events:
            span = next((s for s in mapped if s["source_start"] <= e["time"] < s["source_end"]), None)
            if span is None or span["output_end"] <= span["output_start"]:
                continue
            if span["kind"] == "speech_slot":
                fraction = (e["time"] - span["source_start"]) / (span["source_end"] - span["source_start"])
                center = span["output_start"] + fraction * (span["output_end"] - span["output_start"])
            else:
                center = span["output_start"] + e["time"] - span["source_start"]
            x = reader.read(e["start"], e["end"])
            x *= excerpt_envelope(len(x))
            offset = round((center - (e["time"] - e["start"])) * SR)
            lo, hi = max(0, offset), min(len(target), offset + len(x))
            if hi <= lo:
                continue
            if lo != offset or hi != offset + len(x):
                counts["clipped_at_complete_audio_boundary"] += 1
            target.seek(lo)
            current = target.read(hi - lo, dtype="float32")
            current += x[lo - offset:hi - offset]
            target.seek(lo)
            target.write(current)
            counts[span["kind"]] += 1
    return counts


def timeline_intervals(data, source_duration: float, output_duration: float) -> list[dict]:
    """Accept assemble_lesson JSON or original_start/end + new_start/end."""
    segments = data.get("segments", data) if isinstance(data, dict) else data
    if not isinstance(segments, list) or not segments:
        raise ValueError("A nonempty dubbing timeline is required")
    result, source_cursor, output_cursor = [], 0., 0.
    for s in segments:
        a, b = float(s["original_start"]), float(s["original_end"])
        c = float(s.get("output_start", s.get("new_start")))
        d = float(s.get("output_end", s.get("new_end", s.get("output_audio_end"))))
        if not all(math.isfinite(v) for v in (a, b, c, d)) or a < source_cursor - .002 or c < output_cursor - .002 or b <= a or d <= c:
            raise ValueError("Timeline source/output intervals must be ordered, finite, and nonoverlapping")
        if a > source_cursor or c > output_cursor:
            result.append(dict(source_start=source_cursor, source_end=a, output_start=output_cursor, output_end=c, kind="gap"))
        result.append(dict(source_start=a, source_end=b, output_start=c, output_end=d, kind="speech_slot"))
        source_cursor, output_cursor = b, d
    if source_cursor > source_duration + .15 or output_cursor > output_duration + .15:
        raise ValueError("Timeline exceeds source or output audio")
    if source_duration > source_cursor or output_duration > output_cursor:
        result.append(dict(source_start=source_cursor, source_end=source_duration, output_start=output_cursor, output_end=output_duration, kind="trailing_gap"))
    return result


def estimated_room_ir(rt60: float, seed: int = 576) -> np.ndarray:
    """Low-coloration diffuse tail; deterministically estimated, not measured."""
    rt60 = float(np.clip(rt60, .15, 1.5))
    length = round((rt60 + .08) * SR)
    rng = np.random.default_rng(seed)
    t = np.arange(length) / SR
    decay = np.exp(-math.log(1000) * t / rt60)
    tail = rng.standard_normal(length).astype(np.float32) * decay
    tail = signal.sosfilt(signal.butter(2, [180, 6000], fs=SR, btype="bandpass", output="sos"), tail).astype(np.float32)
    tail[:round(.014 * SR)] = 0
    tail /= max(float(np.linalg.norm(tail)), 1e-9)
    # A few quiet early reflections; diffuse tail carries most of the energy.
    for delay, amplitude in ((.018, .30), (.031, .23), (.047, .16), (.073, .11)):
        tail[round(delay * SR)] += amplitude
    tail /= max(float(np.linalg.norm(tail)), 1e-9)
    return tail.astype(np.float32)


def mix(args) -> dict:
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    report_path = args.soundscape_dir / "soundscape_report.json"
    prepared = json.loads(report_path.read_text(encoding="utf-8"))
    stem = Reader(args.soundscape_dir / prepared["soundscape_file"])
    speech = Reader(args.speech)
    data = json.loads(args.timeline.read_text(encoding="utf-8"))
    mapped = timeline_intervals(data, stem.duration, speech.duration)
    rt60 = args.rt60 or prepared["room_decay"]["median_seconds"] or .45
    ir = estimated_room_ir(rt60)
    wet_gain, bed_gain = 10 ** (args.reverb_db / 20), 10 ** (args.bed_db / 20)
    tone_reader = Reader(args.soundscape_dir / "room_tone.flac")
    tone = tone_reader.read(0, tone_reader.duration)
    tone_reader.close()
    # Crossfade a real quiet sample into a periodic low-level background. It is
    # deliberately quieter than the preserved original pauses to avoid hiss.
    crossfade = min(round(.03 * SR), len(tone) // 4)
    if crossfade:
        ramp = np.linspace(0, 1, crossfade, dtype=np.float32)
        tone[:crossfade] = tone[-crossfade:] * (1 - ramp) + tone[:crossfade] * ramp
        tone = tone[:-crossfade]
    tone_gain = 10 ** (args.room_tone_db / 20)
    with tempfile.TemporaryDirectory(prefix="soundscape-mix-", dir=output.parent) as tmp:
        tmp = Path(tmp)
        # Float prevents a compressed timeline's overlapping keys from clipping
        # before the single final master-gain decision.
        mapped_bed = tmp / "mapped_bed.wav"
        gap_adjustments = []
        with sf.SoundFile(str(mapped_bed), "w", samplerate=SR, channels=1, subtype="FLOAT") as writer:
            for span in mapped:
                count = round(span["output_end"] * SR) - round(span["output_start"] * SR)
                if count > 0:
                    if span["kind"] == "speech_slot":
                        x = np.zeros(count, np.float32)
                    else:
                        x = stem.read(span["source_start"], span["source_end"])
                        remove_excerpt_windows(x, span["source_start"], prepared["rendered_transient_events"])
                        if abs(count - len(x)) > 4:
                            gap_adjustments.append({"source_start": span["source_start"], "difference_seconds": (count - len(x)) / SR})
                        # Preserve gaps at natural speed. Extra output padding
                        # receives continuous room tone at the final mix stage.
                        x = np.pad(x, (0, max(0, count - len(x))))[:count]
                    writer.write(x)
        # Overlay once globally: an event crossing a speech/gap or language
        # boundary must neither lose its tail nor play that tail twice.
        event_counts = overlay_global_transients(mapped_bed, stem, mapped, prepared["rendered_transient_events"])
        bed = Reader(mapped_bed)
        raw = tmp / "mixed_float.wav"
        tail = np.zeros(len(ir) - 1, np.float32)
        peak, total_energy, samples = 0., 0., 0
        with sf.SoundFile(str(raw), "w", samplerate=SR, channels=1, subtype="FLOAT") as writer:
            for base in np.arange(0, speech.duration, BLOCK):
                end = min(speech.duration, base + BLOCK)
                x = speech.read(float(base), float(end))
                ambience = bed.read(float(base), float(end))
                ambience = np.pad(ambience, (0, max(0, len(x) - len(ambience))))[:len(x)]
                reverberant = signal.oaconvolve(x, ir).astype(np.float32)
                reverberant[:len(tail)] += tail
                tail = reverberant[len(x):].copy()
                room_tone = tone[(np.arange(len(x)) + round(base * SR)) % len(tone)] if len(tone) else 0.
                # A continuous quiet floor avoids repeated noise onsets at the
                # beginning of sentences. Its level is set independently.
                y = x + wet_gain * reverberant[:len(x)] + bed_gain * ambience + tone_gain * room_tone
                peak = max(peak, float(np.max(np.abs(y))))
                total_energy += float(np.sum(y.astype(np.float64) ** 2))
                samples += len(y)
                writer.write(y)
        bed.close()
        gain = min(1., 10 ** (-args.peak_headroom_db / 20) / max(peak, 1e-9))
        command = ["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-threads", "2", "-i", str(raw), "-af", f"volume={gain:.12f}", "-map_metadata", "-1"]
        if output.suffix.lower() == ".mp3":
            command += ["-c:a", "libmp3lame", "-b:a", "160k"]
        elif output.suffix.lower() == ".flac":
            command += ["-c:a", "flac", "-sample_fmt", "s16"]
        elif output.suffix.lower() == ".wav":
            command += ["-c:a", "pcm_s24le"]
        else:
            raise ValueError("Output must be MP3, FLAC, or WAV")
        command.append(str(output))
        subprocess.run(command, check=True, capture_output=True)
    stem.close()
    speech.close()
    report = {"schema_version": 1, "output": output.name, "output_sha256": sha256(output),
              "speech_sha256": sha256(args.speech), "source_sha256": prepared["source_sha256"],
              "timeline_sha256": sha256(args.timeline), "duration_seconds": samples / SR,
              "room_reverb": "deterministic approximation, not original impulse response",
              "rt60_seconds": rt60, "reverb_gain_db": args.reverb_db, "source_bed_gain_db": args.bed_db,
              "reused_source_room_tone_gain_db": args.room_tone_db,
              "pre_normalization_peak": peak, "master_gain_db": 20 * math.log10(gain),
              "final_estimated_peak_dbfs": 20 * math.log10(max(peak * gain, 1e-9)),
              "final_rms_dbfs": 10 * math.log10(max(total_energy / samples, 1e-18)) + 20 * math.log10(gain),
              "mapped_intervals": len(mapped),
              "retiming_method": "transient centers mapped proportionally; excerpt duration/envelope unchanged; original gaps copied at 1x"}
    report["transients_mapped_inside_speech_slots"] = event_counts["speech_slot"]
    report["transient_overlay_counts"] = event_counts
    report["gap_padding_or_trimming"] = gap_adjustments
    write_json(args.report or output.with_suffix(".soundscape.json"), report)
    return report


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    q = sub.add_parser("prepare")
    q.add_argument("--source", type=Path, required=True)
    q.add_argument("--speech-spans", type=Path, required=True)
    q.add_argument("--reviewed-source-spans", type=Path, help="optional reviewed ASR source intervals to union with VAD for speech exclusion")
    q.add_argument("--output-dir", type=Path, required=True)
    q.add_argument("--guard-before", type=float, default=.10)
    q.add_argument("--guard-after", type=float, default=.20)
    q.add_argument("--template-threshold", type=float, default=.75)
    m = sub.add_parser("mix")
    m.add_argument("--speech", type=Path, required=True)
    m.add_argument("--timeline", type=Path, required=True)
    m.add_argument("--soundscape-dir", type=Path, required=True)
    m.add_argument("--output", type=Path, required=True)
    m.add_argument("--report", type=Path)
    m.add_argument("--rt60", type=float)
    m.add_argument("--reverb-db", type=float, default=-19.)
    m.add_argument("--bed-db", type=float, default=-3.)
    m.add_argument("--room-tone-db", type=float, default=-12., help="level relative to the actual quiet source-room recording")
    m.add_argument("--peak-headroom-db", type=float, default=1.5)
    args = p.parse_args()
    try:
        result = prepare(args) if args.command == "prepare" else mix(args)
    except (ValueError, OSError, subprocess.SubprocessError) as exc:
        print(f"Soundscape failed: {exc}", file=sys.stderr)
        return 1
    if args.command == "prepare":
        print(json.dumps({k: result[k] for k in ("retained_nonspeech_seconds", "nonspeech_transient_candidates_retained", "overlapping_transients_extracted", "room_decay", "stem_peak")}, ensure_ascii=False))
    else:
        print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
