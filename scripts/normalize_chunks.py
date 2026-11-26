#!/usr/bin/env python3
"""Match rendered speech levels without changing duration or original WAVs.

Reads synthesize_spans.py's manifest and writes corresponding files into a
different directory. Applies one scalar gain per utterance, measured on active
20 ms speech frames; preserves every sample and avoids peak clipping.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import soundfile as sf


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def active_rms(wave, rate):
    frame = round(rate * 0.020)
    padded = np.pad(wave, (0, (-len(wave)) % frame))
    levels = np.sqrt(np.mean(padded.reshape(-1, frame).astype(np.float64)**2, axis=1))
    threshold = max(0.001, float(levels.max()) * 0.1)
    active = levels[levels >= threshold]
    if not len(active):
        raise ValueError("No sufficiently audible speech frames")
    return float(np.sqrt(np.mean(active**2))), int(len(active)), threshold


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-rms", type=float, default=0.065)
    parser.add_argument("--peak-ceiling", type=float, default=0.95)
    args = parser.parse_args()
    if args.input_dir.resolve() == args.output_dir.resolve():
        raise ValueError("Output directory must differ from original chunks")
    if not 0.001 <= args.target_rms <= 0.3 or not 0.5 <= args.peak_ceiling < 1:
        raise ValueError("Invalid requested gain targets")
    original = json.loads((args.input_dir / "manifest.json").read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = args.output_dir / "manifest.json"
    previous = json.loads(manifest.read_text()) if manifest.exists() else {}
    identity = {"source_identity": original["identity"], "script_sha256": digest(__file__),
                "target_active_rms": args.target_rms, "peak_ceiling": args.peak_ceiling,
                "frame_seconds": 0.020, "active_threshold_relative_to_max": 0.1,
                "active_threshold_absolute": 0.001}
    records = previous.get("segments", {}) if previous.get("identity") == identity else {}
    for name, info in original["segments"].items():
        if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
            raise ValueError("Unsafe segment identifier")
        source, output = args.input_dir / (name + ".wav"), args.output_dir / (name + ".wav")
        source_hash = digest(source)
        if source_hash != info["wav_sha256"]:
            raise ValueError("Source chunk differs from render manifest: " + name)
        prior = records.get(name, {})
        if prior.get("source_wav_sha256") == source_hash and output.is_file() and digest(output) == prior.get("wav_sha256"):
            print(name + ": reused verified normalized WAV", flush=True)
            continue
        wave, rate = sf.read(source, dtype="float32")
        if rate != 24000 or wave.ndim != 1 or not len(wave) or not np.isfinite(wave).all():
            raise ValueError("Expected finite 24 kHz mono audio: " + name)
        rms, frames, threshold = active_rms(wave, rate)
        peak = float(abs(wave).max())
        gain = min(args.target_rms / rms, args.peak_ceiling / peak)
        normalized = wave * gain
        sf.write(output, normalized, rate, subtype="PCM_16")
        after = sf.info(output)
        if after.frames != len(wave) or after.samplerate != rate:
            raise ValueError("Normalization unexpectedly changed timebase")
        records[name] = {**info, "source_wav_sha256": source_hash, "wav_sha256": digest(output),
                         "normalization": {"scalar_gain": gain, "active_rms_before": rms,
                                           "active_rms_after_estimate": rms*gain,
                                           "peak_before": peak, "peak_after_estimate": peak*gain,
                                           "active_frames": frames, "frame_threshold": threshold,
                                           "sample_count_unchanged": True}}
        temporary = manifest.with_suffix(".json.tmp")
        temporary.write_text(json.dumps({"identity": identity, "segments": records},
                                        ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
        temporary.replace(manifest)
        print(f"{name}: gain {gain:.3f}; active RMS {rms:.4f} -> {rms*gain:.4f}; duration unchanged", flush=True)
    print(f"Normalized {len(original['segments'])} chunks", flush=True)


if __name__ == "__main__":
    main()
