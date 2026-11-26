#!/usr/bin/env python3
"""Remove terminal service padding from converted SK/EN, preserving PCM exactly.

Retains the maximum of last WordBoundary end +300 ms, last nonquiet native
frame +125 ms, and last nonquiet converted frame +125 ms. Does not shorten
internal pauses, change speed, fade speech, or replace the untrimmed inputs.
These checks establish signal integrity, not native pronunciation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import soundfile as sf


PARAMETERS = {"word_boundary_guard_ms": 300, "audio_end_guard_ms": 125,
              "frame_ms": 10, "speech_percentile": 95,
              "relative_rms_threshold": 0.004,
              "absolute_rms_floor": 2 / 32768,
              "absolute_rms_ceiling": 0.0005}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def nonquiet_end(samples, sample_rate):
    """Conservatively bound speech using low-threshold 10 ms RMS frames."""
    values = np.asarray(samples, dtype=np.float64)
    if values.ndim != 1 or len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("A finite, nonempty mono signal is required")
    width = max(1, round(sample_rate * PARAMETERS["frame_ms"] / 1000))
    padded = np.pad(values, (0, (-len(values)) % width))
    levels = np.sqrt(np.mean(padded.reshape(-1, width) ** 2, axis=1))
    reference = float(np.percentile(levels, PARAMETERS["speech_percentile"]))
    threshold = max(PARAMETERS["absolute_rms_floor"], min(
        PARAMETERS["absolute_rms_ceiling"], reference * PARAMETERS["relative_rms_threshold"]))
    nonquiet = np.flatnonzero(levels >= threshold)
    if not len(nonquiet):
        raise ValueError("No nonquiet speech was found; refusing to trim")
    end_sample = min(len(values), (int(nonquiet[-1]) + 1) * width)
    return {"last_nonquiet_seconds": end_sample / sample_rate,
            "threshold_rms": threshold, "reference_rms": reference,
            "frame_ms": width / sample_rate * 1000}


def decide_trim(native, converted, sample_rate, last_word_end):
    native_end = nonquiet_end(native, sample_rate)
    converted_end = nonquiet_end(converted, sample_rate)
    duration = len(converted) / sample_rate
    if not np.isfinite(last_word_end) or not 0 < last_word_end <= duration + .05:
        raise ValueError("WordBoundary timing does not fit the supplied waveform")
    requested = max(last_word_end + PARAMETERS["word_boundary_guard_ms"] / 1000,
                    native_end["last_nonquiet_seconds"] + PARAMETERS["audio_end_guard_ms"] / 1000,
                    converted_end["last_nonquiet_seconds"] + PARAMETERS["audio_end_guard_ms"] / 1000)
    keep = min(len(converted), max(1, int(np.ceil(requested * sample_rate))))
    removed = converted[keep:]
    # Every removed frame is after the last nonquiet frame of both signals.
    # No word or phonetic tail is shortened to meet an original timing slot.
    return keep, {"original_seconds": duration, "output_seconds": keep / sample_rate,
                  "removed_seconds": (len(converted) - keep) / sample_rate,
                  "last_word_boundary_end_seconds": last_word_end,
                  "native_end": native_end, "converted_end": converted_end,
                  "removed_rms": float(np.sqrt(np.mean(removed ** 2))) if len(removed) else 0.0,
                  "removed_peak": float(np.max(np.abs(removed))) if len(removed) else 0.0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-dir", required=True, type=Path)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--segments", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--language", choices=("sk", "en"),
                        help="Select a required language; absent means paired files present in these directories")
    args = parser.parse_args()
    if args.output_dir.resolve() in {args.input_dir.resolve(), args.native_dir.resolve()}:
        raise ValueError("Keep the untrimmed source and conversion in their original directories")
    raw = json.loads(args.segments.read_text(encoding="utf-8"))
    rows = raw["segments"] if isinstance(raw, dict) else raw
    selected, seen = [], set()
    for row in rows:
        if row["language"] not in {"sk", "en"} or args.language and row["language"] != args.language:
            continue
        name = str(row["id"])
        if name in seen or not re.fullmatch(r"[A-Za-z0-9_-]+", name):
            raise ValueError("Repeated or unsafe segment identifier")
        seen.add(name)
        native_path, input_path = args.native_dir / (name + ".wav"), args.input_dir / (name + ".wav")
        if not args.language and not native_path.exists() and not input_path.exists():
            continue
        if not native_path.is_file() or not input_path.is_file():
            raise FileNotFoundError("Missing native/converted pair for " + name)
        selected.append(row)
    if not selected:
        raise ValueError("No paired foreign-language spans selected")
    source_manifest = json.loads((args.input_dir / "manifest.json").read_text(encoding="utf-8"))
    identity = {"backend": "metadata-and-two-waveform-terminal-padding-trim",
                "script_sha256": sha256(__file__), "parameters": PARAMETERS,
                "source_conversion_identity": source_manifest["identity"]}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "manifest.json"
    previous = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    records = previous.get("segments", {}) if previous.get("identity") == identity else {}
    for row in selected:
        name = str(row["id"])
        native_path = args.native_dir / (name + ".wav")
        input_path = args.input_dir / (name + ".wav")
        metadata_path = args.native_dir / (name + ".json")
        output_path = args.output_dir / (name + ".wav")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        source_record = source_manifest["segments"].get(name)
        if source_record is None:
            raise ValueError("Converted manifest has no record for " + name)
        if metadata["identity"]["text"] != row["text"] or source_record.get("text") != row["text"]:
            raise ValueError("Canonical/native/converted text differs for " + name)
        native_hash, converted_hash = sha256(native_path), sha256(input_path)
        if metadata.get("sha256") != native_hash or source_record.get("wav_sha256") != converted_hash:
            raise ValueError("Input waveform does not match its existing manifest for " + name)
        fingerprint = hashlib.sha256(json.dumps(row, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        metadata_hash = sha256(metadata_path)
        old = records.get(name, {})
        if (old.get("segment_sha256") == fingerprint and old.get("native_wav_sha256") == native_hash
                and old.get("untrimmed_wav_sha256") == converted_hash
                and old.get("native_metadata_sha256") == metadata_hash
                and output_path.is_file() and old.get("wav_sha256") == sha256(output_path)):
            print(name + ": reused verified terminal trim", flush=True)
            continue
        info = sf.info(input_path)
        native_info = sf.info(native_path)
        if (info.channels != 1 or native_info.channels != 1 or info.subtype != "PCM_16"
                or info.samplerate != native_info.samplerate or info.frames != native_info.frames):
            raise ValueError("Use aligned mono native/converted WAVs; converted PCM must be 16 bit")
        pcm, sample_rate = sf.read(input_path, dtype="int16")
        converted = pcm.astype(np.float64) / 32768
        native, native_rate = sf.read(native_path, dtype="float64")
        boundaries = metadata.get("boundaries", [])
        if not boundaries:
            raise ValueError("Native WordBoundary events are required for " + name)
        if any(b.get("offset", -1) < 0 or b.get("duration", -1) < 0 for b in boundaries):
            raise ValueError("Invalid native WordBoundary events")
        event_end = max((b["offset"] + b["duration"]) / 10_000_000 for b in boundaries)
        event_end -= metadata.get("prefix_removed_seconds", 0.0)
        keep, diagnostics = decide_trim(native, converted, sample_rate, event_end)
        temporary = output_path.with_suffix(".wav.tmp")
        sf.write(temporary, pcm[:keep], sample_rate, format="WAV", subtype="PCM_16")
        written, written_rate = sf.read(temporary, dtype="int16")
        if written_rate != sample_rate or not np.array_equal(written, pcm[:keep]):
            temporary.unlink(missing_ok=True)
            raise ValueError("Retained PCM prefix changed")
        temporary.replace(output_path)
        record = {**source_record, "segment_sha256": fingerprint,
                  "native_wav_sha256": native_hash, "untrimmed_wav_sha256": converted_hash,
                  "native_metadata_sha256": metadata_hash, "wav_sha256": sha256(output_path),
                  "source_start": row.get("start"), "source_end": row.get("end"),
                  "output_seconds": keep / sample_rate, "sample_rate": sample_rate,
                  "retained_pcm_sha256": hashlib.sha256(written.astype("<i2").tobytes()).hexdigest(),
                  "prefix_pcm_bit_exact": True, "untrimmed_input_retained": True,
                  "terminal_padding": diagnostics, "native_pronunciation_accepted": False}
        records[name] = record
        write_json(args.output_dir / (name + ".json"), record)
        write_json(manifest_path, {"identity": identity, "segments": records,
                                  "signal_integrity_checks_only": True,
                                  "removed_seconds": sum(r["terminal_padding"]["removed_seconds"] for r in records.values())})
        print(f"{name}: removed {diagnostics['removed_seconds']:.3f}s terminal padding; "
              f"retained {keep/sample_rate:.3f}s PCM exactly", flush=True)


if __name__ == "__main__":
    main()
