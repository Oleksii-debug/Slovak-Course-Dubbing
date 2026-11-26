#!/usr/bin/env python3
"""Assemble synthesized speech while preserving the source lesson's pauses.

Only Python's standard library, ffmpeg and ffprobe are required. Input is a JSON
array of {id, start, end, language, text, audio_path}; times are source seconds.
Relative audio paths are resolved relative to the JSON file. No network calls.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import wave


SAMPLE_RATE = 24_000
SAMPLE_WIDTH = 2


class AssemblyError(ValueError):
    """An invalid manifest or failed audio operation."""


def _number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AssemblyError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise AssemblyError(f"{label} must be a finite number")
    return result


def _seconds(frames: int) -> float:
    return round(frames / SAMPLE_RATE, 9)


def read_manifest(manifest: Path, source_duration: float) -> list[dict]:
    source_duration = _number(source_duration, "source duration")
    if source_duration <= 0:
        raise AssemblyError("source duration must be greater than zero")
    try:
        data = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AssemblyError(f"cannot read manifest: {exc}") from exc
    if not isinstance(data, list) or not data:
        raise AssemblyError("manifest must be a nonempty JSON array")

    spans = []
    seen = set()
    previous_end = 0.0
    previous_end_frame = 0
    for index, raw in enumerate(data):
        label = f"segment {index + 1}"
        if not isinstance(raw, dict):
            raise AssemblyError(f"{label} must be an object")
        missing = {"id", "start", "end", "language", "text", "audio_path"} - raw.keys()
        if missing:
            raise AssemblyError(f"{label} missing fields: {', '.join(sorted(missing))}")
        if isinstance(raw["id"], bool) or not isinstance(raw["id"], (str, int)):
            raise AssemblyError(f"{label} id must be a string or integer")
        segment_id = str(raw["id"])
        if not segment_id.strip() or segment_id in seen:
            raise AssemblyError(f"{label} has an empty or duplicate id: {segment_id!r}")
        seen.add(segment_id)
        start = _number(raw["start"], f"{label} start")
        end = _number(raw["end"], f"{label} end")
        if start < 0 or end <= start:
            raise AssemblyError(f"{label} must have 0 <= start < end")
        if start < previous_end:
            raise AssemblyError(f"{label} overlaps a preceding segment or is out of order")
        if end > source_duration:
            raise AssemblyError(f"{label} extends beyond source duration")
        start_frame, end_frame = round(start * SAMPLE_RATE), round(end * SAMPLE_RATE)
        if end_frame <= start_frame or start_frame < previous_end_frame:
            raise AssemblyError(f"{label} has an invalid interval at {SAMPLE_RATE} Hz")
        for field in ("language", "text", "audio_path"):
            if not isinstance(raw[field], str) or not raw[field].strip():
                raise AssemblyError(f"{label} {field} must be a nonempty string")
        audio_path = Path(raw["audio_path"]).expanduser()
        if not audio_path.is_absolute():
            audio_path = manifest.parent / audio_path
        audio_path = audio_path.resolve()
        if not audio_path.is_file():
            raise AssemblyError(f"{label} audio file does not exist: {audio_path}")
        spans.append({
            "id": raw["id"], "start": start, "end": end,
            "language": raw["language"], "text": raw["text"],
            "audio_path": audio_path, "start_frame": start_frame, "end_frame": end_frame,
        })
        previous_end, previous_end_frame = end, end_frame
    return spans


def _run(command: list[str], operation: str) -> str:
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode:
        detail = result.stderr.strip()[-2000:]
        raise AssemblyError(f"{operation} failed: {detail}")
    return result.stdout


def _write_silence(writer: wave.Wave_write, frames: int) -> None:
    # Bounded memory, including long student-response pauses.
    silence = b"\0" * SAMPLE_RATE * SAMPLE_WIDTH
    while frames:
        count = min(frames, SAMPLE_RATE)
        writer.writeframesraw(silence[: count * SAMPLE_WIDTH])
        frames -= count


def _copy_pcm(writer: wave.Wave_write, normalized: Path) -> int:
    with wave.open(str(normalized), "rb") as reader:
        if (reader.getframerate(), reader.getnchannels(), reader.getsampwidth()) != (SAMPLE_RATE, 1, SAMPLE_WIDTH):
            raise AssemblyError("normalized audio has unexpected PCM format")
        frames = reader.getnframes()
        if frames <= 0:
            raise AssemblyError("a synthesized audio segment is empty")
        copied = 0
        while chunk := reader.readframes(SAMPLE_RATE):
            if len(chunk) % SAMPLE_WIDTH:
                raise AssemblyError("a synthesized audio segment has truncated PCM data")
            copied += len(chunk) // SAMPLE_WIDTH
            writer.writeframesraw(chunk)
        if copied != frames:
            raise AssemblyError("a synthesized audio segment is truncated")
    return frames


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assemble(manifest: Path, source_duration: float, output: Path, timeline: Path, *, pad_shorter: bool = True) -> dict:
    manifest, output, timeline = (Path(p).expanduser().resolve() for p in (manifest, output, timeline))
    spans = read_manifest(manifest, source_duration)
    if output.suffix.lower() != ".mp3":
        raise AssemblyError("output must have an .mp3 extension")
    if output == timeline:
        raise AssemblyError("audio and timeline output paths must be different")
    inputs = {manifest, *(span["audio_path"] for span in spans)}
    if output in inputs or timeline in inputs:
        raise AssemblyError("an output path must not overwrite an input file")
    ffmpeg, ffprobe = shutil.which("ffmpeg"), shutil.which("ffprobe")
    if not ffmpeg or not ffprobe:
        raise AssemblyError("ffmpeg and ffprobe must both be available on PATH")
    output.parent.mkdir(parents=True, exist_ok=True)
    timeline.parent.mkdir(parents=True, exist_ok=True)

    source_frames = round(source_duration * SAMPLE_RATE)
    records = []
    written_ids = []
    cursor = 0
    previous_source_end = 0
    with tempfile.TemporaryDirectory(prefix="assemble-", dir=output.parent) as temporary:
        temporary = Path(temporary)
        pcm_path = temporary / "lesson.wav"
        with wave.open(str(pcm_path), "wb") as writer:
            writer.setnchannels(1)
            writer.setsampwidth(SAMPLE_WIDTH)
            writer.setframerate(SAMPLE_RATE)
            for span in spans:
                gap = span["start_frame"] - previous_source_end
                _write_silence(writer, gap)
                cursor += gap
                output_start = cursor
                normalized = temporary / "segment.wav"
                _run([
                    ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
                    "-i", str(span["audio_path"]), "-map", "0:a:0", "-vn",
                    "-ar", str(SAMPLE_RATE), "-ac", "1", "-c:a", "pcm_s16le", str(normalized),
                ], f"normalize segment {span['id']}")
                audio_frames = _copy_pcm(writer, normalized)
                cursor += audio_frames
                audio_end = cursor
                original_frames = span["end_frame"] - span["start_frame"]
                padding = max(0, original_frames - audio_frames) if pad_shorter else 0
                _write_silence(writer, padding)
                cursor += padding
                records.append({
                    "id": span["id"], "language": span["language"], "text": span["text"],
                    "original_start": span["start"], "original_end": span["end"],
                    "output_start": _seconds(output_start),
                    "output_audio_end": _seconds(audio_end), "output_end": _seconds(cursor),
                    "audio_duration": _seconds(audio_frames),
                    "gap_before": _seconds(gap), "padding_after": _seconds(padding),
                    "expansion": _seconds(max(0, audio_frames - original_frames)),
                    "compression": _seconds(max(0, original_frames - audio_frames)) if not pad_shorter else 0.0,
                    "audio_sha256": _sha256(span["audio_path"]),
                })
                written_ids.append(str(span["id"]))
                previous_source_end = span["end_frame"]
                normalized.unlink()
            trailing = source_frames - previous_source_end
            _write_silence(writer, trailing)
            cursor += trailing

        expected_ids = [str(span["id"]) for span in spans]
        if written_ids != expected_ids or any(count != 1 for count in Counter(written_ids).values()):
            raise AssemblyError("segment integrity check failed: ids must occur exactly once, in order")
        with wave.open(str(pcm_path), "rb") as reader:
            if reader.getnframes() != cursor:
                raise AssemblyError("assembled PCM duration does not match the timeline")

        encoded = temporary / "lesson.mp3"
        _run([
            ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-y", "-i", str(pcm_path),
            "-map", "0:a:0", "-map_metadata", "-1", "-c:a", "libmp3lame", "-b:a", "128k",
            "-ar", str(SAMPLE_RATE), "-ac", "1", str(encoded),
        ], "MP3 encoding")
        probe = json.loads(_run([
            ffprobe, "-v", "error", "-select_streams", "a:0", "-show_entries",
            "stream=codec_name,sample_rate,channels:format=duration", "-of", "json", str(encoded),
        ], "MP3 verification"))
        stream = probe.get("streams", [{}])[0]
        encoded_duration = float(probe["format"]["duration"])
        if stream.get("codec_name") != "mp3" or stream.get("channels") != 1 or int(stream.get("sample_rate", 0)) != SAMPLE_RATE:
            raise AssemblyError("encoded MP3 format verification failed")
        # MP3 container duration includes encoder delay/padding; the PCM timeline does not.
        if not math.isfinite(encoded_duration) or abs(encoded_duration - cursor / SAMPLE_RATE) > 0.15:
            raise AssemblyError("encoded MP3 duration differs unexpectedly from the PCM timeline")
        report = {
            "schema_version": 1, "sample_rate": SAMPLE_RATE, "channels": 1,
            "bitrate_kbps": 128,
            "timing_policy": "preserve_gaps_expand_speech_slots" if pad_shorter else "natural_speech_preserve_source_gaps",
            "source_duration": source_duration, "timeline_duration": _seconds(cursor),
            "mp3_container_duration": encoded_duration,
            "total_expansion": _seconds(cursor - source_frames),
            "trailing_silence": _seconds(trailing), "segment_count": len(records),
            "segment_ids_verified_once": True, "manifest_sha256": _sha256(manifest),
            "output_sha256": _sha256(encoded), "segments": records,
        }
        # A failed generation cannot replace an existing result. Final writes are atomic
        # per file; the two-file audio/report pair is not a filesystem transaction.
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", suffix=".json", prefix=".timeline-", dir=timeline.parent, delete=False) as handle:
            temp_timeline = Path(handle.name)
            try:
                json.dump(report, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            except BaseException:
                temp_timeline.unlink(missing_ok=True)
                raise
        try:
            os.replace(encoded, output)
            os.replace(temp_timeline, timeline)
        finally:
            temp_timeline.unlink(missing_ok=True)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="JSON array of source intervals and synthesized audio paths")
    parser.add_argument("--source-duration", type=float, required=True, help="complete original lesson duration in seconds")
    parser.add_argument("--output", type=Path, required=True, help="destination .mp3 (mono, 24 kHz, 128 kbps)")
    parser.add_argument("--timeline", type=Path, required=True, help="destination timing and integrity JSON")
    parser.add_argument("--natural-timing", action="store_true", help="preserve original pauses without adding silence when synthesized speech is shorter")
    args = parser.parse_args(argv)
    try:
        report = assemble(args.manifest, args.source_duration, args.output, args.timeline, pad_shorter=not args.natural_timing)
    except (AssemblyError, OSError, wave.Error, KeyError, json.JSONDecodeError) as exc:
        print(f"Assembly failed: {exc}", file=sys.stderr)
        return 1
    print(f"Assembled {report['segment_count']} segments; duration {report['timeline_duration']:.3f} s; speech expansion {report['total_expansion']:.3f} s.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
