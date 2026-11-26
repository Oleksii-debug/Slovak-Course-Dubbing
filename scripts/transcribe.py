#!/usr/bin/env python3
"""Stream a reviewable CPU Whisper transcript with source-relative timestamps.

This script transcribes; it does not translate, certify language boundaries, or
approve a transcript for speech synthesis. See docs/TRANSCRIPTION.md.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import math
from pathlib import Path
import platform
import signal
import sys
import time


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def seconds(value):
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise argparse.ArgumentTypeError("Use a finite, nonnegative number of seconds")
    return number


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("Use at least one CPU thread")
    return number


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--input", required=True, type=Path, help="Source audio or video file")
    result.add_argument("--output-dir", required=True, type=Path,
                        help="Directory without earlier transcript output files")
    result.add_argument("--language", choices=("ru", "sk", "uk", "auto"), default="auto",
                        help="Force a decoding language, or detect it per segment (default: auto)")
    result.add_argument("--model", default="medium",
                        help="faster-whisper model name or local model directory (default: medium)")
    result.add_argument("--threads", type=positive_int, default=2,
                        help="CPU inference threads (default: 2)")
    result.add_argument("--start", type=seconds, default=0.0,
                        help="Start in source seconds (default: 0)")
    result.add_argument("--end", type=seconds,
                        help="End in source seconds; clipped to the decoded source duration")
    result.add_argument("--download-root", type=Path,
                        help="Model cache directory, such as models/whisper")
    result.add_argument("--local-files-only", action="store_true",
                        help="Require cached model files; never download model weights")
    return result


def interrupted(signum, frame):
    del frame
    raise KeyboardInterrupt(f"Received signal {signum}")


def main(argv=None):
    arguments = parser()
    args = arguments.parse_args(argv)
    source = args.input.expanduser().resolve()
    if not source.is_file():
        arguments.error(f"Input file does not exist: {source}")
    if args.end is not None and args.end <= args.start:
        arguments.error("--end must be greater than --start")

    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    names = ("metadata.json", "segments.jsonl", "transcript.txt", "segments.json")
    existing = [name for name in names if (output / name).exists()]
    if existing:
        arguments.error("Choose a new output directory; existing files: " + ", ".join(existing))

    metadata_path = output / "metadata.json"
    final_path = output / "segments.json"
    started = time.monotonic()
    metadata = {
        "schema_version": 1,
        "complete": False,
        "status": "running",
        "stage": "initializing",
        "started_at_utc": utc_now(),
        "input_name": source.name,
        "input_path": str(source),
        "model": args.model,
        "device": "cpu",
        "compute_type": "int8",
        "threads": args.threads,
        "requested_language": args.language,
        "language_mode": "model_detected_per_segment" if args.language == "auto" else "forced",
        "requested_start_seconds": args.start,
        "requested_end_seconds": args.end,
        "timestamps": "Absolute seconds in the decoded source, including the requested start offset",
        "segments_written": 0,
        "options": {
            "task": "transcribe",
            "beam_size": 3,
            "word_timestamps": True,
            "condition_on_previous_text": False,
            "vad_filter": True,
            "vad_min_silence_duration_ms": 450,
        },
    }
    write_json(metadata_path, metadata)
    signal.signal(signal.SIGINT, interrupted)
    signal.signal(signal.SIGTERM, interrupted)

    try:
        # Lazy imports let --help work before installing the optional ASR stack.
        from faster_whisper import WhisperModel
        from faster_whisper.audio import decode_audio

        metadata["versions"] = {
            "python": platform.python_version(),
            **{name: version(name) for name in ("faster-whisper", "ctranslate2", "av")},
        }
        metadata["input_sha256"] = sha256_file(source)
        metadata["stage"] = "loading_model"
        write_json(metadata_path, metadata)
        print(f"Loading {args.model}: CPU int8, {args.threads} threads", flush=True)
        model = WhisperModel(
            args.model,
            device="cpu",
            compute_type="int8",
            cpu_threads=args.threads,
            num_workers=1,
            download_root=str(args.download_root.expanduser().resolve()) if args.download_root else None,
            local_files_only=args.local_files_only,
        )
        audio = decode_audio(str(source), sampling_rate=16000)
        duration = len(audio) / 16000
        if args.start >= duration:
            raise ValueError(f"--start is outside the decoded audio ({duration:.3f} seconds)")
        start_sample = int(args.start * 16000)
        end_sample = min(len(audio), int(args.end * 16000)) if args.end is not None else len(audio)
        if end_sample <= start_sample:
            raise ValueError("The selected interval contains no audio samples")
        offset = start_sample / 16000
        metadata.update({
            "stage": "transcribing",
            "source_duration_seconds": duration,
            "effective_start_seconds": offset,
            "effective_end_seconds": end_sample / 16000,
        })
        write_json(metadata_path, metadata)
        segments, info = model.transcribe(
            audio[start_sample:end_sample],
            task="transcribe",
            language=None if args.language == "auto" else args.language,
            multilingual=args.language == "auto",
            beam_size=3,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 450},
            word_timestamps=True,
            condition_on_previous_text=False,
        )
        metadata["model_reported_language"] = info.language
        metadata["model_reported_language_probability"] = info.language_probability
        metadata["selected_duration_after_vad_seconds"] = info.duration_after_vad
        write_json(metadata_path, metadata)

        rows = []
        with (output / "segments.jsonl").open("w", encoding="utf-8") as jsonl, \
                (output / "transcript.txt").open("w", encoding="utf-8") as transcript:
            for segment in segments:
                row = asdict(segment)
                row["start"] = round(row["start"] + offset, 6)
                row["end"] = round(row["end"] + offset, 6)
                for word in row.get("words") or []:
                    word["start"] = round(word["start"] + offset, 6)
                    word["end"] = round(word["end"] + offset, 6)
                rows.append(row)
                jsonl.write(json.dumps(row, ensure_ascii=False) + "\n")
                jsonl.flush()
                line = f"[{row['start']:09.2f} - {row['end']:09.2f}] {row['text'].strip()}"
                transcript.write(line + "\n")
                transcript.flush()
                metadata["segments_written"] = len(rows)
                metadata["last_source_end_seconds"] = row["end"]
                write_json(metadata_path, metadata)
                print(line, flush=True)

        # Commit the final array only after consuming the transcription generator.
        write_json(final_path, rows)
        metadata.update({
            "complete": True,
            "status": "complete",
            "stage": "finished",
            "finished_at_utc": utc_now(),
            "elapsed_seconds": round(time.monotonic() - started, 3),
        })
        write_json(metadata_path, metadata)
        print(f"Complete: {len(rows)} segments; {final_path}", flush=True)
        return 0
    except (KeyboardInterrupt, Exception) as error:
        final_path.unlink(missing_ok=True)
        metadata.update({
            "complete": False,
            "status": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
            "error_type": type(error).__name__,
            "error": str(error),
            "finished_at_utc": utc_now(),
            "elapsed_seconds": round(time.monotonic() - started, 3),
        })
        write_json(metadata_path, metadata)
        print(f"{metadata['status']}: {error}. Partial output remains in {output}", file=sys.stderr)
        return 130 if isinstance(error, KeyboardInterrupt) else 1


if __name__ == "__main__":
    raise SystemExit(main())
