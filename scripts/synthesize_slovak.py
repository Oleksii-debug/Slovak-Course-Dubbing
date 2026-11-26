#!/usr/bin/env python3
"""Synthesize reviewed Slovak spans, or explicit English examples.

Uses the Edge speech service through edge-tts; this stage requires network.
The Slovak voice needs no language primer. Legacy multilingual voices still
prime short words and crop that primer using returned word timings.
Only the reviewed text is sent. The source recording is never uploaded.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time
import wave


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def clean_word(word):
    return re.sub(r"[^\w]", "", word, flags=re.UNICODE).casefold()


async def run(args):
    import edge_tts
    import edge_tts.communicate
    import edge_tts.voices

    # Trust the explicitly configured runtime CA while keeping TLS verification.
    certificate = os.environ.get("CODEX_PROXY_CERT")
    if certificate:
        edge_tts.communicate._SSL_CTX.load_verify_locations(certificate)
        edge_tts.voices._SSL_CTX.load_verify_locations(certificate)
    raw = json.loads(args.segments.read_text(encoding="utf-8"))
    rows = raw["segments"] if isinstance(raw, dict) else raw
    rows = [row for row in rows if row["language"] == args.language]
    if not rows:
        raise ValueError("No reviewed " + args.language + " text was supplied")
    if args.language == "en" and not args.voice.startswith("en-"):
        raise ValueError("English examples require an explicit English voice")
    ids = [str(row["id"]) for row in rows]
    if len(ids) != len(set(ids)) or any(not re.fullmatch(r"[A-Za-z0-9_-]+", i) for i in ids):
        raise ValueError("Invalid or duplicate segment identifiers")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    semaphore = asyncio.Semaphore(args.concurrency)

    async def render(row):
        name = str(row["id"])
        text = str(row.get("text", "")).strip()
        if not text:
            raise ValueError("Missing reviewed text for " + name)
        # A single a/sa/si must not be interpreted as an English word by the
        # multilingual voice. The spoken prefix is removed, never delivered.
        native_slovak = args.voice.startswith("sk-SK-")
        default_prefix = "Po slovensky poviem: " if args.language == "sk" and not native_slovak and len(text.split()) < 5 else ""
        prefix = str(row.get("tts_prefix", default_prefix))
        if prefix and prefix != "Po slovensky poviem: ":
            raise ValueError("Only the validated Slovak context prefix is supported")
        identity = {"text": text, "language": args.language, "voice": args.voice,
                    "rate": args.rate, "prefix": prefix}
        fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True,
                                               ensure_ascii=False).encode()).hexdigest()
        wav_path = args.output_dir / (name + ".wav")
        meta_path = args.output_dir / (name + ".json")
        if wav_path.exists() and meta_path.exists():
            previous = json.loads(meta_path.read_text())
            if previous.get("fingerprint") == fingerprint and previous.get("sha256") == digest(wav_path):
                print(name + ": reused verified speech WAV", flush=True)
                return {"id": name, **previous}
        async with semaphore:
            start = time.monotonic()
            raw_path = args.output_dir / (name + ".raw.mp3")
            events = []
            for attempt in range(3):
                try:
                    events = []
                    speech = edge_tts.Communicate(
                        prefix + text, args.voice, rate=args.rate,
                        boundary="WordBoundary", proxy=os.environ.get("HTTPS_PROXY"),
                        connect_timeout=15, receive_timeout=45,
                    )
                    with raw_path.open("wb") as stream:
                        async for event in speech.stream():
                            if event["type"] == "audio":
                                stream.write(event["data"])
                            elif event["type"] == "WordBoundary":
                                events.append({k: event[k] for k in ("offset", "duration", "text")})
                    if raw_path.stat().st_size < 100:
                        raise ValueError("Empty service audio")
                    break
                except Exception:
                    if attempt == 2:
                        raise
                    await asyncio.sleep(2 ** attempt)
            cut_start = 0.0
            if prefix:
                expected = ["po", "slovensky", "poviem"]
                if len(events) <= 3 or [clean_word(x["text"]) for x in events[:3]] != expected:
                    raise ValueError("Cannot safely remove spoken language prefix for " + name)
                cut_start = max(0.0, events[3]["offset"] / 10_000_000 - 0.045)
            cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(raw_path)]
            if cut_start:
                cmd += ["-ss", str(cut_start)]
            cmd += ["-ac", "1", "-ar", "24000", "-c:a", "pcm_s16le", str(wav_path)]
            await asyncio.to_thread(subprocess.run, cmd, check=True)
            with wave.open(str(wav_path), "rb") as f:
                frames, rate = f.getnframes(), f.getframerate()
                if rate != 24000 or f.getnchannels() != 1 or frames < 2400:
                    raise ValueError("Invalid synthesized audio for " + name)
            result = {"fingerprint": fingerprint, "sha256": digest(wav_path),
                      "identity": identity, "seconds": frames / rate,
                      "prefix_removed_seconds": cut_start, "boundaries": events,
                      "wall_seconds": time.monotonic() - start}
            write_json(meta_path, result)
            print(f"{name}: {args.language} TTS {frames/rate:.2f}s", flush=True)
            return {"id": name, **result}

    results = await asyncio.gather(*(render(row) for row in rows))
    write_json(args.output_dir / args.manifest_name, {"voice": args.voice, "segments": results})
    print(f"Completed {len(results)} reviewed {args.language} spans", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--segments", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--voice", default="sk-SK-LukasNeural")
    p.add_argument("--language", choices=("sk", "en"), default="sk")
    p.add_argument("--rate", default="+0%")
    p.add_argument("--concurrency", type=int, default=2, choices=range(1, 5))
    p.add_argument("--manifest-name", default="manifest.json")
    asyncio.run(run(p.parse_args()))


if __name__ == "__main__":
    main()
