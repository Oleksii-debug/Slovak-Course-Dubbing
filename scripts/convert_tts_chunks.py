#!/usr/bin/env python3
"""Convert clean foreign-language TTS into the selected dordiai voice.

Consumes already generated ID.mp3/WAV files without calling a speech service.
Use a clean source reference from the same base voice, and inspect pronunciation
after conversion. Timing and finite PCM alone do not prove native pronunciation.
Keep utterances under 30 s.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import time

import numpy as np
import soundfile as sf

from synthesize_spans import LessonVoice, sha256, write_json


def convert_tts(voice, audio, sample_rate, source_embedding):
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    seconds = len(audio) / sample_rate
    if not 0 < seconds <= 30 or not np.isfinite(audio).all():
        raise ValueError("Provide finite utterances between 0 and 30 seconds")
    wave = voice.resample(audio, sample_rate, 22050)
    guard = 2048
    spec = voice.spectrogram(np.pad(wave, (guard, guard))).T.copy()[None]
    latent = voice.flow.run(None, {
        "spec": spec, "spec_lengths": np.array([spec.shape[-1]], np.int64),
        "src_g": source_embedding, "tgt_g": voice.target,
    })[0]
    pieces = []
    for start in range(0, latent.shape[-1], 96):
        end = min(latent.shape[-1], start+96)
        left, right = max(0, start-16), min(latent.shape[-1], end+16)
        decoded = voice.decoder.run(None, {
            "/Mul_output_0": latent[:,:,left:right].copy(),
            "/ConstantOfShape_1_output_0": np.zeros_like(voice.target),
        })[0].reshape(-1)
        pieces.append(decoded[(start-left)*256:(end-left)*256])
    output = np.concatenate(pieces)[guard:guard+len(wave)]
    output = voice.resample(output, 22050, 24000)
    expected = round(len(audio) / sample_rate * 24000)
    output = np.pad(output, (0, max(0, expected-len(output))))[:expected]
    if not np.isfinite(output).all() or not len(output) or abs(output).max() < 1e-6:
        raise ValueError("Invalid or silent projected speech")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--segments", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-reference", type=Path, required=True)
    parser.add_argument("--extension", choices=(".mp3", ".wav"), default=".mp3")
    parser.add_argument("--source-voice", default="sk-SK-LukasNeural")
    parser.add_argument("--language", choices=("sk", "en"), help="select one language when base voices differ")
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--extras", type=Path, required=True)
    parser.add_argument("--profile", default="oleksii_day1")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.input_dir.resolve() == args.output_dir.resolve():
        raise ValueError("Output must not replace source TTS chunks")
    raw = json.loads(args.segments.read_text(encoding="utf-8"))
    rows = raw["segments"] if isinstance(raw, dict) else raw
    rows = [s for s in rows if s["language"] in {"sk", "en"}]
    if args.language:
        rows = [s for s in rows if s["language"] == args.language]
    if not rows:
        raise ValueError("No reviewed foreign-language segments")
    seen = set()
    for segment in rows:
        name = str(segment["id"])
        if not re.fullmatch(r"[A-Za-z0-9_-]+", name) or name in seen:
            raise ValueError("Invalid or repeated segment identifier")
        seen.add(name)
        if not segment.get("text", "").strip():
            raise ValueError("A reviewed foreign-language transcript is required")
        if segment.get("source_language", segment["language"]) not in {"sk", "en"}:
            raise ValueError("Unexpected source language")
        if not (args.input_dir / (name+args.extension)).is_file():
            raise FileNotFoundError(args.input_dir / (name+args.extension))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_files = [args.models / "OpenVoiceV2" / name for name in
                   ("tone_ref_encoder.onnx", "encoder-flow.onnx", "decoder.onnx")]
    model_files.append(args.models / "voices/classmate_locales" / (args.profile+".npy"))
    identity = {"profile": args.profile, "source_tts_voice": args.source_voice,
                "source_reference_sha256": sha256(args.source_reference),
                "script_sha256": sha256(__file__),
                "base_adapter_sha256": sha256(Path(__file__).with_name("synthesize_spans.py")),
                "models": {str(p.relative_to(args.models)): sha256(p) for p in model_files},
                "backend": "clean-foreign-tts-to-dordiai-openvoice-whole-utterance"}
    path = args.output_dir / "manifest.json"
    previous = json.loads(path.read_text()) if path.exists() else {}
    records = previous.get("segments", {}) if previous.get("identity") == identity else {}
    voice = LessonVoice(args.runtime, args.models, args.extras, args.profile, args.threads)
    reference, rate = sf.read(args.source_reference, dtype="float32")
    embedding = voice.speaker_embedding(reference, rate)
    np.save(args.output_dir / "source_tts_embedding.npy", embedding, allow_pickle=False)
    for number, segment in enumerate(rows, start=1):
        name = str(segment["id"])
        source = args.input_dir / (name+args.extension)
        output = args.output_dir / (name+".wav")
        source_hash = sha256(source)
        segment_hash = hashlib.sha256(json.dumps(segment,ensure_ascii=False,sort_keys=True).encode()).hexdigest()
        old = records.get(name, {})
        if (old.get("source_tts_sha256") == source_hash and old.get("segment_sha256") == segment_hash
                and output.is_file() and sha256(output) == old.get("wav_sha256")):
            print(f"{number}/{len(rows)} {name}: reused verified TTS projection", flush=True)
            continue
        begin = time.monotonic()
        audio, rate = sf.read(source, dtype="float32")
        projected = convert_tts(voice, audio, rate, embedding)
        peak = float(abs(projected).max())
        gain = min(1.0, 0.96/peak)
        sf.write(output, projected*gain, 24000, subtype="PCM_16")
        records[name] = {"segment_sha256": segment_hash, "source_tts_sha256": source_hash,
                         "wav_sha256": sha256(output), "mode": "tts_convert",
                         "language": segment["language"], "source_language": segment.get("source_language",segment["language"]),
                         "source_start": segment["start"], "source_end": segment["end"],
                         "text": segment["text"], "output_seconds": len(projected)/24000,
                         "sample_rate": 24000, "peak_before_gain": peak, "gain": gain,
                         "wall_seconds": time.monotonic()-begin,
                         "pronunciation_reviewed": False}
        write_json(path, {"identity": identity, "segments": records,
                          "sampled_route_intelligibility_checked": False,
                          "every_chunk_pronunciation_reviewed": False})
        print(f"{number}/{len(rows)} {name}: TTS projected {len(projected)/24000:.2f}s", flush=True)
    print(f"Converted {len(rows)} foreign-language TTS chunks", flush=True)


if __name__ == "__main__":
    main()
