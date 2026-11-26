#!/usr/bin/env python3
"""Render reviewed UK text and voice-convert original SK/EN lesson spans.

Uses an existing private dordiai-voices runtime/model bundle. Never downloads
models, uploads audio, calls a paid API, or modifies the supplied resources.
Output is per-span 24 kHz PCM WAV plus a resumable manifest. This is not an
automatic transcript/translation checker or an auditory quality certification.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
import time


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


class LessonVoice:
    def __init__(self, runtime, models, extras, profile="oleksii_day1", threads=2):
        sys.path.insert(0, str(Path(extras).resolve()))
        sys.path.insert(0, str(Path(runtime).resolve()))
        import numpy as np
        import onnxruntime as ort
        from converted_voices import spectrogram, resample
        from ukrainian_voices import UkrainianVoices
        from voice_policy import require_voice_allowed

        require_voice_allowed(profile)
        ort.disable_telemetry_events()
        self.np, self.resample, self.spectrogram = np, resample, spectrogram
        self.profile, self.models = profile, Path(models)
        self.ukrainian = UkrainianVoices(self.models, self.models / "voices", threads=threads)
        options = ort.SessionOptions()
        options.intra_op_num_threads, options.inter_op_num_threads = threads, 1
        options.add_session_config_entry("session.intra_op.allow_spinning", "0")
        def session(name):
            return ort.InferenceSession(str(self.models / "OpenVoiceV2" / name), options,
                                        providers=["CPUExecutionProvider"])
        self.encoder = session("tone_ref_encoder.onnx")
        self.flow = session("encoder-flow.onnx")
        self.decoder = session("decoder.onnx")
        self.target = np.load(self.models / "voices/classmate_locales" / (profile + ".npy"), allow_pickle=False)
        if self.target.shape != (1, 256, 1) or not np.isfinite(self.target).all():
            raise ValueError("Invalid selected voice identity")

    def say_ukrainian(self, text):
        return self.ukrainian.synthesize(self.profile, "uk", text, speed=1.0)

    def speaker_embedding(self, audio, sample_rate):
        """Mean raw conditioning vector; never unit-normalize OpenVoice vectors."""
        np = self.np
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        wave = self.resample(audio, int(sample_rate), 22050)
        if wave.size < 22050 or not np.isfinite(wave).all() or abs(wave).max() < 1e-5:
            raise ValueError("Use at least one second of nonsilent clean teacher speech")
        windows = [wave[start:start + 8*22050]
                   for start in range(0, min(len(wave), 40*22050), 8*22050)]
        embeddings = [self.encoder.run(None, {"spec": self.spectrogram(w)[None]})[0].reshape(1,256,1)
                      for w in windows if len(w) >= 22050]
        value = np.mean(embeddings, axis=0).astype(np.float32)
        if not np.isfinite(value).all() or np.linalg.norm(value) < 1e-6:
            raise ValueError("Invalid teacher conditioning vector")
        return value

    def convert_original(self, audio, sample_rate, source_embedding):
        """Preserve utterance and duration, project through selected target voice."""
        np = self.np
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        original_length = len(audio)
        wave = self.resample(audio, int(sample_rate), 22050)
        guard = 2048
        spec = self.spectrogram(np.pad(wave, (guard, guard))).T.copy()[None]
        frames = spec.shape[-1]
        pieces = []
        # Context overlap bounds memory for long preserved-language spans.
        for group_start in range(0, frames, 288):
            group_end = min(frames, group_start + 288)
            lo, hi = max(0, group_start-40), min(frames, group_end+40)
            latent = self.flow.run(None, {
                "spec": spec[:,:,lo:hi].copy(),
                "spec_lengths": np.array([hi-lo], np.int64),
                "src_g": np.asarray(source_embedding, np.float32), "tgt_g": self.target,
            })[0]
            for start in range(group_start, group_end, 96):
                end = min(group_end, start + 96)
                left, right = max(lo,start-16), min(hi,end+16)
                decoded = self.decoder.run(None, {
                    "/Mul_output_0": latent[:,:,left-lo:right-lo].copy(),
                    "/ConstantOfShape_1_output_0": np.zeros_like(self.target),
                })[0].reshape(-1)
                pieces.append(decoded[(start-left)*256:(end-left)*256])
        result = np.concatenate(pieces)[guard:guard+len(wave)]
        result = self.resample(result, 22050, 24000)
        expected = round(original_length / int(sample_rate) * 24000)
        result = np.pad(result, (0,max(0,expected-len(result))))[:expected]
        if not np.isfinite(result).all():
            raise ValueError("Invalid converted waveform")
        return result, 24000


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--segments", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--extras", type=Path, required=True,
                        help="Pure Python holos_onnx/ipa_uk/ukrainian_word_stress package directory")
    parser.add_argument("--profile", default="oleksii_day1")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--reference-start", type=float, default=16.14)
    parser.add_argument("--reference-end", type=float, default=50.0)
    args = parser.parse_args()
    import numpy as np
    import soundfile as sf

    source, sample_rate = sf.read(args.source, dtype="float32", always_2d=False)
    duration = len(source)/sample_rate
    raw = json.loads(args.segments.read_text(encoding="utf-8"))
    segments = raw["segments"] if isinstance(raw, dict) else raw
    if not isinstance(segments, list) or not segments:
        raise ValueError("Provide a nonempty reviewed list of segments")
    seen = set()
    for segment in segments:
        name = str(segment["id"])
        if not re.fullmatch(r"[A-Za-z0-9_-]+", name) or name in seen:
            raise ValueError("Segment identifiers must be unique safe filenames")
        seen.add(name)
        if not 0 <= float(segment["start"]) < float(segment["end"]) <= duration + 0.001:
            raise ValueError("Segment timestamps are outside the source")
        if segment["mode"] == "tts":
            if segment["language"] != "uk" or not segment.get("text", "").strip():
                raise ValueError("TTS segments require Ukrainian output text")
        elif segment["mode"] == "convert":
            if segment["language"] not in {"sk", "en"} or segment.get("source_language", segment["language"]) not in {"sk", "en"}:
                raise ValueError("Only reviewed SK/EN originals may remain in voice conversion")
        else:
            raise ValueError("Unsupported rendering mode")
    if not 0 <= args.reference_start < args.reference_end <= duration:
        raise ValueError("Teacher enrollment interval is outside source")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    identity_files = [args.models / "OpenVoiceV2" / name for name in
                      ("tone_ref_encoder.onnx", "encoder-flow.onnx", "decoder.onnx")]
    identity_files += [args.models / "Holos" / name for name in
                       ("encoder.onnx", "prosody.onnx", "source.onnx", "decoder_window.onnx")]
    identity_files += [args.models / "voices" / group / (args.profile + ".npy")
                       for group in ("holos", "classmate_locales")]
    identity = {"profile": args.profile, "source_sha256": sha256(args.source),
                "script_sha256": sha256(__file__),
                "models": {str(p.relative_to(args.models)): sha256(p) for p in identity_files},
                "teacher_reference": [args.reference_start, args.reference_end]}
    manifest_path = args.output_dir / "manifest.json"
    previous = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    records = previous.get("segments", {}) if previous.get("identity") == identity else {}
    voice = LessonVoice(args.runtime, args.models, args.extras, args.profile, args.threads)
    reference = source[round(args.reference_start*sample_rate):round(args.reference_end*sample_rate)]
    teacher = voice.speaker_embedding(reference, sample_rate)
    np.save(args.output_dir / "teacher_embedding.npy", teacher, allow_pickle=False)
    for number, segment in enumerate(segments, start=1):
        name = str(segment["id"])
        path = args.output_dir / (name + ".wav")
        fingerprint = hashlib.sha256(json.dumps(segment,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
        prior = records.get(name, {})
        if (prior.get("segment_sha256") == fingerprint and path.is_file()
                and sha256(path) == prior.get("wav_sha256")):
            print(f"{number}/{len(segments)} {name}: reused verified WAV", flush=True)
            continue
        begin = time.monotonic()
        if segment["mode"] == "tts":
            wave, rate = voice.say_ukrainian(segment["text"])
        else:
            clip = source[round(segment["start"]*sample_rate):round(segment["end"]*sample_rate)]
            wave, rate = voice.convert_original(clip, sample_rate, teacher)
        if wave.ndim != 1 or rate != 24000 or not len(wave) or not np.isfinite(wave).all():
            raise ValueError("Invalid output for " + name)
        peak = float(abs(wave).max())
        if peak < 1e-6:
            raise ValueError("Silent output for " + name)
        # Preserve signal ratios; only prevent integer PCM clipping.
        gain = min(1.0, 0.96 / peak)
        sf.write(path, wave * gain, rate, subtype="PCM_16")
        records[name] = {"segment_sha256": fingerprint, "wav_sha256": sha256(path),
                         "mode": segment["mode"], "language": segment["language"],
                         "source_start": segment["start"], "source_end": segment["end"],
                         "output_seconds": len(wave)/rate, "sample_rate": rate,
                         "peak_before_gain": peak, "gain": gain,
                         "wall_seconds": time.monotonic()-begin,
                         "text": segment.get("text", ""),
                         "pronunciation_reviewed": False}
        write_json(manifest_path, {"identity": identity, "segments": records,
                                  "automatic_audio_checks_only": True})
        print(f"{number}/{len(segments)} {name}: {segment['mode']} {len(wave)/rate:.2f}s", flush=True)
    print(f"Rendered {len(segments)} reviewed spans to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
