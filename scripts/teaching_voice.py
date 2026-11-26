#!/usr/bin/env python3
"""Render reviewed Ukrainian lessons with explicit stress and auditable pauses.

Uses the existing private dordiai Holos speaker profile. No model downloads,
new speaker training, arbitrary pitch shift, or identity substitution. The
profile is a speaker identity; this adapter is not a trained "teacher style".
Its measured changes are punctuation timing and token-aligned noise removal.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field, replace
import hashlib
import json
from pathlib import Path
import re
import sys
import time
import unicodedata


ACUTE = "\u0301"
DEFAULT_PAUSES_MS = {",": 225, ";": 350, ":": 350, "—": 300,
                     ".": 400, "?": 450, "!": 400, "…": 500}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


@dataclass(frozen=True)
class TeachingSettings:
    pauses_ms: dict = field(default_factory=lambda: dict(DEFAULT_PAUSES_MS))
    final_pause_ms: float = 125.0
    boundary_margin_ms: float = 20.0
    punctuation_margin_ms: float = 60.0
    fade_ms: float = 5.0
    max_quiet_frame_rms: float = 0.005

    def validate(self):
        if set(self.pauses_ms) - set(DEFAULT_PAUSES_MS):
            raise ValueError("Only supported punctuation pause tokens may be changed")
        if any(not 100 <= float(v) <= 1500 for v in self.pauses_ms.values()):
            raise ValueError("Punctuation pauses must be between 100 and 1500 ms")
        if not 75 <= self.final_pause_ms <= 400:
            raise ValueError("The final span pause must be between 75 and 400 ms")
        if not 10 <= self.boundary_margin_ms <= 60:
            raise ValueError("Keep at least 10 ms of phonetic boundary context")
        if not 40 <= self.punctuation_margin_ms <= 100:
            raise ValueError("Keep 40–100 ms of phonetic context around punctuation")
        if not 1 <= self.fade_ms <= 10:
            raise ValueError("Use a 1–10 ms edge fade")
        if not 0.0001 <= self.max_quiet_frame_rms <= 0.01:
            raise ValueError("Punctuation energy guard must remain conservative")


class TeachingDurationEncoder:
    """Set punctuation timing while preserving every speech-token duration."""
    def __init__(self, session, tokenizer, frame_ms, settings):
        self.session, self.tokenizer = session, tokenizer
        self.frame_ms, self.settings = frame_ms, settings
        self.final_sentence = False
        self.preserve_predicted_pauses = False

    def __getattr__(self, key):
        return getattr(self.session, key)

    def run(self, outputs, inputs):
        import numpy as np
        acoustic, prosodic, durations = self.session.run(outputs, inputs)
        durations = np.asarray(durations).copy()
        # A learned punctuation token can contain voiced articulation. Extending
        # it is not equivalent to inserting silence. Colon/dash spans use the
        # ordinary comma boundary and the source runtime's predicted durations.
        if self.preserve_predicted_pauses:
            return acoustic, prosodic, durations
        tokens = inputs["tokens"][0]
        for symbol, ms in self.settings.pauses_ms.items():
            token = self.tokenizer._token_to_id.get(symbol)
            if token is not None:
                durations[0, tokens == token] = max(1, round(ms / self.frame_ms))
        # The assembler retains the original gap between teaching turns. Do
        # not add a full sentence pause at that same boundary a second time.
        # Other sentences inside this one span keep their explanatory pauses.
        if self.final_sentence:
            closing = set(' $"«»“”()')
            for index in range(len(tokens) - 1, -1, -1):
                symbol = self.tokenizer.vocab[int(tokens[index])]
                if symbol in closing:
                    continue
                if symbol in self.settings.pauses_ms:
                    durations[0, index] = max(1, round(self.settings.final_pause_ms / self.frame_ms))
                break
        return acoustic, prosodic, durations


def quiet_aligned_pauses(audio, tokens, durations, vocab, sample_rate,
                         samples_per_frame, settings):
    """Silence only the centers of model-labeled non-speech tokens.

    No amplitude gate on speech phonemes or cutting of initial fricatives.
    Context margins preserve articulation either side of punctuation. This
    cannot remove artifacts located inside a speech phoneme. A model's
    punctuation alignment is not perfect: any energetic 20 ms frame in a
    proposed punctuation center leaves the whole center untouched.
    """
    import numpy as np
    output = np.asarray(audio, dtype=np.float32).copy()
    edges = np.concatenate(([0], np.cumsum(durations))) * samples_per_frame
    margin = round(settings.punctuation_margin_ms * sample_rate / 1000)
    boundary_margin = round(settings.boundary_margin_ms * sample_rate / 1000)
    fade = round(settings.fade_ms * sample_rate / 1000)
    records = []
    for index, token in enumerate(tokens):
        symbol = vocab[int(token)]
        if token != 0 and symbol not in settings.pauses_ms:
            continue
        left, right = int(edges[index]), int(edges[index + 1])
        if index == 0:
            lo, hi = left, max(left, right - boundary_margin)
        elif index == len(tokens) - 1:
            lo, hi = min(right, left + boundary_margin), right
        else:
            lo, hi = left + margin, right - margin
        if hi <= lo + 2 * fade:
            continue
        before = output[lo:hi].copy()
        before_rms = float(np.sqrt(np.mean(before ** 2)))
        if token != 0:
            frame = max(1, round(sample_rate * .020))
            padded = np.pad(before, (0, (-len(before)) % frame))
            peak_frame_rms = float(np.sqrt(np.mean(padded.reshape(-1, frame) ** 2, axis=1)).max())
            if peak_frame_rms > settings.max_quiet_frame_rms:
                records.append({"token_index": index, "token": symbol,
                                "start_ms": lo / sample_rate * 1000,
                                "end_ms": hi / sample_rate * 1000,
                                "action": "preserved_high_energy",
                                "before_rms": before_rms, "after_rms": before_rms,
                                "max_frame_rms": peak_frame_rms,
                                "guard_rms": settings.max_quiet_frame_rms})
                continue
        output[lo:hi] = 0
        if index != 0:
            output[lo:lo + fade] = before[:fade] * np.linspace(1, 0, fade)
        if index != len(tokens) - 1:
            output[hi - fade:hi] = before[-fade:] * np.linspace(0, 1, fade)
        records.append({"token_index": index, "token": symbol,
                        "start_ms": lo / sample_rate * 1000,
                        "end_ms": hi / sample_rate * 1000,
                        "action": "silenced_token_center",
                        "before_rms": before_rms,
                        "after_rms": float(np.sqrt(np.mean(output[lo:hi] ** 2)))})
    # The leading/trailing boundary token is padding, not a spoken token.
    # Keep the specified context rather than using amplitude-dependent trims.
    begin = max(0, int(edges[1]) - boundary_margin)
    end = min(len(output), int(edges[-2]) + boundary_margin)
    if end <= begin:
        raise ValueError("Model produced no speech between boundary tokens")
    return output[begin:end], records, begin, end


class TeachingVoice:
    def __init__(self, runtime, models, extras, profile="oleksii_day1", threads=2,
                 settings=None):
        sys.path.insert(0, str(Path(extras).resolve()))
        sys.path.insert(0, str(Path(runtime).resolve()))
        import onnxruntime as ort
        from ukrainian_voices import UkrainianVoices
        from voice_policy import require_voice_allowed
        require_voice_allowed(profile)
        ort.disable_telemetry_events()
        self.settings = settings or TeachingSettings()
        self.settings.validate()
        self.profile, self.models = profile, Path(models)
        self.engine = UkrainianVoices(self.models, self.models / "voices", threads=threads)
        self.model = self.engine.holos
        self.model.encoder_session = TeachingDurationEncoder(
            self.model.encoder_session, self.model.tokenizer,
            self.model.samples_per_frame / self.model.sample_rate * 1000, self.settings)

    def prepare_text(self, text):
        from speech_text import speak_numbers
        from ukrainian_stress import accent_text, WORDS, VOWELS
        expanded = re.sub(r"\s+", " ", speak_numbers(text, "uk")).strip()
        expanded = re.sub(r"\.{3,}", "…", expanded)
        if re.search(r"[A-Za-z]", expanded):
            raise ValueError("Separate foreign-language examples from Ukrainian speech")
        marked = accent_text(expanded)
        audit = []
        original_words = list(WORDS.finditer(expanded))
        marked_words = list(WORDS.finditer(marked))
        if len(original_words) != len(marked_words):
            raise ValueError("Stress preparation changed the word count")
        for original, prepared in zip(original_words, marked_words):
            word = prepared.group()
            if ACUTE in original.group():
                original_word = original.group().lower()
                if (original_word.count(ACUTE) != 1 or original_word.index(ACUTE) == 0
                        or original_word[original_word.index(ACUTE) - 1] not in VOWELS):
                    raise ValueError("Place exactly one manual stress mark after a vowel: " + original.group())
            if sum(c in VOWELS for c in word.lower()) < 2:
                continue
            audit.append({"word": original.group(), "marked": word,
                          "status": "manual" if ACUTE in original.group() else
                                    "dictionary" if ACUTE in word else "unresolved"})
        return marked, audit

    def say_ukrainian(self, text):
        import numpy as np
        from ipa_uk import ipa
        from ukrainian_voices import SENTENCES
        from voice_delivery import condition_shared
        marked, stress_audit = self.prepare_text(text)
        natural_boundaries = bool(re.search(r"[:—]", marked))
        synthesis_text = re.sub(r"\s*[:—]\s*", ", ", marked) if natural_boundaries else marked
        # This changes editorial punctuation only. Keep every reviewed word,
        # manual stress mark and speaker profile; preserve coarticulation by
        # synthesizing the complete sentence rather than isolated fragments.
        lexical = lambda value: re.findall(r"[\w’']+", value, re.UNICODE)
        if lexical(synthesis_text) != lexical(marked):
            raise ValueError("Punctuation normalization changed reviewed words")
        pause_settings = replace(self.settings, pauses_ms={}) if natural_boundaries else self.settings
        self.model.encoder_session.preserve_predicted_pauses = natural_boundaries
        clips, sentences = [], []
        sentence_matches = list(SENTENCES.finditer(synthesis_text))
        for sentence_index, sentence in enumerate(sentence_matches):
            written = sentence.group().strip()
            phonemes = ipa(written)
            if not phonemes.strip(" .,!?:;…—"):
                continue
            self.model.encoder_session.final_sentence = sentence_index == len(sentence_matches) - 1
            result = self.model.generate_with_details(
                phonemes, voice=self.engine._holos_style(self.profile), speed=1.0)
            tokens = self.model.tokenizer.encode_with_boundaries(phonemes)
            if len(tokens) != len(result.durations):
                raise ValueError("Cannot bind audio to phoneme tokens")
            wave, silenced, begin, end = quiet_aligned_pauses(
                result.audio, tokens, result.durations, self.model.tokenizer.vocab,
                result.sample_rate, self.model.samples_per_frame, pause_settings)
            delivery = self.engine.delivery_profiles.get(self.profile)
            if delivery:
                wave = np.concatenate(list(condition_shared(
                    (wave,), delivery["gain"], delivery.get("quiet", 0))))
            if not np.isfinite(wave).all() or len(wave) == 0:
                raise ValueError("Holos returned invalid teaching audio")
            offset = sum(len(c) for c in clips)
            sentences.append({"text": written, "phonemes": phonemes,
                              "output_start": offset / result.sample_rate,
                              "output_seconds": len(wave) / result.sample_rate,
                              "durations_frames": result.durations.tolist(),
                              "token_symbols": self.model.tokenizer.decode(tokens, as_list=True),
                              "quiet_regions_before_boundary_trim": [r for r in silenced
                                  if r["action"] == "silenced_token_center"],
                              "preserved_high_energy_regions_before_boundary_trim": [r for r in silenced
                                  if r["action"] == "preserved_high_energy"],
                              "boundary_trim_samples": [begin, len(result.audio) - end]})
            clips.append(wave)
        if not clips:
            raise ValueError("No Ukrainian speech to render")
        self.last_report = {"marked_text": marked, "synthesis_text": synthesis_text,
                            "punctuation_strategy": "comma_boundaries_with_predicted_durations"
                                if natural_boundaries else "reviewed_token_durations",
                            "word_stress": stress_audit,
                            "unresolved_words": sorted(set(x["word"] for x in stress_audit
                                                           if x["status"] == "unresolved")),
                            "sentences": sentences,
                            "human_pronunciation_accepted": False,
                            "trained_teacher_style": False}
        return np.concatenate(clips), self.model.sample_rate


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--segments", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--runtime", required=True, type=Path)
    p.add_argument("--models", required=True, type=Path)
    p.add_argument("--extras", required=True, type=Path)
    p.add_argument("--profile", default="oleksii_day1")
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--settings", type=Path)
    args = p.parse_args()
    import numpy as np
    import soundfile as sf
    raw = json.loads(args.segments.read_text(encoding="utf-8"))
    rows = raw["segments"] if isinstance(raw, dict) else raw
    rows = [r for r in rows if r["language"] == "uk"]
    if not rows:
        raise ValueError("Provide reviewed Ukrainian spans")
    names = [str(r["id"]) for r in rows]
    if len(set(names)) != len(names) or any(not re.fullmatch(r"[A-Za-z0-9_-]+", n) for n in names):
        raise ValueError("Invalid or repeated span identifier")
    settings = TeachingSettings(**json.loads(args.settings.read_text())) if args.settings else TeachingSettings()
    settings.validate()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_files = [args.models / "Holos" / n for n in
                   ("encoder.onnx", "prosody.onnx", "source.onnx", "decoder_window.onnx")]
    model_files += [args.models / "voices/holos" / (args.profile + ".npy")]
    model_files += [args.models / "voices" / n for n in
                    ("delivery_profiles.json", "reference_prosody.json", "ukrainian_articulation.json")
                    if (args.models / "voices" / n).is_file()]
    identity = {"profile": args.profile, "backend": "native-holos-reviewed-teaching-pauses",
                "settings": vars(settings), "script_sha256": digest(__file__),
                "models": {str(f.relative_to(args.models)): digest(f) for f in model_files},
                "runtime_sources": {n: digest(args.runtime / n) for n in
                                    ("ukrainian_voices.py", "ukrainian_stress.py", "voice_delivery.py",
                                     "reference_prosody.py", "speech_text.py")}}
    manifest_path = args.output_dir / "manifest.json"
    previous = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    records = previous.get("segments", {}) if previous.get("identity") == identity else {}
    voice = TeachingVoice(args.runtime, args.models, args.extras, args.profile, args.threads, settings)
    for row in rows:
        name = str(row["id"])
        if row.get("tts_text") and row.get("text_stressed") and row["tts_text"] != row["text_stressed"]:
            raise ValueError("Conflicting manual-stress fields for " + name)
        text = str(row.get("tts_text", row.get("text_stressed", row["text"]))).strip()
        # tts_text may add stress marks, but must not secretly change the lesson.
        strip_stress = lambda x: re.sub(r"\s+", " ", unicodedata.normalize("NFC", x).replace(ACUTE, "")).strip()
        if strip_stress(text) != strip_stress(row["text"]):
            raise ValueError("tts_text may add stress marks only; edit reviewed text for content changes")
        fingerprint = hashlib.sha256(json.dumps(row, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        path = args.output_dir / (name + ".wav")
        old = records.get(name, {})
        if old.get("segment_sha256") == fingerprint and path.is_file() and digest(path) == old.get("wav_sha256"):
            print(name + ": reused verified teaching WAV", flush=True)
            continue
        begin = time.monotonic()
        wave, rate = voice.say_ukrainian(text)
        peak = float(np.max(np.abs(wave)))
        gain = min(1.0, .96 / max(peak, 1e-8))
        sf.write(path, wave * gain, rate, subtype="PCM_16")
        report = {"segment_sha256": fingerprint, "wav_sha256": digest(path),
                  "mode": "tts", "language": "uk", "text": row["text"],
                  "output_seconds": len(wave) / rate, "sample_rate": rate,
                  "gain": gain, "wall_seconds": time.monotonic() - begin,
                  **voice.last_report}
        write_json(args.output_dir / (name + ".json"), report)
        records[name] = report
        write_json(manifest_path, {"identity": identity, "segments": records})
        print(f"{name}: {len(wave)/rate:.2f}s; unresolved stress: {report['unresolved_words']}", flush=True)


if __name__ == "__main__":
    main()
