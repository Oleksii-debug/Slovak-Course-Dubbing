"""Real ffmpeg integration checks using small generated PCM tones."""

import importlib.util
import json
import math
from pathlib import Path
import shutil
import struct
import subprocess
import tempfile
import unittest
import wave


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "assemble_lesson.py"
SPEC = importlib.util.spec_from_file_location("assemble_lesson", SCRIPT)
assembler = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(assembler)


def tone(path, duration, frequency):
    with wave.open(str(path), "wb") as writer:
        writer.setparams((1, 2, 24000, 0, "NONE", "not compressed"))
        writer.writeframes(b"".join(struct.pack("<h", round(8000 * math.sin(2 * math.pi * frequency * n / 24000))) for n in range(round(duration * 24000))))


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "ffmpeg and ffprobe required")
class AssemblyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        tone(self.root / "first.wav", 0.2, 440)
        tone(self.root / "second.wav", 0.5, 880)
        tone(self.root / "third.wav", 0.1, 1320)
        self.spans = [
            {"id": "first", "start": 0.1, "end": 0.4, "language": "uk", "text": "Перше.", "audio_path": "first.wav"},
            {"id": "second", "start": 0.7, "end": 0.9, "language": "sk", "text": "Druhé.", "audio_path": "second.wav"},
            {"id": "third", "start": 1.0, "end": 1.1, "language": "en", "text": "Third.", "audio_path": "third.wav"},
        ]
        self.manifest = self.root / "manifest.json"

    def write_manifest(self):
        self.manifest.write_text(json.dumps(self.spans, ensure_ascii=False), encoding="utf-8")

    def test_preserves_order_gaps_and_expands_speech_without_truncation(self):
        self.write_manifest()
        output = self.root / "lesson.mp3"
        report = assembler.assemble(self.manifest, 1.4, output, self.root / "timeline.json")
        first, second, third = report["segments"]
        self.assertEqual([x["id"] for x in report["segments"]], ["first", "second", "third"])
        self.assertEqual(report["segment_count"], 3)
        self.assertTrue(report["segment_ids_verified_once"])
        self.assertAlmostEqual(first["output_start"], 0.1)
        self.assertAlmostEqual(first["output_audio_end"], 0.3)
        self.assertAlmostEqual(first["output_end"], 0.4)
        self.assertAlmostEqual(first["padding_after"], 0.1)
        self.assertAlmostEqual(second["gap_before"], 0.3)
        self.assertAlmostEqual(second["output_start"], 0.7)
        self.assertAlmostEqual(second["output_audio_end"], 1.2)
        self.assertAlmostEqual(second["expansion"], 0.3)
        self.assertAlmostEqual(third["gap_before"], 0.1)
        self.assertAlmostEqual(third["output_start"], 1.3)
        self.assertAlmostEqual(third["output_audio_end"], 1.4)
        self.assertAlmostEqual(report["trailing_silence"], 0.3)
        self.assertAlmostEqual(report["timeline_duration"], 1.7)
        self.assertAlmostEqual(report["total_expansion"], 0.3)
        # Decode the delivered MP3, not merely its metadata. Verify leading and
        # inter-speech silence and that different tones retain their order.
        decoded = self.root / "decoded.wav"
        subprocess.run(["ffmpeg", "-v", "error", "-i", str(output), "-c:a", "pcm_s16le", str(decoded)], check=True)
        with wave.open(str(decoded), "rb") as reader:
            samples = struct.unpack("<" + "h" * reader.getnframes(), reader.readframes(reader.getnframes()))
        self.assertEqual(len(samples), 40800)
        for left, right in [(0.01, 0.07), (0.45, 0.65), (1.5, 1.65)]:
            clip = samples[round(left * 24000):round(right * 24000)]
            self.assertLess(max(abs(s) for s in clip), 30)
        for left, right, expected_frequency in [(0.14, 0.26, 440), (0.8, 1.1, 880), (1.32, 1.38, 1320)]:
            clip = samples[round(left * 24000):round(right * 24000)]
            crossings = sum(a <= 0 < b for a, b in zip(clip, clip[1:]))
            self.assertAlmostEqual(crossings / (right - left), expected_frequency, delta=12)

    def test_invalid_intervals_and_duplicate_ids_are_rejected_before_output(self):
        cases = [
            ("overlap", lambda: self.spans[1].update(start=0.3)),
            ("out of range", lambda: self.spans[1].update(end=2.0)),
            ("negative", lambda: self.spans[0].update(start=-0.1)),
            ("zero interval", lambda: self.spans[0].update(end=0.1)),
            ("nan", lambda: self.spans[0].update(start=float("nan"))),
            ("duplicate", lambda: self.spans[1].update(id="first")),
        ]
        original = json.dumps(self.spans)
        for name, mutate in cases:
            with self.subTest(name=name):
                self.spans = json.loads(original)
                mutate()
                self.write_manifest()
                with self.assertRaises(assembler.AssemblyError):
                    assembler.assemble(self.manifest, 1.4, self.root / "bad.mp3", self.root / "bad.json")
                self.assertFalse((self.root / "bad.mp3").exists())

    def test_audio_only_timing_keeps_pauses_without_inventing_padding(self):
        self.write_manifest()
        report = assembler.assemble(self.manifest, 1.4, self.root / "natural.mp3",
                                    self.root / "natural.json", pad_shorter=False)
        first, second, third = report["segments"]
        self.assertAlmostEqual(first["padding_after"], 0.0)
        self.assertAlmostEqual(first["output_end"], 0.3)
        self.assertAlmostEqual(second["gap_before"], 0.3)
        self.assertAlmostEqual(second["output_start"], 0.6)
        self.assertAlmostEqual(second["audio_duration"], 0.5)
        self.assertAlmostEqual(third["output_start"], 1.2)
        self.assertAlmostEqual(report["trailing_silence"], 0.3)
        self.assertAlmostEqual(report["timeline_duration"], 1.6)
        self.assertEqual(report["timing_policy"], "natural_speech_preserve_source_gaps")

    def test_lossless_intermediate_preserves_every_pcm_sample(self):
        self.write_manifest()
        output = self.root / "speech.flac"
        report = assembler.assemble(self.manifest, 1.4, output,
                                    self.root / "lossless.json", pad_shorter=False)
        decoded = subprocess.run(["ffmpeg", "-v", "error", "-i", str(output),
                                  "-f", "s16le", "pipe:1"],
                                 capture_output=True, check=True).stdout
        def pcm(name):
            with wave.open(str(self.root / name), "rb") as reader:
                return reader.readframes(reader.getnframes())
        silence = lambda seconds: b"\0" * (round(seconds * 24000) * 2)
        expected = (silence(.1) + pcm("first.wav") + silence(.3) +
                    pcm("second.wav") + silence(.1) + pcm("third.wav") + silence(.3))
        self.assertEqual(decoded, expected)
        self.assertEqual(report["format"], "flac")


if __name__ == "__main__":
    unittest.main()
