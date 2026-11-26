"""Safety properties of noise cleanup at generated speech boundaries."""
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import numpy as np


path = Path(__file__).resolve().parents[1] / "scripts" / "teaching_voice.py"
spec = importlib.util.spec_from_file_location("teaching_voice", path)
teaching = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = teaching
spec.loader.exec_module(teaching)


class BoundaryCleanupTests(unittest.TestCase):
    def test_natural_boundary_mode_never_extends_learned_punctuation(self):
        # A short learned comma may contain articulation. The production fix
        # must not turn it into a longer voiced sound by overriding duration.
        vocabulary = ["$", "a", ",", "."]
        durations = np.array([[2, 5, 3, 7, 1, 2]], dtype=np.int64)
        session = SimpleNamespace(run=lambda outputs, inputs: (
            np.zeros((1, 6, 1)), np.zeros((1, 6, 1)), durations.copy()))
        tokenizer = SimpleNamespace(vocab=vocabulary,
            _token_to_id={symbol: index for index, symbol in enumerate(vocabulary)})
        encoder = teaching.TeachingDurationEncoder(
            session, tokenizer, 25.0, teaching.TeachingSettings())
        encoder.preserve_predicted_pauses = True
        encoder.final_sentence = True
        _, _, actual = encoder.run(None, {"tokens": np.array([[0, 1, 2, 1, 3, 0]])})
        np.testing.assert_array_equal(actual, durations)

    def test_initial_fricative_and_speech_samples_are_untouched(self):
        # Noise-like /sh/ is deliberately as quiet as padding; amplitude gating
        # would wrongly erase it. The cleanup must use phoneme alignment only.
        rng = np.random.default_rng(728)
        durations = np.array([8, 8, 5, 16, 6, 7, 8])
        tokens = np.array([0, 1, 2, 3, 2, 1, 0])
        vocab = ["$", "ʃ", "a", "."]
        wave = rng.standard_normal(int(durations.sum()) * 600).astype(np.float32) * .002
        result, records, begin, end = teaching.quiet_aligned_pauses(
            wave, tokens, durations, vocab, 24000, 600, teaching.TeachingSettings())
        cumulative = np.concatenate(([0], np.cumsum(durations))) * 600
        for index in [1, 2, 4, 5]:
            lo, hi = int(cumulative[index]), int(cumulative[index + 1])
            np.testing.assert_array_equal(result[lo - begin:hi - begin], wave[lo:hi])
        self.assertEqual(len(result), end - begin)
        self.assertEqual({r["token_index"] for r in records}, {0, 3, 6})
        self.assertTrue(all(r["after_rms"] < r["before_rms"] for r in records))

    def test_short_pause_has_no_destructive_gate(self):
        durations = np.array([1, 5, 2, 5, 1])
        tokens = np.array([0, 1, 2, 1, 0])
        wave = np.linspace(-.1, .1, int(durations.sum()) * 600, dtype=np.float32)
        result, records, begin, end = teaching.quiet_aligned_pauses(
            wave, tokens, durations, ["$", "s", ","], 24000, 600,
            teaching.TeachingSettings())
        self.assertEqual(records, [])
        np.testing.assert_array_equal(result, wave[begin:end])

    def test_high_energy_articulation_inside_punctuation_is_preserved(self):
        # Real Holos punctuation labels can contain trailing articulation at
        # speech level. A token label alone must never authorize erasing it.
        durations = np.array([3, 8, 16, 3])
        tokens = np.array([0, 1, 2, 0])
        wave = np.zeros(int(durations.sum()) * 600, dtype=np.float32)
        start, stop = (3 + 8) * 600, (3 + 8 + 16) * 600
        time = np.arange(stop - start, dtype=np.float32) / 24000
        wave[start:stop] = .07 * np.sin(2 * np.pi * 150 * time)
        result, records, begin, end = teaching.quiet_aligned_pauses(
            wave, tokens, durations, ["$", "a", "."], 24000, 600,
            teaching.TeachingSettings())
        np.testing.assert_array_equal(result, wave[begin:end])
        skipped = [r for r in records if r["action"] == "preserved_high_energy"]
        self.assertEqual(len(skipped), 1)
        self.assertEqual(skipped[0]["token_index"], 2)
        self.assertGreater(skipped[0]["max_frame_rms"], .005)


if __name__ == "__main__":
    unittest.main()
