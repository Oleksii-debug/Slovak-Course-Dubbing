# Reviewed pronunciation and lesson delivery

The second lesson uses the existing `oleksii_day1` profile (display name
**Олексій 2**), selected explicitly earlier by the owner. This mapping identifies
the saved synthesizer profile; it does not independently establish which original
human recording created it. No speaker vectors or enrollment recordings belong
in this repository.

## Ukrainian

`scripts/teaching_voice.py` renders reviewed text using the native Ukrainian
Holos model and the unchanged selected speaker embedding. The new controls are:

- An optional `text_stressed` or `tts_text` field uses combining acute accents,
  for example `уро́к`, `на́голос`, `послу́хайте`, `при́клад`.
  The stress field must match `text` after accents are removed. If both aliases
  are present, they must match exactly.
- The manifest records the actual accented text, resulting IPA, every token's
  duration, and unresolved multisyllabic words. Dictionary stress alone leaves
  some common teaching words unresolved; successful ASR cannot verify stress.
- Model punctuation-token durations are explicit: commas 225 ms, semicolons and colons
  350 ms, dashes 300 ms, periods and exclamation marks 400 ms, question marks
  450 ms, ellipses 500 ms. Ellipsis written as `...` is normalized to a single
  symbol. The last punctuation of each output span is limited to 125 ms,
  because assembly already preserves the original gap between teaching turns.
  Settings can be overridden with a JSON file passed to `--settings`.
- Only the centers of model-labeled non-speech tokens are silenced. Phonetic
  context remains around punctuation, and 20 ms of boundary context remains
  at each utterance edge. Punctuation labels alone do not prove silence: if
  any 20 ms frame in the proposed center has RMS above 0.005, that entire
  center remains unchanged and is recorded for review. This protects trailing
  articulation when the model's alignment spills into punctuation. There is
  no amplitude gate on initial consonants such as /s/, /sh/, or /f/.
- Existing private delivery calibration is loaded when present. For the
  recovered Олексій 2 resources, the profile supplies a gain of 1.111278 and no
  quiet expander. No separate reference-pitch calibration exists for this
  profile in the recovered bundle.

A period between sentences within one output span therefore occupies
approximately 440 ms including the two retained boundary margins. The final
span punctuation occupies 125 ms plus its 20 ms trailing boundary margin.
These are model alignment durations, not a guarantee of equally long acoustic
silence: voiced articulation can extend into punctuation and must be retained.
Original lesson gaps are an additional assembly decision; do not add an
arbitrary extra paragraph pause on top of both. The adapter does not change speech-token durations, pitch, or
speaker identity to manufacture an untested expressive style.

Example invocation (private resource paths are supplied by the operator):

```bash
python scripts/teaching_voice.py \
  --segments reviewed_spans.json \
  --output-dir uk_chunks \
  --runtime /path/to/dordiai/kx47Runtime \
  --models /path/to/private/models \
  --extras /path/to/private/python/Lib/site-packages \
  --profile oleksii_day1 --threads 2
```

The private resource extraction must include the optional configuration files
under `models/voices`: `delivery_profiles.json`, `reference_prosody.json`,
`ukrainian_articulation.json`, and `delivery_equalizers.json`. Omitting these
files silently disables the source runtime's corresponding calibrations.

## Slovak

The earlier pilot's `en-US-BrianMultilingualNeural` base did not satisfy the
owner's native-Slovak pronunciation requirement. The new bounded comparison
uses the genuine Slovak locale voice **`sk-SK-LukasNeural`**, then projects its
clean speech into the same `oleksii_day1` OpenVoice target. Native-locale source
speech is a better grounded input, but a locale name alone does not certify the
converted result.

Use `synthesize_slovak.py --voice sk-SK-LukasNeural` with `tts_prefix: ""` in
reviewed Slovak rows. The prefix formerly needed to prevent multilingual short
words from being read in English is unnecessary for this native-locale voice.
Use a clean reference synthesized by **the same Lukas voice** when invoking
`convert_tts_chunks.py --source-voice sk-SK-LukasNeural`. Never reuse the Brian
conditioning vector or change target identity implicitly.

The bounded comparison contains vowel-length contrasts, soft consonants,
diphthongs, syllabic r/l, and a vacation-related explanation. All five converted
files retain exact source sample counts and have zero integer-PCM clipping;
20 ms amplitude-envelope correlations range from 0.974 to 0.989. These are
timing and signal-integrity checks only. They cannot prove native accent,
phoneme preservation, or human speaker identity.

## Remove redundant service padding

The speech service often appends roughly a second of near-silence. Keeping it
and then restoring the source lesson's original gap produces an unintended
double pause, particularly after one-word examples. `trim_service_padding.py`
removes only the terminal padding after conversion, retaining the untrimmed
native and converted inputs in their original private directories.

```bash
python scripts/trim_service_padding.py \
  --native-dir sk_native --input-dir sk_converted \
  --segments reviewed_spans.json --output-dir sk_trimmed --language sk
```

Use the analogous directories with `--language en` for English. The last
retained sample is no earlier than the largest of:

- last service WordBoundary end plus 300 ms;
- last nonquiet native frame plus 125 ms;
- last nonquiet converted frame plus 125 ms.

The 10 ms frame detector uses 0.004 of the signal's 95th-percentile frame RMS,
bounded between 2/32768 and 0.0005. The service timing guard is deliberately
larger than the desired quiet tail: measured speech can extend beyond the last
word event. A later nonquiet frame in either waveform extends the retained
audio. If these safeguards leave no redundant tail, nothing is removed.

Canonical text and the existing native/converted hashes must agree before
processing. The output is an exact 16-bit PCM prefix of the converted input;
this is verified after writing. No time scaling or speech fade is applied.
Four actual short and long clips (1.872–11.568 s before trimming) retained
their prefixes exactly while removing 0.693–0.870 s of terminal padding each.
Per-file reports include thresholds, endpoints, retained PCM hash and removed
signal level. These checks do not establish pronunciation quality.

## Validation limits

The three safety tests protect quiet initial fricatives, leave pauses unchanged
when there is insufficient context to silence safely, and retain speech-level
articulation inside a model-labeled punctuation interval. Three real Ukrainian
probes and all 68 final Ukrainian lesson spans rendered successfully. The final
pass contains 160.775 seconds of dry speech and 182 audited multisyllabic word
occurrences, with zero unresolved stress words or unambiguous dictionary
conflicts. Every final canonical row fingerprint and WAV hash matched; all
outputs were finite mono 24 kHz PCM without clipping. The energy guard retained
52 punctuation regions. Token-level noise reports are saved beside WAVs.

This is not a newly trained classroom delivery style. Holos retains its learned
intonation and the selected voice profile. Noise inside a speech phoneme is
not removed by the boundary treatment. Human listening remains necessary to
accept teacher delivery and natural Slovak pronunciation; ASR is a content
check, not an accent or stress certificate.
