# Existing dordiai-voices backend

**Pilot QA update:** Ukrainian output passed the sampled automatic content and
number checks. The initial split-graph OpenVoice path failed the sampled Slovak
intelligibility check on the original teacher recording. Those converted chunks
must not be treated as accepted lesson audio. The technical smoke below shows
that PCM was generated, not that foreign-language pronunciation was preserved.
An unmodified full FP32 converter also failed this raw-recording check.
The chosen repair is **clean Slovak TTS followed by conversion to Олексій 2**;
the sampled repaired output retained all three Slovak test sentences in ASR.

## Chosen route for the listening lesson

1. Reviewed Russian explanation text is translated into Ukrainian and read by
   native Holos with the existing `oleksii_day1` profile.
2. Reviewed Slovak text is first read by `en-US-BrianMultilingualNeural` through
   the available Edge speech service. The service receives text, not the
   teacher recording or private voice models.
3. The clean Slovak TTS is projected into the saved `oleksii_day1` OpenVoice
   target profile. This uses the actual Brian Slovak TTS reference embedding.

The native Brian Ukrainian sample failed intelligibility QA and is **not** used.
The native Brian Slovak sample and its conversion into Олексій 2 both produced
the complete expected Slovak text in a forced-Slovak, unprompted ASR check,
including `Aj keby si sa na hlavu postavil, o tom nemôže byť ani reči.`
This repairs sampled intelligibility without pretending that raw-speech
conversion was successful. Audible similarity of the final two synthesis
branches still needs the user's listening judgment.

`scripts/convert_tts_chunks.py` consumes the cleaned generated `ID.wav` files and
the canonical reviewed segment JSON. It selects Slovak/English text and writes
separate `ID.wav` files plus hashes; no original or Ukrainian WAV is replaced.
It does not make speech-service calls itself. The tested whole-utterance
projection uses 96-frame decoder windows with context, capped at 30 seconds per
input utterance. For short foreign words the TTS stage should supply a native
language context and remove that context before conversion.

```sh
python scripts/convert_tts_chunks.py \
  --input-dir "$DUBBING_FOREIGN_TTS" \
  --segments "$DUBBING_REVIEWED_SEGMENTS" \
  --output-dir "$DUBBING_FOREIGN_PROJECTED" \
  --source-reference "$DUBBING_BRIAN_SLOVAK_REFERENCE" \
  --extension .wav \
  --runtime "$DUBBING_RUNTIME_SOURCE" \
  --models "$DUBBING_MODELS" \
  --extras "$DUBBING_PURE_PYTHON_PACKAGES" \
  --profile oleksii_day1 --threads 2
```

The listening lesson must select Ukrainian files from the native TTS directory
and Slovak files from the repaired projection directory. Previously generated
raw-teacher conversions are rejected candidates and must not be assembled into
the deliverable.

The pilot uses the existing **Олексій 2** profile, internal ID
`oleksii_day1`. It does not train a new synthesizer.

Ukrainian translated explanations are synthesized with the saved native Holos
profile. Reviewed Slovak or English spans are passed through the saved OpenVoice
V2 tone converter with the matching `oleksii_day1` target embedding. The source
speaker embedding is extracted from clean speech in the input lesson. This path
is intended to keep the original examples and timing while changing their
voice. The pilot demonstrated that timing preservation does not ensure
intelligibility; the raw-speech conversion path requires repair or replacement.

## Source and model identity

Runtime source belongs to
`Oleksii-debug/ChatGPT-Autopilot-ExtensionChatGPT-Autopilot-Extension`, path
`dordiai-voices/source/0.12.2/kx47Runtime`, source commit
`09f4ec8453344b500dd15ecf7bd279339910c5fe`.
The saved tracked-source tar SHA-256 is
`964562dc4da4a99c34d266127f6e7c13026d5deedb76f7c6bf3529dce60bb8ce`.
The consumed `ukrainian_voices.py`, `converted_voices.py` and `voice_policy.py`
were byte-compared with that snapshot. The synthesizer's own policy excludes
removed profiles; the adapter calls its `require_voice_allowed` function.

The already saved core resource archive is
`dordiai-voices-0.13.1-dev.4.nvda-addon`, 1,767,223,748 bytes, SHA-256
`b80f0f2a428ba097b37437ccfd028ce41630b73fd92c5cfa9539da1b08a1b90e`.
It was restored from its four saved ZIP parts using the existing checksum-
verifying restoration tool. Only relevant models, profile embeddings and pure
Python dependencies were extracted for this pilot. No Windows installer was
executed. These private voice assets are not part of this public repository.

The small compatible reference encoder is the pinned FP32 OpenVoice V2 ONNX
export at:

https://huggingface.co/TigreGotico/voiceclonnx-openvoice-v2/resolve/34d010c192c97f763207f488f6057fd07fee42ad/tone_ref_encoder.onnx

Its size is 3,259,275 bytes and SHA-256 is
`3dd4918cab90e1acf7fa5c6f7539c27710e7a3cdfba550468c5ea49399178bf7`.
This exactly matches the existing project's pinned preparation recipe.
The source OpenVoice model carries its own MIT notice in the saved bundle;
the private profile data retains its original permissions and is not relicensed.

## Linux portability

The original Holos decoder uses 48 custom `dordiai::Snake` activation nodes and a
Windows DLL. In a task-local model copy, `scripts/prepare_linux_holos.py` expands
each node to the same `x + b * sin(a * x)^2` formula with standard ONNX operators.
All model weights are unchanged. The original ONNX file and DLL are retained
under explicit backup names. Floating point bit parity with the Windows kernel
is not claimed. Existing project checkouts and original archives remain intact.

```sh
python scripts/prepare_linux_holos.py --models "$DUBBING_MODELS"
```

## Render a reviewed segment manifest

Install the package versions in `requirements-voice.lock.txt` into an isolated
Python 3.12 environment. The supplied private runtime's pure Python
`holos_onnx`, `ipa_uk`, and `ukrainian_word_stress` packages, including the stress
dictionary, must also be available through the `--extras` directory. The pilot
used CPU only, two inference threads. No inference network calls are required.

Each reviewed segment supplies `id`, `start`, `end`, `language`, `source_language`,
`text` and `mode`. `mode=tts` requires Ukrainian output text; `mode=convert`
accepts only explicitly reviewed Slovak or English original speech.

```sh
python scripts/synthesize_spans.py \
  --source "$DUBBING_SOURCE_WAV" \
  --segments "$DUBBING_REVIEWED_SEGMENTS" \
  --output-dir "$DUBBING_CHUNKS" \
  --runtime "$DUBBING_RUNTIME_SOURCE" \
  --models "$DUBBING_MODELS" \
  --extras "$DUBBING_PURE_PYTHON_PACKAGES" \
  --profile oleksii_day1 --threads 2
```

The default teacher enrollment interval is 16.14–50.0 seconds for lesson 575.
For another file, pass clean reference boundaries with `--reference-start` and
`--reference-end`. Do not enroll background music as teacher speech.

Outputs are mono 24 kHz, 16-bit WAV chunks named after segment IDs. Converted
spans retain their exact durations. Ukrainian TTS duration can differ from the
Russian source. Assembly should preserve adequate pauses and must not crop
the translated speech to force it into a shorter interval.

`manifest.json` records source/model/segment/WAV hashes and real output durations.
Rerunning with an extended manifest reuses only chunks whose identities and
actual bytes still match. `teacher_embedding.npy` stays in the private working
directory. No credentials or private waveform data are written into source code.

## Observed smoke result

On 26 November 2026 the adapter produced a 2.56-second Ukrainian test utterance
with this actual Holos profile and a 4.12-second converted original Slovak
introduction. The warmed per-request times were approximately 2.51 and 3.34
seconds respectively. A separate 7.00-second original conversion also returned
exactly 7.00 seconds. These are observed small-sample timings, not a full-course
performance guarantee. The second CLI run correctly reused the previously
verified Ukrainian WAV. `voice-smoke-evidence.json` records exact identities.

Successful ONNX inference, valid PCM and stable timing do not certify correct
translation, language boundaries, accent, speaker similarity or listening
comfort. Those remain checks on the completed lesson.
