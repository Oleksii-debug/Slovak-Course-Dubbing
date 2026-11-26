# Reproducible transcription for mixed Russian and Slovak lessons

`scripts/transcribe.py` runs the tested faster-whisper CPU stack. It produces
reviewable text and word timestamps. It does not translate or approve the text
for synthesis. Keep original audio and transcript outputs in the ignored
`inputs/` and `work/` directories, outside public commits.

## Install the tested dependencies

The lock file records the actual Python 3.12 Linux environment used for the
first lesson. Use a separate environment from the voice synthesis dependencies.

```bash
python3.12 -m venv .venv/asr
.venv/asr/bin/python -m pip install -r requirements-asr.lock.txt
.venv/asr/bin/python scripts/transcribe.py --help
```

On Windows, the interpreter is `.venv\asr\Scripts\python.exe`.
Other Python versions and platforms have not been verified with this lock file.

Two pins are material to the pilot:

- `av==16.1.0`: PyAV 19 rejected the `metadata_errors` argument used by
  faster-whisper 1.2.1. Version 16.1.0 decoded the supplied MP3 files successfully.
- `socksio==1.0.0`: needed by the installed HTTP transport when the environment
  supplies a SOCKS proxy. Keep the environment's configured proxy settings.

`ctranslate2==4.9.1` and `faster-whisper==1.2.1` were used with CPU `int8`
inference. The default two threads reduce contention with other workers. Start
with one model process at a time; concurrent ASR and voice models exhausted the
pilot's approximately 10 GB memory budget.

## Transcribe two passes

Use a new output directory for each run. Existing transcript output is never
overwritten by this command. The first model load downloads official model files
when they are absent; `--local-files-only` instead requires an existing cache.

```bash
.venv/asr/bin/python scripts/transcribe.py \
  --input inputs/lesson.mp3 --output-dir work/lesson/asr-ru \
  --language ru --model medium --threads 2 --download-root models/whisper

.venv/asr/bin/python scripts/transcribe.py \
  --input inputs/lesson.mp3 --output-dir work/lesson/asr-sk \
  --language sk --model medium --threads 2 --download-root models/whisper \
  --local-files-only
```

For an existing local model directory, pass its path as `--model` instead of
`medium`. An already populated `models/whisper` cache can be reused by later
workers without another weight download.

Every run uses `task="transcribe"`, word timestamps, VAD with a 450 ms minimum
silence, beam size 3, and `condition_on_previous_text=False`. No translation
prompt is supplied. `--language auto` detects language per segment; `ru`, `sk`,
and `uk` force the decoder language.

## Bounded processing under memory pressure

For lesson 576, the medium process was killed under the shared environment's
memory pressure. The completed alternative used `small`, CPU int8, four
threads, beam 3, and overlapping 90-second windows with three seconds of
context on each side. Russian and Slovak were decoded sequentially in one
model instance; uncertain short intervals were checked separately with beam 5.
The smaller model still required editorial corrections and source-coverage
checks. Do not treat decoder agreement as proof of every word or language.

The CLI also accepts `--model small` and bounded `--start` / `--end` intervals:

```bash
.venv/asr/bin/python scripts/transcribe.py \
  --input inputs/lesson.mp3 --output-dir work/lesson/chunk-000-sk \
  --language sk --model small --threads 2 --start 0 --end 93 \
  --download-root models/whisper
```

On later windows, subtract the overlap when building the unique reviewed
transcript; never synthesize the same repeated recognition twice.

## Why both passes and text review are necessary

In the pilot, automatic detection confidently selected Slovak for a window
containing Russian spoken with a Slovak accent. It then wrote Russian words
phonetically in the Slovak alphabet. A forced Russian pass recovered those
Russian explanations but also translated some real Slovak examples into Russian
despite the `transcribe` task. Longer forced Russian decoding of Slovak speech
eventually produced unreliable fragments.

Consequently, neither the automatic language label nor a fluent transcript is
sufficient evidence of the language actually spoken. A forced-language
probability of `1` is a decoder setting, not verified confidence.

Compare the Russian and Slovak passes with the source. Identify actual language
switches, including short glosses inside a sentence. Translate only the Russian
utterances into Ukrainian; preserve the Slovak and English examples. Correct
ASR spelling and names before synthesis. Keep repetitions, answer letters,
exercise order and instructional explanations. Mark unresolved readings for
review instead of filling them with invented text. Automatic timestamps are
estimates and require extra care at language boundaries.

The original pilot required a full Russian introduction pass, a Slovak pass of
the exercises, and a few short checks. Repeated whole-file passes are not
required once a specific uncertainty can be checked in a small interval.

```bash
.venv/asr/bin/python scripts/transcribe.py \
  --input inputs/lesson.mp3 --output-dir work/lesson/check-495-505-sk \
  --language sk --model medium --threads 2 \
  --start 495 --end 505 --download-root models/whisper --local-files-only
```

`--start` and `--end` are source seconds. An end beyond the decoded source is
clipped to the source duration; a start outside it is rejected. Segment and word
`start`/`end` values include the source offset, so clips remain comparable to the
full-file passes. Other model fields, including `seek`, retain their internal
model values.

## Outputs and completion

| File | Meaning |
| --- | --- |
| `metadata.json` | Input hash, model and dependency versions, options, interval, progress, and completion receipt. |
| `segments.jsonl` | One flushed segment per line, including word timestamps; available during processing. |
| `transcript.txt` | The same streaming text with source timestamps. |
| `segments.json` | Final segment array, written after the transcription generator finishes. |

Treat a run as complete only when `metadata.json` has `complete: true` **and**
`segments.json` exists. The final array alone is not a completion certificate.
`complete` means the selected interval finished transcription; it does not mean
the transcript is correct or reviewed.

The metadata starts with `complete: false` before model loading and is updated
atomically after each segment. Ctrl+C and SIGTERM leave `complete: false`, record
`status: "interrupted"`, preserve partial JSONL/text, and return a nonzero exit
status. Other handled failures record `status: "failed"`.

A hard kill or power loss cannot run the cleanup handler. In that case metadata
can still say `status: "running"`, but `complete: false` remains the authoritative
state. A final incomplete JSONL line may need to be discarded. Partial output is
evidence of progress, not a finished lesson. A new run must use another output
directory; this minimal CLI does not automatically resume inference from a
partial transcript.

## Validation scope

The reusable CLI was checked for Python syntax and command-line help. Its ASR
settings and dependency pins come from the actual lesson-processing runs; no
additional full lesson was processed solely to test the wrapper. The text and
coverage review does not certify final synthesized pronunciation or auditory
quality.

Upstream references: [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
and the [official converted medium model](https://huggingface.co/Systran/faster-whisper-medium).
