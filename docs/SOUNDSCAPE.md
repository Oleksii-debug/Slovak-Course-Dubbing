# Real room background and typing

`scripts/soundscape.py` handles the acoustic background separately from the
new voice. It requires NumPy, SciPy, SoundFile and FFmpeg; it does not download
models or send source audio to a service.

## Preparation

```bash
python scripts/soundscape.py prepare \
  --source /private/lesson.mp3 \
  --speech-spans /private/vad_spans.json \
  --reviewed-source-spans /private/additional_verified_speech.json \
  --output-dir /private/soundscape
```

The VAD file is a JSON array of `start` and `end` seconds. Additional verified
speech intervals use the same fields. Supply tight word/onset intervals only
where residual original speech needs extra exclusion. Broad paragraph intervals
can include legitimate typing pauses and should not be supplied indiscriminately.
The default exclusion adds 100 ms before and 200 ms after speech,
with 25 ms fades outside that exclusion.

The preparation step preserves actual source samples in the remaining pauses.
It also builds waveform references from loud, broadband transients in
VAD-confirmed pauses. Only strong matches are recovered from overlapping
speech, using a brief spectral residual with the original phase. The process
does not put a quieter copy of the teacher under the replacement voice.

Outputs are `source_soundscape.flac`, a short `room_tone.flac`, and a JSON report
with source hashes, intervals, retained transient timings and room-decay fits.
These outputs contain private source audio and should stay outside the public
repository. Re-run speech detection on the isolated stem and inspect any
remaining speech candidates before mixing.

## Mixing and timing

```bash
python scripts/soundscape.py mix \
  --speech /private/synthesized_lesson.mp3 \
  --timeline /private/lesson_timeline.json \
  --soundscape-dir /private/soundscape \
  --output /private/lesson_with_room.flac \
  --reverb-db -19 --bed-db -3 --room-tone-db -12
```

The script accepts the timeline from `assemble_lesson.py`, or a JSON array
with `original_start`, `original_end`, `new_start`, `new_end`. The synthesized
track should contain silence where the source introduction or closing music
will be supplied by the soundscape; do not include that music in both tracks.

Original pauses are copied at their natural speed. Within speech intervals,
transient centers move proportionally to the new voice timing, but the
source excerpts keep their original duration and pitch. A word expanded from
0.4 to 1.8 seconds therefore does not stretch its keyboard click to 4.5 times
its original duration. Source-excerpt boundaries are divided between nearby
events to avoid copying the same neighboring click twice. The event windows are
removed from the copied background and overlaid exactly once on the complete
output timeline, so a key crossing a language or pause boundary keeps its tail.
Very compressed
timing can cause events to overlap; a float intermediate avoids clipping
before the final master gain is selected.

Very quiet, crossfaded source room tone is continuous under the replacement
voice. This avoids turning a noise floor on again at every sentence. Its gain
is independent of the original keyboard/background gain.

The room tail added to the new voice is a deterministic approximation based
on measured source transient decays. It is **not the original room impulse
response**. Decay from a keyboard recording can include keyboard/body
resonance, so the reported apparent RT60 is a working estimate. The existing
source room sound and keyboard tails in retained pauses remain source audio.

## Practical limits and verification

This is a conservative separation method for a mixed recording. Quiet keys
overlapping speech may not be recoverable. Missed speech in a VAD region can
leak into a pause; adding reviewed source intervals and checking the isolated
stem reduces this risk. A detector returning no speech is useful evidence,
not a guarantee that every residual is inaudible.

The longest-lesson development checks included full-stem speech detection,
source transient-decay fits, and a fivefold timeline expansion of a real
117 ms key excerpt. That excerpt's central waveform remained sample-identical
after remapping. No claim of preserving every individual key is made.

Audio processing is in 30-second source blocks. Final mixing uses temporary
mono float WAV files (about 11 MB per minute in total for the two temporary
tracks at 24 kHz); allow enough free disk for the complete lesson. Output
formats are FLAC, 24-bit WAV, and 160 kbps MP3. A report records gains, timing
adjustments and final peak/RMS estimates.
