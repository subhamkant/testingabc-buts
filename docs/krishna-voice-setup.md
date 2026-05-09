# Krishna direct-address — voice setup guide

The Krishna direct-address slot (13:00 UTC daily cron) needs a different
voice than the standard Mahabharata narrator to land emotionally. The
default ElevenLabs voice (`pNInz6obpgDQGcFmaJgB` / "Adam") is a calm
narrator — wrong for battlefield speeches like "उठो पार्थ! शस्त्र उठाओ!"

This guide documents three options, in increasing effort.

## Option 1 — pick a more dramatic library voice (5 minutes)

Browse https://elevenlabs.io/app/voice-library and search for **dramatic
narrator** / **storyteller** / **deep male** + filter language to Hindi
or Multilingual.

Voices that historically work for Hindi mythology content:
- **Brian** — deep narrator, more emotional than Adam
- **Antoni** — warmer, more dramatic on Multilingual v2
- **Adam Stone** — battlefield-like delivery
- **Daniel** — older / wiser tone

Once you find one you like:

1. Click "Add to my voices" → ElevenLabs assigns you a voice ID
2. Copy the voice ID (looks like `IKne3meq5aSn9XLyUdCD`)
3. Add it as a GitHub Secret:
   - Repo → Settings → Secrets and variables → Actions → New repository secret
   - Name: `ELEVENLABS_VOICE_ID_KRISHNA`
   - Value: the voice ID
4. Also add to local `.env` for testing:
   ```
   ELEVENLABS_VOICE_ID_KRISHNA=<voice-id>
   ```

The next Krishna run will use this voice automatically. Standard Mahabharata
videos still use the default narrator.

## Option 2 — clone a real Hindi actor's voice (15 minutes)

This is what produces actor-grade Krishna delivery (closest to references
like Sumedh Mudgalkar's Star Bharat Krishna or Saurabh Jain).

You need: an ElevenLabs **Creator plan** ($5/mo for Instant Voice Cloning).

**Pick the source actor.** Some royalty-free / fair-use options for
training audio:
- An actor's free YouTube interview audio (1–3 minutes of clean speech)
- Public-domain Hindi audiobook recordings
- Any voice you've recorded yourself / have permission to clone

**Train the clone:**

1. Extract 60–180 seconds of CLEAN audio (no music, no background noise):
   ```
   ffmpeg -i source.mp4 -ss 00:01:23 -t 90 -vn -ar 44100 -ac 1 sample.wav
   ```
2. ElevenLabs → Voices → Add Voice → Instant Voice Cloning
3. Upload `sample.wav`, name the voice (e.g. "Krishna-dramatic")
4. Copy the new voice ID
5. Set it as `ELEVENLABS_VOICE_ID_KRISHNA` (both repo Secret + local `.env`)

**Quality tip:** the cleaner and more EMOTIONAL the training sample,
the better the clone. A flat news-reader sample makes a flat clone.
Pick a sample where the actor is clearly performing emotion.

## Option 3 — Professional Voice Cloning (1+ hour)

For broadcast-grade results, ElevenLabs **Pro plan** ($99/mo) unlocks
Professional Voice Cloning which trains on 30+ minutes of audio. Far
more accurate than Instant Cloning. Only worth it if the channel scales
past hobby and you commit to a Krishna-direct-address daily slot
indefinitely.

## How the pipeline uses the voice ID

[pipeline/tts_generator.py](../pipeline/tts_generator.py) resolves the
voice ID in this order, falling back if any are unset:

1. `ELEVENLABS_VOICE_ID_KRISHNA` (per-series override)
2. `ELEVENLABS_VOICE_ID` (global override)
3. `pNInz6obpgDQGcFmaJgB` (hardcoded "Adam" default)

So setting just `ELEVENLABS_VOICE_ID_KRISHNA` affects only the Krishna
slot — Mahabharata third-person videos stay on the default narrator.

## Per-scene voice settings (already configured)

The pipeline already varies voice_settings scene-by-scene for Krishna:

| Scene | Role | stability | style |
|---|---|---|---|
| 1 | Opening address | 0.40 | 0.45 |
| 2 | Hard truth | 0.40 | 0.50 |
| 3 | **Imperative peak** | **0.10** | **0.85** |
| 4 | Reframe | 0.45 | 0.45 |
| 5 | Blessing / charge | 0.30 | 0.55 |
| Outro | Subscribe CTA | 0.45 | 0.40 |

Lower stability → more emotional swing. Higher style → more dramatic
inflection. Scene 3 also gets ElevenLabs v3 audio tags
(`[determined] [intense]`) when v3 is available on the API key — these
push the model toward the commanding-peak delivery.

If your chosen voice sounds too volatile or too monotone with these
settings, edit `_KRISHNA_PER_SCENE_SETTINGS` in
[pipeline/tts_generator.py](../pipeline/tts_generator.py).
