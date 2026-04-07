# Speaker Detect — Mac (M-series)

Real-time speaker detection for macOS. Notifies you visually and with a native
notification whenever a **different person** starts speaking — built for users
who need visual cues for speaker changes.

## Features

- 🎤 Real-time microphone capture at 16 kHz
- 🧠 SpeechBrain **ECAPA-TDNN** speaker embeddings (192-dim, SOTA)
- ⚡ **Apple Silicon MPS** acceleration (auto-detected, falls back to CPU)
- 🎨 Full-screen PyQt6 UI: giant speaker letter + flash on change
- 🔔 Native macOS notifications via `osascript` (works out of the box)
- 🟢 Live energy meter and debug line showing similarity scores
- 🔄 Works offline after first model download (no HuggingFace token needed)

## Requirements

- macOS 12+ (Apple Silicon recommended, Intel works too)
- Python 3.10 or newer
- ~500 MB disk (PyTorch + model weights)
- Microphone permission

## Quick start

```bash
cd mac
./run.sh
```

That's it. First run creates a virtualenv, installs dependencies, downloads
ECAPA-TDNN (~20 MB) and launches the app. Subsequent runs start in a few
seconds.

When macOS asks for microphone permission, accept it. If you don't see the
prompt, grant it manually in **System Settings → Privacy & Security → Microphone**.

## Manual setup (if you don't want `run.sh`)

```bash
cd mac
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python speaker_detect.py
```

## How it works

1. `sounddevice` captures mono 16 kHz audio into a 3-second ring buffer.
2. Every ~1.2 seconds the buffer is fed through **ECAPA-TDNN** to produce a
   192-dim speaker embedding (L2-normalised).
3. The embedding is compared against known speaker profiles via
   **cosine similarity**.
4. If similarity ≥ 0.55 → same speaker; otherwise a new profile is created.
5. Profiles freeze after 8 confirmations (averaged embedding) to stay stable.
6. On speaker change → screen flashes white twice + Mac notification fires.

## Tuning

Open `speaker_detect.py` and tweak the constants at the top:

| Constant | Default | Meaning |
|---|---|---|
| `SIMILARITY_THRESHOLD` | `0.55` | Higher = stricter (fewer false merges, more false splits) |
| `PROCESS_INTERVAL` | `1.2` s | How often detection runs |
| `BUFFER_SECONDS` | `3.0` s | Audio window fed to the model |
| `FREEZE_AFTER` | `8` | Number of confirmations before a profile is locked |
| `RMS_SILENCE_THRESHOLD` | `0.003` | Below this the audio is treated as silence |

If the app keeps merging two different people → raise the threshold to `0.6`.
If it keeps splitting one person into many → lower it to `0.5`.

## Troubleshooting

**"MODEL FAILED" on startup**
Usually a network error while downloading ECAPA-TDNN. Re-run, or check
`~/.cache/speaker_detect/ecapa`.

**No microphone input / energy bar stays empty**
Grant microphone permission in System Settings, then restart the app.

**MPS not used (slow on M-series)**
Verify PyTorch is recent: `pip install -U torch torchaudio`. The first line
printed on launch says which device is in use:
`[engine] using device: mps`.

**Notifications not showing**
macOS silences notifications from unsigned terminals by default. Allow them
for "Script Editor" and "Terminal" in System Settings → Notifications.

## Files

```
mac/
├── speaker_detect.py   # main app (engine + PyQt6 UI)
├── requirements.txt    # Python dependencies
├── run.sh              # one-shot launcher (creates venv, installs, runs)
└── README.md           # this file
```
