# BCI Mental State Data Collection

EEG data collection system for classifying relaxed vs. focused mental states using an OpenBCI Cyton board.

## Overview

Participants listen to audio instructions and perform either a relaxation or mental arithmetic task while EEG is recorded. Data is saved for offline classification.

- **Relaxed**: Rain sounds play in the background
- **Focused**: Random numbers are spoken every 1.5 seconds, participant keeps a running total

## Setup for Windows 11

```
git clone https://github.com/TimothyMao/bci-mental-state.git
cd bci-mental-state
pip install virtualenv
virtualenv pyenv --python=3.11.9
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
pyenv\Scripts\activate
pip install -r requirements.txt
```

## Files

- `mental_state_collect.py` — main data collection script
- `requirements.txt` — Python dependencies
- `rain.wav` — ambient rain audio for relaxed trials (not included, add your own)

## Usage

1. Set `MOCK_MODE = False`
2. Set `SUBJECT`, `SESSION`, and `RUN` at the top of the script
3. Connect the Cyton dongle via USB
4. Run:
```
python mental_state_collect.py
```

## Experiment Design

- 2 classes: relaxed, focused
- 20 trials per class per session (40 total)
- 10 second trial duration
- Trials alternate strictly between relaxed and focused
- Recommended: 2 sessions per participant with a break in between

## Data

Saved to `data/mental_state/sub-{SUBJECT}/ses-{SESSION}/`:
- `eeg_trials_run-{RUN}.npy` — EEG array per trial
- `labels_run-{RUN}.npy` — class labels
- `markers_run-{RUN}.npy` — onset timestamps
- `prompts_run-{RUN}.npy` — audio prompts used