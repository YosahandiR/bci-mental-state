from psychopy import visual, core
from psychopy.hardware import keyboard
import numpy as np
import os, time, glob, sys
from threading import Thread, Event
from queue import Queue

# ──────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────
MOCK_MODE      = False
SUBJECT        = 1
SESSION        = 1
RUN            = 1
SAVE_DIR       = f'data/mental_state/sub-{SUBJECT:02d}/ses-{SESSION:02d}/'
SAMPLING_RATE  = 250
N_PER_CLASS    = 3 if MOCK_MODE else 20
TRIAL_DURATION = 10.0
BASELINE_DUR   = 1.0 if MOCK_MODE else 2.0
REST_BETWEEN   = 1.0 if MOCK_MODE else 3.0
CLASSES        = ['relaxed', 'focused']
WINDOW_W       = 1280
WINDOW_H       = 720

RAIN_FILE = 'rain.wav'

AUDIO_PROMPTS = {

    'relaxed': "Please relax.",
    'focused': "Keep a running total.",
}

# Running total task: numbers spoken every N seconds during focused trials
RUNNING_TOTAL_INTERVAL = 1.0   # seconds between each number
RUNNING_TOTAL_RANGE    = (2, 20)  # range of numbers to add each time

# ──────────────────────────────────────────────
# MOCK BOARD
# ──────────────────────────────────────────────
CYTON_BOARD_ID = 0
BAUD_RATE      = 115200
ANALOGUE_MODE  = '/2'
N_CHANNELS     = 8

def start_mock():
    from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
    print("[MOCK MODE] Using BrainFlow SYNTHETIC_BOARD")

    params = BrainFlowInputParams()
    board  = BoardShim(BoardIds.SYNTHETIC_BOARD, params)
    board.prepare_session()
    board.start_stream()

    stop_event = Event()
    queue_in   = Queue()

    def _collect(q, stop):
        while not stop.is_set():
            data = board.get_board_data()
            eeg  = data[BoardShim.get_eeg_channels(BoardIds.SYNTHETIC_BOARD)]
            ts   = data[BoardShim.get_timestamp_channel(BoardIds.SYNTHETIC_BOARD)]
            if len(ts) > 0:
                q.put((eeg, ts))
            time.sleep(0.05)

    Thread(target=_collect, args=(queue_in, stop_event), daemon=True).start()
    return board, stop_event, queue_in


# ──────────────────────────────────────────────
# REAL BRAINFLOW  (reused from SSVEP script)
# ──────────────────────────────────────────────

def find_openbci_port():
    import serial
    if sys.platform.startswith('win'):
        ports = ['COM%s' % (i + 1) for i in range(256)]
    elif sys.platform.startswith('linux') or sys.platform.startswith('cygwin'):
        ports = glob.glob('/dev/ttyUSB*')
    elif sys.platform.startswith('darwin'):
        ports = glob.glob('/dev/cu.usbserial*')
    else:
        raise EnvironmentError('Unsupported OS')
    for port in ports:
        try:
            s = serial.Serial(port=port, baudrate=BAUD_RATE, timeout=None)
            s.write(b'v')
            time.sleep(2)
            if s.inWaiting():
                line = ''
                while '$$$' not in line:
                    line += s.read().decode('utf-8', errors='replace')
                if 'OpenBCI' in line:
                    s.close()
                    return port
            s.close()
        except Exception:
            pass
    raise OSError('Cannot find OpenBCI port.')


def start_brainflow():
    from brainflow.board_shim import BoardShim, BrainFlowInputParams
    params             = BrainFlowInputParams()
    params.serial_port = 'COM4'
    board              = BoardShim(CYTON_BOARD_ID, params)
    board.prepare_session()
    board.config_board('/0')
    board.config_board('//')
    board.config_board(ANALOGUE_MODE)
    board.start_stream(45000)

    stop_event = Event()
    queue_in   = Queue()

    def _collect(q, stop):
        while not stop.is_set():
            data = board.get_board_data()
            eeg  = data[BoardShim.get_eeg_channels(CYTON_BOARD_ID)]
            ts   = data[BoardShim.get_timestamp_channel(CYTON_BOARD_ID)]
            if len(ts) > 0:
                q.put((eeg, ts))
            time.sleep(0.05)

    Thread(target=_collect, args=(queue_in, stop_event), daemon=True).start()
    return board, stop_event, queue_in


# ──────────────────────────────────────────────
# AUDIO
# ──────────────────────────────────────────────

def play_beep(frequency=880, duration=0.3):
    """Short beep via sounddevice to signal start of recording."""
    try:
        import sounddevice as sd
        srate = 44100
        t     = np.linspace(0, duration, int(srate * duration), endpoint=False)
        tone  = 0.5 * np.sin(2 * np.pi * frequency * t)
        fade  = int(srate * 0.01)
        tone[:fade]  *= np.linspace(0, 1, fade)
        tone[-fade:] *= np.linspace(1, 0, fade)
        print(f"[BEEP] device: {sd.query_devices(kind='output')['name']}")
        sd.play(tone*0.3, srate)
        sd.wait()
        print("[BEEP] done")
    except Exception as e:
        print(f"[BEEP ERROR] {e}")


def speak(text, rate=150):
    """Speak text via pyttsx3 (Windows SAPI). Falls back to print."""
    try:
        import pyttsx3
        engine = pyttsx3.init()
        engine.setProperty('rate', rate)
        engine.say(text)
        engine.runAndWait()
        engine.stop()
    except Exception as e:
        print(f'[PROMPT] {text}  (error: {e})')


def play_rain(stop_event):
    """
    Play rain.wav on loop in a background thread during relaxed trials.
    Uses pygame for reliable looping audio file playback.
    """
    try:
        import pygame
        pygame.mixer.init()
        pygame.mixer.music.load(RAIN_FILE)
        pygame.mixer.music.set_volume(0.8)
        pygame.mixer.music.play(-1)   # -1 = loop indefinitely
        while not stop_event.is_set():
            time.sleep(0.05)
        pygame.mixer.music.stop()
        pygame.mixer.quit()
    except Exception as e:
        print(f'[RAIN ERROR] {e}')


def play_running_total(duration_s, trial_seed, stop_event):
    """
    Speak a starting instruction then a new random number every
    RUNNING_TOTAL_INTERVAL seconds for the duration of the focused trial.
    Participant keeps a running total by adding each number to the last.
    Uses trial_seed for reproducibility per trial.
    """
    import random
    rng = random.Random(trial_seed)

    deadline = time.time() + duration_s
    while time.time() < deadline and not stop_event.is_set():
        num = rng.randint(*RUNNING_TOTAL_RANGE)
        print(f"[RUNNING TOTAL] Speaking: {num}")
        speak(str(num), rate=120)   # slower for clarity
        # Wait between numbers, checking stop_event frequently
        wait_until = time.time() + RUNNING_TOTAL_INTERVAL
        while time.time() < wait_until and not stop_event.is_set():
            time.sleep(0.05)


# ──────────────────────────────────────────────
# PSYCHOPY HELPERS
# ──────────────────────────────────────────────

def build_window():
    return visual.Window(
        size=[WINDOW_W, WINDOW_H],
        fullscr=True,
        allowGUI=False,
        units='norm',
        color='black',
    )


def show_text(win, msg, pos=(0, 0), height=0.09, color='white'):
    visual.TextStim(win, text=msg, pos=pos, height=height,
                    color=color, wrapWidth=1.8).draw()
    win.flip()


def wait_with_escape(kb, win, seconds):
    deadline = core.getTime() + seconds
    while core.getTime() < deadline:
        if kb.getKeys(keyList=['escape']):
            return True
        core.wait(0.02)
    return False


# ──────────────────────────────────────────────
# TRIAL SEQUENCE — strictly alternating
# ──────────────────────────────────────────────

def build_trial_sequence(n_per_class=N_PER_CLASS, seed=RUN):
    import random
    rng   = random.Random(seed)
    first = rng.choice(CLASSES)

    trials      = []
    focused_idx = 0
    for i in range(n_per_class * 2):
        cls = CLASSES[i % 2] if first == 'relaxed' else CLASSES[(i + 1) % 2]
        if cls == 'relaxed':
            # Relaxed trials all use the same instruction
            trials.append((cls, AUDIO_PROMPTS['relaxed'], None))
        else:
            # Each focused trial gets a unique seed so numbers differ per trial
            trial_seed = seed * 1000 + focused_idx
            trials.append((cls, AUDIO_PROMPTS['focused'], trial_seed))
            focused_idx += 1
    return trials


# ──────────────────────────────────────────────
# EEG COLLECTION
# ──────────────────────────────────────────────

def drain_queue(queue_in, eeg_buf, ts_buf):
    while not queue_in.empty():
        eeg_chunk, ts_chunk = queue_in.get()
        eeg_buf.append(eeg_chunk)
        ts_buf.append(ts_chunk)


def collect_trial_eeg(queue_in, duration_s, kb, sampling_rate=SAMPLING_RATE):
    eeg_chunks = []
    ts_chunks  = []
    n_needed   = int(duration_s * sampling_rate)
    deadline   = core.getTime() + duration_s + 1.0

    while True:
        if kb.getKeys(keyList=['escape']):
            return None, True
        drain_queue(queue_in, eeg_chunks, ts_chunks)
        total = sum(c.shape[1] for c in eeg_chunks) if eeg_chunks else 0
        if total >= n_needed or core.getTime() > deadline:
            break
        core.wait(0.02)   # use core.wait instead of time.sleep so PsychoPy stays responsive

    if not eeg_chunks:
        return None, False
    eeg = np.concatenate(eeg_chunks, axis=1)
    return eeg[:, :n_needed], False


def save_data(eeg_trials, labels, prompts_log, markers):
    os.makedirs(SAVE_DIR, exist_ok=True)
    np.save(SAVE_DIR + f'eeg_trials_run-{RUN}.npy', np.array(eeg_trials, dtype=object))
    np.save(SAVE_DIR + f'labels_run-{RUN}.npy',      np.array(labels))
    np.save(SAVE_DIR + f'prompts_run-{RUN}.npy',     np.array(prompts_log))
    np.save(SAVE_DIR + f'markers_run-{RUN}.npy',     np.array(markers, dtype=object))
    print(f"Saved {len(eeg_trials)} trials to {SAVE_DIR}")


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────

def run_experiment():
    win = build_window()
    kb  = keyboard.Keyboard()

    if MOCK_MODE:
        board, stop_event, queue_in = start_mock()
    else:
        board, stop_event, queue_in = start_brainflow()

    trial_sequence = build_trial_sequence()
    eeg_trials, labels, prompts_log, markers = [], [], [], []

    os.makedirs(SAVE_DIR, exist_ok=True)

    # ── Intro ──
    show_text(win,
              "Mental State Recording\n\n"
              "You will hear an instruction before each trial.\n"
              "When you hear a beep, begin following the instruction.\n"
              "For relaxed trials, you will hear rain sounds.\n"
              "For focused trials, you will hear numbers — keep a running total.\n"
              "Keep your eyes closed the entire time.\n\n"
              "Press SPACE to begin.")
    kb.waitKeys(keyList=['space', 'escape'])
    if kb.getKeys(keyList=['escape']):
        win.close(); return

    show_text(win, "Close your eyes now.\n\nTrials starting soon...")
    if wait_with_escape(kb, win, 2.0):
        win.close(); return

    for i_trial, (cls, prompt_text, trial_seed) in enumerate(trial_sequence):

        # ── Inter-trial rest (escapable) ──
        show_text(win, f"Trial {i_trial + 1} / {len(trial_sequence)}")
        if wait_with_escape(kb, win, REST_BETWEEN):
            save_data(eeg_trials, labels, prompts_log, markers)
            stop_event.set(); board.stop_stream(); board.release_session()
            win.close(); return

        # ── Audio cue ──
        show_text(win, f"Trial {i_trial + 1} / {len(trial_sequence)}")
        speak(prompt_text)

        # ── Baseline pause (escapable) ──
        if wait_with_escape(kb, win, BASELINE_DUR):
            save_data(eeg_trials, labels, prompts_log, markers)
            stop_event.set(); board.stop_stream(); board.release_session()
            win.close(); return

        # ── Beep then record ──
        # Wait for pyttsx3 to fully release audio device before beep and thread
        time.sleep(0.8)
        play_beep()
        time.sleep(0.3)   # extra wait after beep before starting audio thread

        drain_queue(queue_in, [], [])   # flush stale data before recording
        markers.append((i_trial, cls, time.time()))

        show_text(win, f"Trial {i_trial + 1} / {len(trial_sequence)}\n\n● Recording...")

        # Start background audio for the trial in a separate thread
        audio_stop = Event()
        if cls == 'relaxed':
            print("[AUDIO] Starting rain...")
            audio_thread = Thread(target=play_rain, args=(audio_stop,), daemon=True)
        else:
            print("[AUDIO] Starting running total...")
            audio_thread = Thread(target=play_running_total,
                                  args=(TRIAL_DURATION, trial_seed, audio_stop), daemon=True)
        audio_thread.start()

        trial_eeg, escaped = collect_trial_eeg(queue_in, TRIAL_DURATION, kb)

        audio_stop.set()
        audio_thread.join(timeout=3.0)

        if escaped:
            save_data(eeg_trials, labels, prompts_log, markers)
            stop_event.set(); board.stop_stream(); board.release_session()
            win.close(); return

        if trial_eeg is not None:
            eeg_trials.append(trial_eeg)
            labels.append(cls)
            prompts_log.append(prompt_text)
            print(f"Trial {i_trial+1}: {cls} — shape {trial_eeg.shape}")
        else:
            print(f"Trial {i_trial+1}: NO DATA — skipping")

    # ── End ──
    save_data(eeg_trials, labels, prompts_log, markers)
    show_text(win, f"Session complete!\n{len(eeg_trials)} trials saved.")
    core.wait(3.0)

    stop_event.set()
    board.stop_stream()
    board.release_session()
    win.close()


if __name__ == '__main__':
    run_experiment()