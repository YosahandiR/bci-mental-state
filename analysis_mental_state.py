#Mental State EEG Analysis
# ------------------------
# Loads EEG trials from mental_state_collect.py
# 1. Applies a notch filter with 60 Hz bandpower noise removal
# 2. Applies bandpass filter 1-40 Hz
# 3. Extracts Beta, Alpha, and Theta
# 4. Compares the results and plots them in a graph

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch, butter, filtfilt, iirnotch
import os

# ──────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────
SUBJECT        = 1
SESSION        = 1
RUN            = 1
DATA_DIR       = f'data/mental_state/sub-{SUBJECT:02d}/ses-{SESSION:02d}/'
SAMPLING_RATE  = 250
NOTCH_FREQ     = 60
TRIAL_DURATION = 10.0

BANDS = {
    'Theta(4-7 Hz)': (4, 7),
    'Alpha (8-12 Hz)': (8,12),
    'Beta (13-30 Hz)': (13, 30),
}

#----------------
# Load Data
# ----------------

def load_data(data_dir, run):
    eeg_path = data_dir + f'eeg_trials_run-{run}.npy'
    labels_path = data_dir + f'labels_run-{run}.npy'

    if not os.path.exists(eeg_path):
        raise FileNotFoundError(f"Could not find: {eeg_path}\nMake sure data is loaded")
    eeg_trials = np.load(eeg_path, allow_pickle=True)
    labels = np.load(labels_path, allow_pickle=True)
    return eeg_trials, labels

#---------------
# FILTERING
#---------------

def notch_filter(eeg, fs=SAMPLING_RATE, freq=NOTCH_FREQ):
    b, a = iirnotch(freq, Q=30, fs=fs)
    return filtfilt(b, a, eeg, axis=1)

def bandpass_filter(eeg, fs=SAMPLING_RATE, low=1.0, high=40.0):
    nyq = fs/2.0
    b, a = butter(4, [low/nyq, high/ nyq], btype='band')
    return filtfilt(b, a, eeg, axis=1)

def preprocess(eeg, fs=SAMPLING_RATE):
    eeg = notch_filter(eeg, fs)
    eeg = bandpass_filter(eeg, fs)
    return eeg


#----------------
# BANDPOWER
#----------------
def bandpower(eeg, fs=SAMPLING_RATE, band=(8,12)):
    freqs, psd = welch(eeg, fs=fs, nperseg=min(fs*2, eeg.shape[1]))
    idx = (freqs >= band[0]) & (freqs <= band[1])
    return psd[:, idx].mean(axis=1) #avg over frequencies in band

def extract_band_power(eeg_trials, labels, fs=SAMPLING_RATE):
    results = {name: {'relaxed': [], 'focused': []} for name in BANDS}

    for trial, label in zip(eeg_trials, labels):
        trial = np.array(trial, dtype=np.float64)
        trial = preprocess(trial, fs)
        for band_name,(lo, hi) in BANDS.items():
            power = bandpower(trial, fs, band=(lo,hi))
            mean_power = power.mean()
            results[band_name][label].append(mean_power)

    return results

#------------
# SUMMARY
#------------
def print_summary(results):
    print("\n" + "="*55)
    print(f"{'Band':<20} {'Relaxed (mean)':<20} {'Focused (mean)'}")
    print("="*55)
    for band_name, conditions in results.items():
        relaxed_mean = np.mean(conditions['relaxed']) if conditions['relaxed'] else float('nan')
        focused_mean = np.mean(conditions['focused']) if conditions['focused'] else float('nan')
        print(f"{band_name:<20} {relaxed_mean:<20.4f} {focused_mean:.4f}")
    print("="*55)

    print("\nInterpretation guide:")
    print("  Alpha ↑ relaxed, ↓ focused  → expected during eyes-closed rest")
    print("  Beta  ↑ focused             → expected during mental effort")
    print("  Theta ↑ can indicate drowsiness or deep relaxation")

#-------------
# PLOT
#-------------
def plot_results(results):
    band_names = list(results.keys())
    n_bands = len(band_names)

    fig, axes = plt.subplots(1, n_bands, figsize=(5 * n_bands, 5))
    fig.suptitle('EEG Bandpower: Relaved vs Focused:', fontsize=14, fontweight='bold')
    colors = {'relaxed': '#4C9BE8', 'focused': '#E8834C'}

    for ax, band_name in zip(axes, band_names):
        relaxed_vals = results[band_name]['relaxed']
        focused_vals = results[band_name]['focused']

        means = [np.mean(relaxed_vals) if relaxed_vals else 0,
                np.mean(focused_vals) if focused_vals else 0]
        stds = [np.std(relaxed_vals) if relaxed_vals else 0,
                np.std(focused_vals) if focused_vals else 0]

        bars = ax.bar(['Relaxed', 'Focused'], means, yerr=stds,
                     color=[colors['relaxed'], colors['focused']],
                      capsize=6, edgecolor='black', linewidth=0.8)

        # Overlay individual trial points
        for i, (condition, vals) in enumerate([('relaxed', relaxed_vals), ('focused', focused_vals)]):
            if vals:
                x_jitter = np.random.uniform(-0.1, 0.1, size=len(vals))
                ax.scatter([i + j for j in x_jitter], vals,
                           color='black', alpha=0.5, s=20, zorder=5)

        ax.set_title(band_name, fontsize=11)
        ax.set_ylabel('Power (µV²/Hz)')
        ax.set_ylim(bottom=0)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig('band_power_results.png', dpi=150, bbox_inches='tight')
    print("\nPlot saved to: band_power_results.png")
    plt.show()



def create_mock_data():
    os.makedirs(DATA_DIR, exist_ok=True)
    n_trials = 6
    n_channels = 8
    n_samples = int(TRIAL_DURATION * SAMPLING_RATE)  # 2500 samples

    eeg_trials = np.array([np.random.randn(n_channels, n_samples) for _ in range(n_trials)], dtype=object)
    labels = np.array(['relaxed', 'focused', 'relaxed', 'focused', 'relaxed', 'focused'])

    np.save(DATA_DIR + f'eeg_trials_run-{RUN}.npy', eeg_trials)
    np.save(DATA_DIR + f'labels_run-{RUN}.npy', labels)
    print("Mock data created!")

# Change your main block to this temporarily:
if __name__ == '__main__':
    create_mock_data()  # ← add this line
    eeg_trials, labels = load_data(DATA_DIR, RUN)
    results = extract_band_power(eeg_trials, labels)
    print_summary(results)
    plot_results(results)