import numpy as np
import os

# ── Paths ─────────────────────────────────────────────────────────────────────
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "processed_data")

# ── Load data ─────────────────────────────────────────────────────────────────
# Each array was saved by load_data.py.
# X_train / X_test have shape (samples, 128, 9):
#   - axis 0 = individual windows (one per row)
#   - axis 1 = 128 time-steps inside a 2.56-second window
#   - axis 2 = 9 sensor channels (body_acc x/y/z, body_gyro x/y/z, total_acc x/y/z)
print("Loading data from processed_data/ ...")
X_train = np.load(os.path.join(PROCESSED_DIR, "X_train.npy"))
X_test  = np.load(os.path.join(PROCESSED_DIR, "X_test.npy"))
y_train = np.load(os.path.join(PROCESSED_DIR, "y_train.npy"))
y_test  = np.load(os.path.join(PROCESSED_DIR, "y_test.npy"))

print(f"  X_train : {X_train.shape}")
print(f"  X_test  : {X_test.shape}")
print(f"  y_train : {y_train.shape}")
print(f"  y_test  : {y_test.shape}")


# ── Feature extraction ────────────────────────────────────────────────────────
# We work on the full batch at once (vectorised NumPy) rather than looping
# window-by-window, so axis=1 always means "collapse across the 128 time-steps".
# Every feature call produces an array of shape (samples, 9) — one value per
# channel per window.  We collect them in a list and concatenate at the end.

def extract_features(X: np.ndarray) -> np.ndarray:
    """
    Given X of shape (n_samples, 128, 9), return a 2-D feature matrix of
    shape (n_samples, n_features) where every row is one window's feature vector.

    Features extracted (8 per channel x 9 channels = 72 total):
        Time-domain  : mean, std, min, max, rms, zero-crossing rate
        Freq-domain  : spectral energy, dominant frequency index
    """
    features = []   # will hold arrays of shape (n_samples, 9), one per feature

    # ── Time-domain features ─────────────────────────────────────────────────

    # MEAN — the average value of the signal over the window.
    # A high mean in body_acc_z might indicate the person is standing upright.
    mean = np.mean(X, axis=1)                       # (samples, 9)
    features.append(mean)

    # STANDARD DEVIATION — how much the signal bounces around its mean.
    # A high std means the sensor reading varies a lot (e.g. during walking).
    std = np.std(X, axis=1)                         # (samples, 9)
    features.append(std)

    # MIN — the smallest value recorded in the window.
    minimum = np.min(X, axis=1)                     # (samples, 9)
    features.append(minimum)

    # MAX — the largest value recorded in the window.
    maximum = np.max(X, axis=1)                     # (samples, 9)
    features.append(maximum)

    # ROOT MEAN SQUARE (RMS) — the square root of the average of squared values.
    # Unlike the mean, it is always positive and captures the overall "energy"
    # or magnitude of the signal regardless of direction.
    rms = np.sqrt(np.mean(X ** 2, axis=1))          # (samples, 9)
    features.append(rms)

    # ZERO-CROSSING RATE (ZCR) — how often the signal flips from positive to
    # negative (or vice versa) within the window, divided by the number of
    # possible crossings (127).  A high ZCR usually means a fast, oscillating
    # motion like running; a low ZCR suggests slow or static activity.
    #
    # np.sign(X)   : turns every value into -1, 0, or +1
    # np.diff(...) : subtracts consecutive signs  → non-zero where a crossing happens
    # != 0         : True/False mask of crossing locations
    sign_changes = np.diff(np.sign(X), axis=1) != 0   # (samples, 127, 9)
    zcr = np.sum(sign_changes, axis=1) / (X.shape[1] - 1)  # (samples, 9)
    features.append(zcr)

    # ── Frequency-domain features (via FFT) ───────────────────────────────────
    # The Fast Fourier Transform (FFT) converts a time-series into a list of
    # frequencies and tells us how strongly each frequency is present.
    # np.fft.rfft is the "real" variant — since our sensor data is real-valued
    # it only returns the non-redundant half of the spectrum, giving 65 values
    # (indices 0–64) for a 128-point input.

    fft_vals   = np.fft.rfft(X, axis=1)        # (samples, 65, 9) — complex numbers
    fft_mag    = np.abs(fft_vals)               # (samples, 65, 9) — magnitudes only

    # SPECTRAL ENERGY — the sum of all squared magnitudes in the spectrum.
    # High spectral energy means the signal contains strong periodic components
    # (e.g. the rhythmic step pattern in walking or running).
    spectral_energy = np.sum(fft_mag ** 2, axis=1)   # (samples, 9)
    features.append(spectral_energy)

    # DOMINANT FREQUENCY INDEX — which frequency bin has the highest magnitude.
    # Index 0 is the "DC" component (just the average offset); higher indices
    # correspond to faster oscillations.  For walking you'd expect a low but
    # non-zero index matching the step cadence (~2 Hz).
    dominant_freq_idx = np.argmax(fft_mag, axis=1)   # (samples, 9)
    features.append(dominant_freq_idx)

    # ── Stack all features into one matrix ────────────────────────────────────
    # features is now a list of 8 arrays, each shaped (samples, 9).
    # np.concatenate along axis=1 places them side-by-side so every row becomes:
    #   [mean_ch0, mean_ch1, ..., mean_ch8,
    #    std_ch0,  std_ch1,  ..., std_ch8,
    #    ...
    #    dom_freq_ch0, ..., dom_freq_ch8]
    # → final shape: (samples, 8 features × 9 channels) = (samples, 72)
    return np.concatenate(features, axis=1)


# ── Apply to train and test sets ──────────────────────────────────────────────
print("\nExtracting features ...")
features_train = extract_features(X_train)
features_test  = extract_features(X_test)

# ── Print shapes ──────────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("Feature matrix shapes")
print("=" * 55)
print(f"features_train : {features_train.shape}")
print(f"  -> {features_train.shape[0]} windows, {features_train.shape[1]} features each")
n_time   = 6   # mean, std, min, max, rms, zcr
n_freq   = 2   # spectral energy, dominant freq index
n_chan   = X_train.shape[2]
print(f"  -> {n_time} time-domain + {n_freq} freq-domain features "
      f"x {n_chan} channels = {(n_time + n_freq) * n_chan} total")
print(f"features_test  : {features_test.shape}")
print(f"y_train        : {y_train.shape}")
print(f"y_test         : {y_test.shape}")
print("=" * 55)

# ── Save ──────────────────────────────────────────────────────────────────────
# We also re-save the labels alongside the features so every downstream script
# only needs to look in one place.
print("\nSaving to processed_data/ ...")
np.save(os.path.join(PROCESSED_DIR, "features_train.npy"), features_train)
np.save(os.path.join(PROCESSED_DIR, "features_test.npy"),  features_test)
np.save(os.path.join(PROCESSED_DIR, "y_train.npy"),        y_train)
np.save(os.path.join(PROCESSED_DIR, "y_test.npy"),         y_test)

for fname, arr in [
    ("features_train.npy", features_train),
    ("features_test.npy",  features_test),
    ("y_train.npy",        y_train),
    ("y_test.npy",         y_test),
]:
    print(f"  Saved {fname}  shape {arr.shape}")

print("Done.")
