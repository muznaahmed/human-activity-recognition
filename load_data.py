import numpy as np
import os

# ── Path setup ──────────────────────────────────────────────────────────────
# Build the base path to the dataset folder.  os.path.join handles the
# backslash / forward-slash differences on Windows for us.
BASE_DIR = os.path.join(
    os.path.dirname(__file__),          # this script's folder  →  project root
    "human+activity+recognition+using+smartphones",
    "UCI HAR Dataset",
    "UCI HAR Dataset",
)

TRAIN_SIGNALS_DIR = os.path.join(BASE_DIR, "train", "Inertial Signals")
TEST_SIGNALS_DIR  = os.path.join(BASE_DIR, "test",  "Inertial Signals")

# ── Signal names ─────────────────────────────────────────────────────────────
# The dataset contains 9 sensor channels.  We list them in a fixed order so
# the final array always has channels in the same position regardless of how
# the OS returns file listings.
SIGNALS = [
    "body_acc_x",
    "body_acc_y",
    "body_acc_z",
    "body_gyro_x",
    "body_gyro_y",
    "body_gyro_z",
    "total_acc_x",
    "total_acc_y",
    "total_acc_z",
]

# ── Helper function ───────────────────────────────────────────────────────────
def load_signals(signals_dir: str, split: str) -> np.ndarray:
    """
    Load all 9 signal files for one split (train or test) and stack them.

    Each file contains one row per time-window.  Every row has 128 whitespace-
    separated numbers – one reading per time-step inside the 2.56-second window.

    After loading all 9 files we end up with a list of nine 2-D arrays each
    shaped (samples, 128).  np.stack turns that list into a single 3-D array
    shaped (samples, 128, 9) – samples first, then time-steps, then channels.

    Parameters
    ----------
    signals_dir : str   Folder that contains the .txt signal files.
    split       : str   "train" or "test" – used to build the file names.

    Returns
    -------
    np.ndarray of shape (samples, 128, 9), dtype float64
    """
    all_channels = []

    for signal_name in SIGNALS:
        # Build the full path to this channel's file, e.g.
        #   .../Inertial Signals/body_acc_x_train.txt
        file_path = os.path.join(signals_dir, f"{signal_name}_{split}.txt")

        # np.loadtxt reads a plain-text matrix where each row is one sample
        # and columns are the 128 time-steps within that window.
        # Result shape: (n_samples, 128)
        channel_data = np.loadtxt(file_path)

        print(f"  Loaded {signal_name}_{split}.txt  shape {channel_data.shape}")
        all_channels.append(channel_data)

    # Stack along a NEW third axis (axis=2) so each channel becomes one
    # "slice" of the last dimension.
    # Before: list of 9 arrays, each (n_samples, 128)
    # After:  single array of shape (n_samples, 128, 9)
    stacked = np.stack(all_channels, axis=2)
    return stacked


# ── Load training inertial signals ───────────────────────────────────────────
print("Loading TRAIN signals …")
X_train = load_signals(TRAIN_SIGNALS_DIR, "train")

# ── Load test inertial signals ────────────────────────────────────────────────
print("\nLoading TEST signals …")
X_test = load_signals(TEST_SIGNALS_DIR, "test")

# ── Load labels ───────────────────────────────────────────────────────────────
# y_train.txt and y_test.txt each contain one integer per line (1-6) that
# identifies the activity performed during the corresponding time-window.
# Activities: 1-WALKING, 2-WALKING_UPSTAIRS, 3-WALKING_DOWNSTAIRS,
#             4-SITTING, 5-STANDING, 6-LAYING
print("\nLoading labels …")
y_train_path = os.path.join(BASE_DIR, "train", "y_train.txt")
y_test_path  = os.path.join(BASE_DIR, "test",  "y_test.txt")

# dtype=int gives us clean integer labels instead of floats
y_train = np.loadtxt(y_train_path, dtype=int)
y_test  = np.loadtxt(y_test_path,  dtype=int)

# ── Combine train + test into one dataset ─────────────────────────────────────
# np.concatenate joins arrays along an existing axis (axis 0 = samples).
# This gives us one big array covering all participants / windows.
X_all = np.concatenate([X_train, X_test], axis=0)
y_all = np.concatenate([y_train, y_test], axis=0)

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 50)
print("Shape summary")
print("=" * 50)
print(f"X_train : {X_train.shape}   (train_samples, 128 time-steps, 9 channels)")
print(f"X_test  : {X_test.shape}    (test_samples,  128 time-steps, 9 channels)")
print(f"X_all   : {X_all.shape}     (total_samples, 128 time-steps, 9 channels)")
print(f"y_train : {y_train.shape}   one activity label per train window")
print(f"y_test  : {y_test.shape}    one activity label per test window")
print(f"y_all   : {y_all.shape}     combined labels")
print(f"\nUnique activity labels: {sorted(np.unique(y_all))}")
print("=" * 50)

# ── Save to disk ──────────────────────────────────────────────────────────────
# Create the output folder next to this script.  exist_ok=True means Python
# won't complain if the folder already exists — safe to run multiple times.
SAVE_DIR = os.path.join(os.path.dirname(__file__), "processed_data")
os.makedirs(SAVE_DIR, exist_ok=True)

# np.save writes a single array to a binary .npy file.  Loading it back later
# with np.load is fast and preserves the exact dtype and shape.
files_to_save = {
    "X_train.npy": X_train,
    "X_test.npy":  X_test,
    "y_train.npy": y_train,
    "y_test.npy":  y_test,
}

print("\nSaving arrays to processed_data/ ...")
for filename, array in files_to_save.items():
    save_path = os.path.join(SAVE_DIR, filename)
    np.save(save_path, array)
    print(f"  Saved {filename}  shape {array.shape}")

print("Done. All files written to:", SAVE_DIR)
