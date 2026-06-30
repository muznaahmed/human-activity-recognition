import os
import sys
import time

# Suppress TensorFlow's C-level info/warning messages before importing it.
# Level 3 = only show fatal errors; without this TF floods the terminal with
# hardware-detection messages that have nothing to do with our model.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import classification_report, confusion_matrix

# Also silence TF's Python-level logger
tf.get_logger().setLevel("ERROR")


# ── Tee: mirror every print() to both terminal and file ──────────────────────
# Identical to the pattern used in train_classifiers.py.
# Replacing sys.stdout with this object means all print() calls automatically
# go to both destinations without any other code changes.
class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            try:
                s.write(data)
            except UnicodeEncodeError:
                # The Windows terminal uses cp1252 which can't render Keras's
                # Unicode box-drawing characters.  Replace unrepresentable chars
                # with '?' so the terminal keeps working while the file (UTF-8)
                # still gets the original text.
                enc  = getattr(s, "encoding", "ascii") or "ascii"
                safe = data.encode(enc, errors="replace").decode(enc)
                s.write(safe)

    def flush(self):
        for s in self.streams:
            s.flush()


# ── Paths ─────────────────────────────────────────────────────────────────────
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "processed_data")
RESULTS_PATH  = os.path.join(PROCESSED_DIR, "results_lstm.txt")

# Open the results file and activate the tee so everything printed from here
# onwards is captured in the file as well as shown in the terminal.
_results_file = open(RESULTS_PATH, "w", encoding="utf-8")
sys.stdout = Tee(sys.__stdout__, _results_file)


# ── Load raw windowed data ────────────────────────────────────────────────────
# Unlike the classical models we load X_train/X_test (shape: samples × 128 × 9)
# rather than the hand-crafted feature matrices.  The LSTM will learn its own
# patterns directly from the raw time-series.
print("Loading data ...")
X_train = np.load(os.path.join(PROCESSED_DIR, "X_train.npy"))   # (7352, 128, 9)
X_test  = np.load(os.path.join(PROCESSED_DIR, "X_test.npy"))    # (2947, 128, 9)
y_train = np.load(os.path.join(PROCESSED_DIR, "y_train.npy"))   # (7352,)  labels 1-6
y_test  = np.load(os.path.join(PROCESSED_DIR, "y_test.npy"))    # (2947,)  labels 1-6

print(f"  X_train : {X_train.shape}")
print(f"  X_test  : {X_test.shape}")
print(f"  y_train : {y_train.shape}  labels {np.unique(y_train)}")
print(f"  y_test  : {y_test.shape}   labels {np.unique(y_test)}")

# ── Shift labels from 1-6 to 0-5 ─────────────────────────────────────────────
# Keras's sparse_categorical_crossentropy expects class indices that start at 0.
# Label 1 (WALKING) becomes 0, label 6 (LAYING) becomes 5, and so on.
y_train_shifted = y_train - 1
y_test_shifted  = y_test  - 1

# Activity names in shifted-label order (index 0 = original label 1, etc.)
ACTIVITY_NAMES = [
    "WALKING",             # 0
    "WALKING_UPSTAIRS",    # 1
    "WALKING_DOWNSTAIRS",  # 2
    "SITTING",             # 3
    "STANDING",            # 4
    "LAYING",              # 5
]

N_CLASSES    = 6
N_TIMESTEPS  = X_train.shape[1]   # 128
N_CHANNELS   = X_train.shape[2]   # 9
EPOCHS       = 20
BATCH_SIZE   = 64


# ── Build the LSTM model ──────────────────────────────────────────────────────
# An LSTM (Long Short-Term Memory) is a type of recurrent neural network that
# reads the input one time-step at a time and keeps a "memory" of what it has
# seen so far.  This makes it naturally suited to time-series data like sensor
# readings where the order of values matters.
#
# Model structure:
#   Input         (128 time-steps, 9 channels per step)
#       ↓
#   LSTM(64)      reads the sequence and outputs a single 64-dimensional summary
#       ↓
#   Dropout(0.5)  randomly zeros 50% of neurons during training to prevent the
#                 model from memorising the training set (overfitting)
#       ↓
#   Dense(6)      one output neuron per class; softmax turns raw scores into
#                 probabilities that sum to 1
print("\nBuilding model ...")
model = keras.Sequential([
    keras.layers.Input(shape=(N_TIMESTEPS, N_CHANNELS)),

    # LSTM layer: 64 units means the layer compresses each 128-step sequence
    # into a single vector of 64 numbers capturing the most important patterns.
    keras.layers.LSTM(64),

    # Dropout: during each training batch, randomly ignore 50% of the LSTM
    # outputs.  This forces the model to learn redundant representations and
    # generalise better to unseen data.
    keras.layers.Dropout(0.5),

    # Dense output: 6 neurons (one per activity), softmax activation gives
    # a probability distribution — the predicted class is the highest one.
    keras.layers.Dense(N_CLASSES, activation="softmax"),
])

# ── Compile ───────────────────────────────────────────────────────────────────
# Adam is an adaptive learning-rate optimiser — it adjusts how big each update
# step is based on recent gradients, making training faster and more stable
# than plain stochastic gradient descent.
#
# sparse_categorical_crossentropy is the standard loss for multi-class
# classification when labels are integers (not one-hot vectors).  Lower loss
# means the model's probability distributions are closer to the true labels.
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

# Print a summary table: layer names, output shapes, and parameter counts.
# This lets us sanity-check the architecture before training.
model.summary(print_fn=print)


# ── Custom epoch callback ─────────────────────────────────────────────────────
# Keras's built-in progress bar writes directly to sys.stderr using special
# terminal codes, so it would look garbled in results_lstm.txt.  Instead we
# use verbose=0 (silent Keras) and print our own clean line after each epoch
# via sys.stdout — which goes through the Tee and lands in the file as well.
class EpochLogger(keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        print(f"\n  {'Epoch':>6}  {'loss':>8}  {'acc':>8}  {'val_loss':>10}  {'val_acc':>9}")
        print(f"  {'-'*50}")

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        print(
            f"  {epoch + 1:>6d}  "
            f"{logs.get('loss', 0):>8.4f}  "
            f"{logs.get('accuracy', 0):>8.4f}  "
            f"{logs.get('val_loss', 0):>10.4f}  "
            f"{logs.get('val_accuracy', 0):>9.4f}"
        )


# ── Train ─────────────────────────────────────────────────────────────────────
# We pass X_test / y_test_shifted as validation data so we can watch how the
# model performs on unseen data after every epoch.
# Note: in a real project you would use a separate validation split and only
# touch the test set once at the very end.
print(f"\nTraining for {EPOCHS} epochs (batch size {BATCH_SIZE}) ...")
t0 = time.perf_counter()

history = model.fit(
    X_train, y_train_shifted,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(X_test, y_test_shifted),
    verbose=0,                      # silence Keras's own output
    callbacks=[EpochLogger()],      # use our clean per-epoch printer instead
)

train_time = time.perf_counter() - t0
print(f"\n  Total training time: {train_time:.2f} seconds")


# ── Evaluate ──────────────────────────────────────────────────────────────────
print("\nEvaluating on test set ...")
# model.predict returns an array of shape (samples, 6) — one probability per
# class per window.  np.argmax picks the class with the highest probability.
y_prob = model.predict(X_test, verbose=0)           # (2947, 6)  probabilities
y_pred = np.argmax(y_prob, axis=1)                  # (2947,)    predicted class 0-5

test_loss, test_acc = model.evaluate(X_test, y_test_shifted, verbose=0)

print(f"\n{'=' * 60}")
print("  LSTM Results")
print(f"{'=' * 60}")
print(f"  Training time : {train_time:.2f} seconds")
print(f"  Test loss     : {test_loss:.4f}")
print(f"  Test accuracy : {test_acc * 100:.2f}%")

# CLASSIFICATION REPORT — precision / recall / f1 per activity class
print(f"\n  Classification report:")
print(classification_report(
    y_test_shifted, y_pred,
    labels=list(range(N_CLASSES)),
    target_names=ACTIVITY_NAMES,
    digits=4,
))

# CONFUSION MATRIX — rows = true label, cols = predicted label
cm = confusion_matrix(y_test_shifted, y_pred, labels=list(range(N_CLASSES)))
print("  Confusion matrix (rows=true, cols=predicted):")

abbrev = ["WALK", "W_UP", "W_DN", "SIT ", "STND", "LAY "]
header = "              " + "  ".join(f"{a:>6}" for a in abbrev)
print(header)
for i, row in enumerate(cm):
    print(f"  {abbrev[i]} (true) |  " + "  ".join(f"{v:6d}" for v in row))

print(f"{'=' * 60}")


# ── Restore stdout and close file ─────────────────────────────────────────────
sys.stdout = sys.__stdout__
_results_file.close()
print(f"\nResults also saved to: {RESULTS_PATH}")
