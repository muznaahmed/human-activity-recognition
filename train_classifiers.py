import numpy as np
import os
import sys
import time

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier

# ── Tee: write to terminal AND a file at the same time ───────────────────────
# Normally print() only writes to sys.stdout (the terminal).  By replacing
# sys.stdout with this Tee object, every print() call automatically goes to
# both the terminal and the results file — no other code needs to change.
class Tee:
    def __init__(self, *streams):
        self.streams = streams          # tuple of file-like objects to write to

    def write(self, data):
        for s in self.streams:
            s.write(data)

    def flush(self):                    # needed so progress shows up immediately
        for s in self.streams:
            s.flush()

# ── Paths ─────────────────────────────────────────────────────────────────────
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "processed_data")

# ── Open results file and activate tee ───────────────────────────────────────
# utf-8 encoding so special characters (dashes, arrows) don't cause issues on
# Windows where the default file encoding is cp1252.
RESULTS_PATH = os.path.join(PROCESSED_DIR, "results_classical.txt")
_results_file = open(RESULTS_PATH, "w", encoding="utf-8")
sys.stdout = Tee(sys.__stdout__, _results_file)   # from here every print() goes to both

# ── Load features and labels ──────────────────────────────────────────────────
# These were saved by extract_features.py.
# features_train/test: shape (samples, 72)  — one row per window, 72 features
# y_train/test:        shape (samples,)     — activity label 1-6 per window
print("Loading features ...")
features_train = np.load(os.path.join(PROCESSED_DIR, "features_train.npy"))
features_test  = np.load(os.path.join(PROCESSED_DIR, "features_test.npy"))
y_train        = np.load(os.path.join(PROCESSED_DIR, "y_train.npy"))
y_test         = np.load(os.path.join(PROCESSED_DIR, "y_test.npy"))

print(f"  features_train : {features_train.shape}")
print(f"  features_test  : {features_test.shape}")
print(f"  y_train        : {y_train.shape}  labels {np.unique(y_train)}")
print(f"  y_test         : {y_test.shape}   labels {np.unique(y_test)}")

# The six activities in label order (1-6).  Used in the classification report
# so we see human-readable names instead of just numbers.
ACTIVITY_NAMES = [
    "WALKING",             # 1
    "WALKING_UPSTAIRS",    # 2
    "WALKING_DOWNSTAIRS",  # 3
    "SITTING",             # 4
    "STANDING",            # 5
    "LAYING",              # 6
]


# ── Helper: print results ─────────────────────────────────────────────────────
def print_results(model_name: str, y_true: np.ndarray, y_pred: np.ndarray,
                  train_seconds: float) -> None:
    """Print accuracy, classification report, and confusion matrix."""

    print(f"\n{'=' * 60}")
    print(f"  {model_name}")
    print(f"{'=' * 60}")
    print(f"  Training time : {train_seconds:.2f} seconds")
    print(f"  Test accuracy : {accuracy_score(y_true, y_pred) * 100:.2f}%")

    # CLASSIFICATION REPORT ───────────────────────────────────────────────────
    # For each activity class this shows:
    #   precision  — of all windows the model labelled as X, how many really were X?
    #   recall     — of all windows that really were X, how many did the model find?
    #   f1-score   — a single number balancing precision and recall (higher = better)
    #   support    — how many true windows of this class are in the test set
    print(f"\n  Classification report:")
    # labels=[1..6] and target_names keep the report aligned with our 1-based labels
    print(classification_report(
        y_true, y_pred,
        labels=list(range(1, 7)),
        target_names=ACTIVITY_NAMES,
        digits=4,
    ))

    # CONFUSION MATRIX ────────────────────────────────────────────────────────
    # A 6x6 grid where:
    #   rows  = the TRUE activity
    #   cols  = what the model PREDICTED
    # Perfect predictions would put all numbers on the diagonal and zeros elsewhere.
    # Off-diagonal values show which activities get confused with each other.
    cm = confusion_matrix(y_true, y_pred, labels=list(range(1, 7)))
    print("  Confusion matrix (rows=true, cols=predicted):")

    # Print column header with abbreviated activity names so it fits on screen
    abbrev = ["WALK", "W_UP", "W_DN", "SIT ", "STND", "LAY "]
    header = "              " + "  ".join(f"{a:>6}" for a in abbrev)
    print(header)

    for i, row in enumerate(cm):
        row_label = f"  {abbrev[i]} (true) |"
        row_vals  = "  ".join(f"{v:6d}" for v in row)
        print(row_label + " " + row_vals)


# ── 1. Random Forest ──────────────────────────────────────────────────────────
# A Random Forest builds many decision trees (here 200), each trained on a
# random subset of the training data and a random subset of features.  The
# final prediction is a majority vote across all trees.
#
# Why it works well for sensor data: it captures non-linear patterns and
# handles the mix of time-domain and frequency-domain features without needing
# any scaling.
#
# n_estimators=200  : number of trees — more trees = more stable, slower
# n_jobs=-1         : use all CPU cores in parallel so it trains faster
# random_state=42   : fixes the random seed so results are reproducible
print("\n\nTraining Random Forest ...")
rf = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42)

t0 = time.perf_counter()
rf.fit(features_train, y_train)
rf_time = time.perf_counter() - t0

rf_preds = rf.predict(features_test)
print_results("Random Forest (200 trees)", y_test, rf_preds, rf_time)


# ── 2. SVM with RBF kernel ────────────────────────────────────────────────────
# A Support Vector Machine (SVM) finds the boundary (hyperplane) that best
# separates the classes while maximising the margin between them.
#
# The RBF (Radial Basis Function) kernel lets the SVM draw curved boundaries,
# making it powerful for data that isn't linearly separable.
#
# One important note: SVM is sensitive to feature scale — features with large
# values can dominate.  For a fair comparison we leave scaling out here since
# all our features are sensor readings in similar ranges, but in production
# you'd typically apply sklearn's StandardScaler first.
#
# C=10    : how much to penalise misclassifications (higher = tighter fit)
# gamma='scale' : automatically sets the RBF width based on data variance
print("\n\nTraining SVM (RBF kernel) ...")
svm = SVC(kernel="rbf", C=10, gamma="scale", random_state=42)

t0 = time.perf_counter()
svm.fit(features_train, y_train)
svm_time = time.perf_counter() - t0

svm_preds = svm.predict(features_test)
print_results("SVM — RBF kernel (C=10)", y_test, svm_preds, svm_time)


# ── 3. XGBoost ────────────────────────────────────────────────────────────────
# XGBoost (eXtreme Gradient Boosting) builds trees sequentially: each new tree
# tries to correct the mistakes made by all previous trees.  This "boosting"
# strategy often achieves state-of-the-art results on tabular/feature data.
#
# IMPORTANT — label shift:
# XGBoost requires class labels to start at 0.  Our labels are 1-6, so we
# subtract 1 before training (giving 0-5) and add 1 back to the predictions
# so everything else in this script stays consistent with the 1-6 convention.
#
# n_estimators=300   : number of boosting rounds (trees)
# learning_rate=0.1  : how much each tree corrects the previous ones
# max_depth=6        : maximum depth of each tree (controls overfitting)
# use_label_encoder  : silences a deprecation warning in recent XGBoost versions
# eval_metric        : which metric XGBoost uses internally (mlogloss = log loss)
print("\n\nTraining XGBoost ...")
xgb = XGBClassifier(
    n_estimators=300,
    learning_rate=0.1,
    max_depth=6,
    n_jobs=-1,
    random_state=42,
    eval_metric="mlogloss",
    verbosity=0,          # suppress XGBoost's own progress output
)

# Shift labels: 1-6  →  0-5
y_train_xgb = y_train - 1
y_test_xgb  = y_test  - 1

t0 = time.perf_counter()
xgb.fit(features_train, y_train_xgb)
xgb_time = time.perf_counter() - t0

# Shift predictions back: 0-5  →  1-6  so the report uses original label names
xgb_preds = xgb.predict(features_test) + 1
print_results("XGBoost (300 rounds)", y_test, xgb_preds, xgb_time)


# ── Summary comparison ────────────────────────────────────────────────────────
print(f"\n\n{'=' * 60}")
print("  Summary")
print(f"{'=' * 60}")
print(f"  {'Model':<30} {'Accuracy':>10}  {'Train time':>12}")
print(f"  {'-'*54}")
for name, preds, t in [
    ("Random Forest (200 trees)",  rf_preds,  rf_time),
    ("SVM — RBF kernel (C=10)",    svm_preds, svm_time),
    ("XGBoost (300 rounds)",       xgb_preds, xgb_time),
]:
    acc = accuracy_score(y_test, preds) * 100
    print(f"  {name:<30} {acc:>9.2f}%  {t:>10.2f}s")
print(f"{'=' * 60}")

# ── Restore stdout and close the file ────────────────────────────────────────
# Always restore sys.stdout before closing the file so any error messages
# after this point still appear in the terminal.
sys.stdout = sys.__stdout__
_results_file.close()
print(f"\nResults also saved to: {RESULTS_PATH}")
