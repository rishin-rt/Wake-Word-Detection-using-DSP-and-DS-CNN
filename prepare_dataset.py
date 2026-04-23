import os
import numpy as np
import librosa
from mfcc_features import extract_mfcc

# Configuration
SAMPLE_RATE = 16000
N_MFCC = 13
MAX_TIME_STEPS = 100  # Pad/truncate all samples to this length
DATA_DIR = "data/"
WAKE_WORD_DIR = os.path.join(DATA_DIR, "wake_word")
BACKGROUND_DIR = os.path.join(DATA_DIR, "background")


def pad_or_truncate(mfcc, max_len=MAX_TIME_STEPS):
    """Pad or truncate MFCC to fixed length."""
    if mfcc.shape[0] < max_len:
        # Pad with zeros
        pad_width = max_len - mfcc.shape[0]
        mfcc = np.pad(mfcc, ((0, pad_width), (0, 0)), mode='constant')
    else:
        # Truncate
        mfcc = mfcc[:max_len, :]
    return mfcc


def load_dataset():
    """
    Load all audio files and extract MFCC features.
    
    Returns:
        X : feature array of shape (num_samples, MAX_TIME_STEPS, N_MFCC)
        y : label array (1 = wake word, 0 = background)
    """
    X = []
    y = []

    print("Loading wake word samples...")
    for filename in os.listdir(WAKE_WORD_DIR):
        if filename.endswith(".wav"):
            path = os.path.join(WAKE_WORD_DIR, filename)
            try:
                mfcc = extract_mfcc(path)
                mfcc = pad_or_truncate(mfcc)
                X.append(mfcc)
                y.append(1)  # Wake word label
            except Exception as e:
                print(f"Error loading {filename}: {e}")

    print(f"Loaded {len(y)} wake word samples")

    print("Loading background/noise samples...")
    for filename in os.listdir(BACKGROUND_DIR):
        if filename.endswith(".wav"):
            path = os.path.join(BACKGROUND_DIR, filename)
            try:
                mfcc = extract_mfcc(path)
                mfcc = pad_or_truncate(mfcc)
                X.append(mfcc)
                y.append(0)  # Background label
            except Exception as e:
                print(f"Error loading {filename}: {e}")

    print(f"Total samples loaded: {len(y)}")

    X = np.array(X)
    y = np.array(y)

    # Shuffle dataset
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]

    return X, y


if __name__ == "__main__":
    X, y = load_dataset()
    print(f"Dataset shape: {X.shape}")
    print(f"Wake word samples: {np.sum(y == 1)}")
    print(f"Background samples: {np.sum(y == 0)}")

    # Save dataset
    np.save("X_data.npy", X)
    np.save("y_data.npy", y)
    print("Dataset saved!")
