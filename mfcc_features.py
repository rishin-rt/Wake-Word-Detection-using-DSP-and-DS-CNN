import numpy as np
import librosa

def extract_mfcc(audio_path, sample_rate=16000, n_mfcc=13, frame_size=0.025, hop_size=0.010):
    """
    Extract MFCC features from an audio file.
    
    Parameters:
        audio_path  : path to audio file
        sample_rate : 16000 Hz
        n_mfcc      : number of MFCC coefficients (13)
        frame_size  : frame duration in seconds (25ms)
        hop_size    : hop duration in seconds (10ms)
    
    Returns:
        mfcc features as numpy array
    """
    # Load audio
    audio, sr = librosa.load(audio_path, sr=sample_rate)

    # Convert frame size and hop size to samples
    n_fft = int(frame_size * sr)       # 400 samples for 25ms
    hop_length = int(hop_size * sr)    # 160 samples for 10ms

    # Apply pre-emphasis filter
    pre_emphasis = 0.97
    audio = np.append(audio[0], audio[1:] - pre_emphasis * audio[:-1])

    # Extract MFCC
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=n_mfcc,
        n_fft=n_fft,
        hop_length=hop_length,
        window='hamming'
    )

    # Normalize
    mfcc = (mfcc - np.mean(mfcc)) / (np.std(mfcc) + 1e-8)

    return mfcc.T  # Shape: (time_steps, n_mfcc)


def extract_mfcc_from_array(audio, sample_rate=16000, n_mfcc=13):
    """
    Extract MFCC features from a numpy audio array.
    Used for real-time detection.
    """
    n_fft = int(0.025 * sample_rate)
    hop_length = int(0.010 * sample_rate)

    mfcc = librosa.feature.mfcc(
        y=audio.astype(float),
        sr=sample_rate,
        n_mfcc=n_mfcc,
        n_fft=n_fft,
        hop_length=hop_length,
        window='hamming'
    )

    mfcc = (mfcc - np.mean(mfcc)) / (np.std(mfcc) + 1e-8)
    return mfcc.T
