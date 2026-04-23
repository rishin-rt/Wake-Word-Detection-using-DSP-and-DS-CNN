import numpy as np
import sounddevice as sd
import tensorflow as tf
from mfcc_features import extract_mfcc_from_array

# Configuration
SAMPLE_RATE = 16000
CHUNK_DURATION = 1.0          # 1 second window
CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_DURATION)
DETECTION_THRESHOLD = 0.85    # Confidence threshold
TFLITE_MODEL_PATH = "models/wake_word_model_quant.tflite"
MAX_TIME_STEPS = 100
N_MFCC = 13


def load_tflite_model(model_path):
    """Load quantized TFLite model."""
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter


def predict(interpreter, mfcc_features):
    """
    Run inference on MFCC features using TFLite model.
    
    Returns:
        probability of wake word detection
    """
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Prepare input
    input_data = mfcc_features[np.newaxis, ..., np.newaxis].astype(np.float32)

    # Resize if needed
    if input_data.shape[1] < MAX_TIME_STEPS:
        pad_width = MAX_TIME_STEPS - input_data.shape[1]
        input_data = np.pad(input_data, ((0,0),(0,pad_width),(0,0),(0,0)))
    else:
        input_data = input_data[:, :MAX_TIME_STEPS, :, :]

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    output = interpreter.get_tensor(output_details[0]['index'])
    return output[0][1]  # Probability of wake word class


def realtime_detection():
    """
    Continuously capture audio and detect wake word in real-time.
    Uses sliding window approach for temporal continuity.
    """
    print("Loading model...")
    interpreter = load_tflite_model(TFLITE_MODEL_PATH)
    print("Model loaded!")
    print(f"Listening for wake word... (threshold: {DETECTION_THRESHOLD})")
    print("Press Ctrl+C to stop\n")

    audio_buffer = np.zeros(CHUNK_SAMPLES)

    def audio_callback(indata, frames, time, status):
        nonlocal audio_buffer

        if status:
            print(f"Audio status: {status}")

        # Update sliding window buffer
        new_audio = indata[:, 0]
        audio_buffer = np.roll(audio_buffer, -len(new_audio))
        audio_buffer[-len(new_audio):] = new_audio

        # Extract MFCC features
        mfcc = extract_mfcc_from_array(audio_buffer, SAMPLE_RATE, N_MFCC)

        # Run inference
        confidence = predict(interpreter, mfcc)

        # Detection decision
        if confidence >= DETECTION_THRESHOLD:
            print(f"🎙️  WAKE WORD DETECTED! Confidence: {confidence:.2%}")

    # Start real-time audio stream
    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype='float32',
            blocksize=int(SAMPLE_RATE * 0.1),  # 100ms blocks
            callback=audio_callback
        ):
            print("Audio stream started...")
            while True:
                sd.sleep(100)

    except KeyboardInterrupt:
        print("\nDetection stopped.")


if __name__ == "__main__":
    realtime_detection()
