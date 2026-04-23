import numpy as np
import tensorflow as tf

MODEL_PATH = "models/wake_word_model.h5"
TFLITE_PATH = "models/wake_word_model_quant.tflite"


def convert_to_tflite_int8(model_path, X_sample):
    """
    Convert Keras model to INT8 quantized TFLite model.
    
    INT8 quantization:
    - Reduces model size by ~4x
    - Speeds up inference on embedded hardware
    - Minimal accuracy loss
    
    Parameters:
        model_path : path to saved .h5 model
        X_sample   : representative dataset for calibration
    """
    # Load model
    model = tf.keras.models.load_model(model_path)
    print("Model loaded successfully")

    # Create TFLite converter
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # Enable INT8 quantization
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    # Representative dataset for calibration
    def representative_data_gen():
        for sample in X_sample[:100]:
            sample = sample[np.newaxis, ..., np.newaxis].astype(np.float32)
            yield [sample]

    converter.representative_dataset = representative_data_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    # Convert
    tflite_model = converter.convert()

    # Save
    with open(TFLITE_PATH, 'wb') as f:
        f.write(tflite_model)

    original_size = sum(
        p.numpy().nbytes for p in tf.keras.models.load_model(model_path).weights
    )
    quantized_size = len(tflite_model)

    print(f"TFLite model saved to {TFLITE_PATH}")
    print(f"Quantized model size: {quantized_size / 1024:.1f} KB")

    return tflite_model


if __name__ == "__main__":
    # Load sample data for calibration
    X = np.load("X_data.npy")
    convert_to_tflite_int8(MODEL_PATH, X)
    print("Quantization complete!")
