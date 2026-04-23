# Wake-Word-Detection-using-DSP-and-DS-CNN
Real-time wake word detection using DSP and DS-CNN
# 🎙️ Real-Time Wake Word Detection System

A real-time wake word detection system using Digital Signal Processing (DSP) and Deep Learning, implemented in Python with TensorFlow Lite. The system continuously monitors audio input and activates upon detecting a predefined keyword with **85–95% accuracy** and **latency below 100ms**.

---

## 🧠 Overview

This project combines DSP techniques with a lightweight DS-CNN (Depthwise Separable Convolutional Neural Network) to build an efficient, real-time wake word detection system suitable for embedded and IoT applications.

The system was trained on a **self-recorded custom dataset** and optimized using INT8 quantization for low-memory deployment.

---

## ⚙️ DSP Pipeline

```
Microphone Input
      ↓
  Framing (25ms frames, 10ms hop)
      ↓
  Hamming Window
      ↓
  Fast Fourier Transform (FFT)
      ↓
  Mel Filter Bank
      ↓
  MFCC Feature Extraction (13 coefficients)
      ↓
  DS-CNN Model Inference
      ↓
  Wake Word Detected ✅
```

---

## 🏗️ Model Architecture

- **Architecture**: Depthwise Separable CNN (DS-CNN)
- **Input**: 13 MFCC features per frame
- **Optimization**: INT8 quantization for embedded deployment
- **Runtime**: TensorFlow Lite
- **Accuracy**: 85–95%
- **Latency**: < 100ms

### Why DS-CNN?
Depthwise Separable Convolutions reduce computational complexity significantly compared to standard CNNs by separating spatial and channel-wise convolutions — ideal for real-time embedded applications.

---

## 📊 Technical Specifications

| Parameter | Value |
|-----------|-------|
| Sampling Rate | 16 kHz |
| Frame Size | 25 ms |
| Hop Size | 10 ms |
| MFCC Coefficients | 13 |
| Model | DS-CNN |
| Quantization | INT8 |
| Accuracy | 85–95% |
| Latency | < 100ms |

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python | Core implementation |
| TensorFlow / TensorFlow Lite | Model training and inference |
| Librosa | Audio processing and MFCC extraction |
| NumPy | Numerical computations |
| SciPy | Signal processing |

---

## 📁 Project Structure

```
wake-word-detection/
│
├── data/                   # Audio dataset (self-recorded)
│   ├── wake_word/          # Positive samples
│   └── background/         # Negative samples
│
├── preprocessing/
│   ├── framing.py          # Audio framing
│   ├── windowing.py        # Hamming window
│   ├── fft.py              # FFT computation
│   └── mfcc.py             # MFCC extraction
│
├── model/
│   ├── ds_cnn.py           # DS-CNN architecture
│   ├── train.py            # Model training
│   └── quantize.py         # INT8 quantization
│
├── inference/
│   └── realtime_detect.py  # Real-time detection pipeline
│
├── models/
│   ├── model.h5            # Trained Keras model
│   └── model_quant.tflite  # Quantized TFLite model
│
└── README.md
```

---

## 🚀 How to Run

### 1. Install Dependencies

```bash
pip install tensorflow librosa numpy scipy sounddevice
```

### 2. Prepare Dataset

Record your own wake word samples or use existing audio files. Place them in the `data/` directory.

### 3. Extract Features

```bash
python preprocessing/mfcc.py
```

### 4. Train the Model

```bash
python model/train.py
```

### 5. Quantize for Deployment

```bash
python model/quantize.py
```

### 6. Run Real-Time Detection

```bash
python inference/realtime_detect.py
```

---

## 📈 Results

| Metric | Value |
|--------|-------|
| Training Accuracy | ~92% |
| Test Accuracy | 85–95% |
| False Positive Rate | Low |
| Inference Latency | < 100ms |
| Model Size (quantized) | Significantly reduced vs full model |

---

## 🔬 Key Features

- ✅ Real-time audio processing pipeline
- ✅ MFCC-based feature extraction mimicking human auditory perception
- ✅ Lightweight DS-CNN architecture for embedded systems
- ✅ INT8 quantization for reduced memory footprint
- ✅ Custom self-recorded dataset
- ✅ Sliding window approach for temporal continuity
- ✅ Suitable for IoT and embedded deployment


---

## 🔮 Future Scope

- Deploy on Raspberry Pi or STM32 microcontroller
- Expand dataset with more speakers and noise conditions
- Implement noise cancellation preprocessing
- Support multiple wake words
- Mobile application integration

---

## 📚 References

- Librosa Documentation — https://librosa.org
- TensorFlow Lite Guide — https://tensorflow.org/lite
- MFCCs for Machine Learning — Davis & Mermelstein, 1980
- DS-CNN for Keyword Spotting — Zhang et al., 2017

---

## 📄 License

This project is for academic purposes under KIIT University.
