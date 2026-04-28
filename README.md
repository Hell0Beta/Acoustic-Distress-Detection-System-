

<div align="center">
  <img src="https://img.shields.io/badge/Status-Production%20Ready-brightgreen?style=for-the-badge" alt="Status">
  <img src="https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white" alt="TensorFlow">
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/Scikit_Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="Scikit-Learn">

  <h1 align="center">🎧 Acoustic Distress Detection System (ADDS)</h1>
  <p align="center">
    <b>A Multi-Modal, Late-Fusion AI Pipeline for Real-Time Safety Monitoring</b><br>
    <i>Late Fusion of YAMNet, Wav2Vec2, and Faster-Whisper via a Skew-Optimized SVM Meta-Classifier</i>
  </p>
</div>

---

### 🎥 Demo / How It Works

*(Note: To add a video, upload an MP4 to YouTube or Loom as "Unlisted", and replace the `src` link below!)*

<div align="center">
  <video src="[https://www.youtube.com/embed/dQw4w9WgXcQ](https://youtu.be/IzRa2TLypRI?si=VQgODai3O4OwuQs_)" width="640" height="360" controls></video>
  <br><i>Alternatively: Convert a short screen-recording to a .GIF and use: <code>![Demo](demo.gif)</code></i>
</div>

---

### 🧠 Architecture Overview

The system does not rely on a single AI model. Instead, it uses a "Committee of Experts" approach. Audio is chunked into 2-second segments and pushed through three parallel deep learning models simultaneously using Python Threading.

1. **🔊 YAMNet:** Analyzes the raw audio waveform for environmental acoustic events (screams, glass breaking).
2. **😊 Wav2Vec2-Large-XLSR (SUPERB):** Extracts Speech Emotion Recognition (SER), specifically looking for the negative-affect cluster (Anger, Fear, Sadness).
3. **🗣️ Faster-Whisper:** Transcribes speech to text to look for lexical distress keywords ("help me", "stop"). *Conditionally gated by YAMNet to save compute.*

These raw probabilities are mathematically combined (Interaction Features) and temporal trends are calculated (Deltas), creating a 10-dimensional vector fed to the final decision layer.

<div align="center">
  <img src="https://img.shields.io/badge/Final_Model-SVM_(Linear)_blue" alt="Final Model">
  <img src="https://img.shields.io/badge/Recall-83%25-brightgreen" alt="Recall">
  <img src="https://img.shields.io/badge/F1_Score-0.73-yellow" alt="F1 Score">
</div>

---

### 📊 Why Support Vector Machine over Random Forest?

Our dataset was heavily skewed (Safe $\gg$ Distress). While a Random Forest baseline achieved an overall accuracy of **81.48%**, it missed 15 out of 58 real distress events (74% Recall). 

We shifted to a `class_weight='balanced'` Linear SVM. It successfully recovered 9 of those missed events, pushing Distress Recall to **83%**. 
*In safety-critical systems, a False Negative (missing a scream) is a catastrophic failure, making Recall our primary optimization metric over overall accuracy.*

| Metric | Random Forest | SVM (Selected) | Why it matters |
| :--- | :---: | :---: | :--- |
| **Accuracy** | 81.48% | 78.40% | RF wins overall, but misses emergencies. |
| **Distress Recall** | 74% | **83%** | SVM catches significantly more real danger. |
| **False Positives** | 15 | 25 | SVM is more aggressive (acceptable trade-off). |

---

### 🛠️ Local Setup & Testing

Because browser-based Javascript audio bridges are highly unstable, we built a robust local testing pipeline.

**1. Install Local Dependencies:**
```bash
pip install sounddevice rich wave
```

**2. Record Audio (Creates a perfectly formatted 16kHz WAV):**
```bash
python record.py ./audio_samples/ --duration 5
```
*(This uses `sounddevice` to bypass OS-level audio compression/limiting issues).*

**3. Run Through Pipeline (In Colab or Local):**
Upload the generated `.wav` file into the testing cell. The script automatically chunks any length of audio into 2-second intervals, runs the multi-threaded pipeline, and outputs a timeline:
```
[00s - 02s] -> 🟢 SAFE (SND: 0.12 | EMO: 0.05 | LEX: 0.00)
[02s - 04s] -> 🔴 DANGER (SND: 0.85 | EMO: 0.92 | LEX: 1.00)
```

---

### 🧬 Key Engineering Decisions

*   **Multicollinearity Suppression:** The SVM inherently identified that `emotion_score` (Wav2Vec2) and `acoustic_score` (YAMNet) were highly collinear (both react to screaming). It actively suppressed the emotion weight to 0.036 to prevent double-counting, relying on the more robust acoustic score (0.818).
*   **Useless Feature Pruning:** Both models assigned an absolute weight of `0.000` to `sound_count`. The math proved that the *loudness* of a single sound event is all that matters; the *quantity* of sound events is mathematically irrelevant.
*   **Conditional Gating:** Faster-Whisper is skipped entirely if YAMNet does not detect human speech, saving ~800ms of compute latency per chunk on silent/noisy environments.

---

### 👥 Project Team

| Name | Index |
| :--- | :--- |
| Blessing Laryea | FCM.41.008.164.23 |
| Tedlee Appiah-Kubi | FCM.41.008.062.23 |
| Agyekum Gideon | FCM.41.008.032.23 |
| Osei Aaron Kwadwo | FCM.41.008.202.23 |
| Aduamah Seth | FCM.41.008.020.23 |
| Maxwell Expensive Honu | FCM.41.008.141.23 |
| Marvin Tamakloe | FCM.41.008.224.23 |

---
<div align="center">
  Made with blood, sweat, and `class_weight='balanced'` 🩸
</div>
```

