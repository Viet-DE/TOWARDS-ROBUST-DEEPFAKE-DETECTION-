# TriXNet: Towards Robust Deepfake Detection

**Official implementation for the thesis/project:** > *"TOWARDS ROBUST DEEPFAKE DETECTION: A MULTI-MODAL APPROACH USING FREQUENCY, MOTION, AND LOCAL PART CONSISTENCY"*

---

## Overview

TriXNet is a novel three-stream neural network architecture designed to robustly detect deepfake videos by exploiting cross-modal inconsistencies. Instead of relying solely on spatial artifacts, TriXNet simultaneously analyzes three distinct modalities:

1. **Signal Integrity (Frequency):** Detects low-level GAN/Diffusion artifacts hidden in the frequency domain.
2. **Physical Motion (Optical Flow):** Captures temporal discontinuities and unnatural movements, especially around blending boundaries.
3. **Biological Semantics (Local Parts):** Ensures semantic consistency between local facial regions (e.g., matching the emotional state of the eyes and mouth).

These three streams are aggregated using a **Cross-Modality Attention (CMA)** mechanism (Transformer-based), allowing the network to cross-verify anomalies across different domains before making a final classification.

## Architecture

* **FRS (Frequency Residual Stream):** Extracts amplitude spectrum differences between consecutive frames using FFT. Features are encoded via an EfficientNet-B0 backbone adapted for 1-channel input.
* **DOF (Dense Optical Flow Stream):** Computes dense optical flow (Farnebäck) to capture motion vectors (2-channel input).
* **LPC (Local Part Consistency Stream):** A Siamese network that shares weights to extract features from cropped Eye and Mouth patches, concatenated to evaluate semantic synchronization.
* **CMA Fusion:** A Multi-head Transformer Encoder that models the interactions between the F_signal, F_motion, and F_bio tokens.

## Project Structure

```text
TriXNet/
├── configs/                # Configuration files (YAML) for models and training
├── data/                   # Processed datasets (frames, flow, frequency, parts)(It's too large, so cannot be pushed up)
├── datasets/               # PyTorch Dataset and DataLoader definitions
├── models/                 # Core network architecture
│   ├── cma/                # Cross-Modality Attention Fusion
│   ├── dof/                # Dense Optical Flow Stream
│   ├── frs/                # Frequency Residual Stream
│   ├── lpc/                # Local Part Consistency Stream
│   └── trixnet.py          # The complete TriXNet model
├── preprocessing/          # Offline feature extraction scripts
├── scripts/                # Utility scripts (e.g., dataset preparation)
├── splits/                 # JSON files defining Train/Val/Test splits
├── utils/                  # Helper functions (metrics, losses, logging)
├── requirements.txt        # Python dependencies
└── README.md               # This file
```
## Installation & Setup

1. **Clone the repository:**
   ```
   git clone https://github.com/Viet-DE/TOWARDS-ROBUST-DEEPFAKE-DETECTION-.git
   ```

2. **Create a virtual environment (Recommended):**
   ```
   python -m venv venv
   # On Windows:
   .\venv\Scripts\activate
   # On Linux/Mac:
   source venv/bin/activate  
   ```

3. **Install dependencies:**
   ```
   pip install -r requirements.txt
   ```

## Dataset Preparation

This project uses the **FaceForensics++ (FF++)** dataset.

1. Download the extracted frames of FF++ to your local drive.
2. Update the `SOURCE_PATH` in `scripts/prepare_ffpp.py` to point to your downloaded dataset.
3. Run the preparation script. This script uses **MediaPipe** to detect, crop faces (256x256), and organize them into the `data/frames/` directory, while automatically generating Train/Val/Test split configurations in `splits/`.

   ```
   python scripts/prepare_ffpp.py
   ```

## Usage (Work in Progress)

**1. Offline Preprocessing (Extract Flow & Frequency):**
*(Scripts for extracting Farnebäck Flow and FFT residuals will be documented here once completed).*

**2. Training the Model:**
```bash
python train.py 
```

**3. Evaluation:**
```bash
python eval.py 
```

## Evaluation Metrics
The model is evaluated based on standard Deepfake detection metrics:
* Accuracy (ACC)
* Area Under the Receiver Operating Characteristic Curve (AUC-ROC)
* Equal Error Rate (EER)
(all from `results/` :D)

## Acknowledgments
This project was developed as part of a research initiative focusing on Multi-modal Deepfake Detection. Special thanks to the creators of FaceForensics++ and MediaPipe for providing robust tools for facial analysis.
