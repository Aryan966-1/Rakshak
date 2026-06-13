# 🛤️ Project Rakshak — Google Colab Training Tutorial

This tutorial provides step-by-step instructions on how to set up, train, and download the trained model bundle for the **Project Rakshak** predictive maintenance pipeline using Google Colab's free T4 GPU.

---

## 📋 Table of Contents
1. [Readiness Status](#-readiness-status)
2. [Workspace Directory Structure](#-workspace-directory-structure)
3. [Step 1: Uploading Files to Google Colab / Google Drive](#-step-1-uploading-files-to-google-colab--google-drive)
4. [Step 2: Configuring the T4 GPU Runtime in Colab](#-step-2-configuring-the-t4-gpu-runtime-in-colab)
5. [Step 3: Installing Dependencies & Mounting Google Drive](#-step-3-installing-dependencies--mounting-google-drive)
6. [Step 4: Running the Training Pipeline](#-step-4-running-the-training-pipeline)
7. [Step 5: Downloading the Trained Model Bundle to Local Computer](#-step-5-downloading-the-trained-model-bundle-to-local-computer)

---

## ⚡ Readiness Status

> [!NOTE]
> **Yes, the files in this directory are 100% ready for Google Colab training!**
> 
> * **`train_colab.ipynb`** is a pre-configured Jupyter notebook designed to run the entire pipeline seamlessly.
> * **`train_colab.py`** is an orchestration script that executes `section_0.py` through `section_7.py` in sequence under a shared namespace, ensuring variables like `CONFIG` and the generated datasets flow smoothly from component to component.
> * **`requirements-colab.txt`** contains the exact list of non-PyTorch dependencies required to run the pipeline.
> * **Data Loading**: The dataset (`RakshakDataset` in `section_1.py`) generates highly realistic synthetic Indian Railways telemetry dynamically in memory. Therefore, no large external datasets need to be extracted or processed to run the training pipeline, making training robust, fast, and free of disk issues.

---

## 📂 Workspace Directory Structure

Ensure the following files are present in the directory you upload:

```text
notebook_sections/
├── requirements-colab.txt   # Colab-specific package dependencies
├── train_colab.ipynb        # Main Google Colab Jupyter Notebook
├── train_colab.py           # Python orchestration script
├── section_0.py             # Environment Setup & Config
├── section_1.py             # Synthetic Data Generation
├── section_2.py             # Anomaly Detection Engine (ADE)
├── section_3.py             # Failure Prediction Model (HM-STT)
├── section_4.py             # Root-Cause Heterogeneous GNN (HGNN)
├── section_7.py             # MLflow Logging & Model Bundling
└── SHARED_CONTRACT.md       # Shared interface definitions
```

---

## 📤 Step 1: Uploading Files to Google Colab / Google Drive

To ensure your checkpoints, figures, and final model bundles are **persisted** (Colab's temporary instance storage is wiped when you disconnect), we highly recommend mounting **Google Drive**.

### Method A: Uploading via Google Drive (Recommended)
1. Go to [Google Drive](https://drive.google.com).
2. Create a folder named `rakshak_v1` in your drive.
3. Inside the `rakshak_v1` folder, upload the entire `notebook_sections` folder containing the code files listed above.
4. When you open the `train_colab.ipynb` notebook from Google Drive, Google Colab will automatically be able to save your checkpoints and final outputs directly to `/content/drive/MyDrive/rakshak_v1/`.

### Method B: Direct Upload via Colab File Explorer
1. Go to [Google Colab](https://colab.research.google.com).
2. Click **Upload** and select `train_colab.ipynb`.
3. Open the left sidebar by clicking on the folder icon (📁).
4. Drag and drop `train_colab.py`, `requirements-colab.txt`, and all `section_*.py` files directly into the file explorer area.

> [!WARNING]
> If you use **Method B**, any files saved to the local directory (like checkpoints or model weights) will be **deleted** when the Google Colab session expires or disconnects. Use **Method A** for automatic persistence.

---

## ⚙️ Step 2: Configuring the T4 GPU Runtime in Colab

Before running any code, configure Colab to use a GPU.

1. Open `train_colab.ipynb` in Colab.
2. In the top-right menu bar, click the dropdown next to **Connect** (or click **Runtime** in the top menu).
3. Select **Change runtime type**.
4. Under **Hardware accelerator**, select **T4 GPU** (available on the free tier).
5. Click **Save**.

Verify that your GPU is active by executing the first code cell:
```python
# Check GPU runtime
!nvidia-smi
```
*Expected Output:* You should see details of the NVIDIA T4 GPU with 15 GB of VRAM.

---

## 🛠️ Step 3: Installing Dependencies & Mounting Google Drive

Run the setup cells in `train_colab.ipynb`:

### 1. Mount Google Drive (if using Method A)
Run the following code block to link your drive:
```python
from google.colab import drive
drive.mount('/content/drive', force_remount=False)
```
Follow the browser prompts to log in and grant Google Drive access.

### 2. Navigate to the Directory
Uncomment and execute the change directory command to point to your uploaded files:
```python
%cd /content/drive/MyDrive/rakshak_v1/notebook_sections
```

### 3. Install Required Packages
Install the required packages using the preconfigured `requirements-colab.txt` file:
```python
%pip install -q -r requirements-colab.txt
```
> [!TIP]
> This command installs core packages like `mlflow` and `torch-geometric` (for the Heterogeneous GNN) in a silent mode (`-q`), keeping the notebook output clean.

---

## 🚀 Step 4: Running the Training Pipeline

The pipeline is set up to let you run a quick verification before starting full training.

### 🧪 Option 1: Run the Smoke Test (Quick Validation)
To verify that everything is integrated correctly and runs from end-to-end, execute the following command:
```bash
!python train_colab.py --smoke-test --skip-mlflow --skip-section-installs
```
* **Duration**: ~2–3 minutes.
* **Details**: Overrides hyper-parameters to use tiny model configurations, 1 epoch of training, and skips MLflow and model bundling. It is ideal for validating GPU memory allocation and package compatibilities.

### 🏋️ Option 2: Run Full Model Training
Once the smoke test succeeds, run the full pipeline:
```bash
!python train_colab.py --skip-section-installs
```
* **Duration**: ~20–30 minutes (depending on CPU/GPU cycles).
* **Details**: Runs all sections sequentially.
  1. **`section_0.py`**: Initializes the random seeds and configurations.
  2. **`section_1.py`**: Simulates the Delhi-Agra corridor station graph and generates train/val/test data.
  3. **`section_2.py`**: Trains the Anomaly Detection Engine (IForest, VAE, and Meta-Classifier).
  4. **`section_3.py`**: Trains the HM-STT failure prediction transformer ensemble.
  5. **`section_4.py`**: Trains the Root-Cause analysis HGNN on the station graph.
  6. **`section_7.py`**: Logs all training metrics and curves to MLflow and packages the models.

---

## 💾 Step 5: Downloading the Trained Model Bundle to Local Computer

Once full training completes, `section_7.py` creates a structured folder containing the trained weights and serialization pickles for all models.

### Where is the model bundle saved?
* **On Google Drive**: `/content/drive/MyDrive/rakshak_v1/models/rakshak_v1/`
* **On Local Colab disk (fallback)**: `./models/rakshak_v1/`

### 📦 Model Bundle Structure
The model bundle is structured as follows:
```text
rakshak_v1/
├── manifest.json            # Model metadata, version, and training config info
├── ade/
│   ├── vae.pt               # PyTorch state dict for Anomaly Detection VAE
│   ├── isolation_forest.pkl # Scikit-learn pickle for Isolation Forest
│   └── meta_classifier.pkl  # Scikit-learn pickle for Meta-classifier
├── fpm/
│   ├── hmstt_best_24h.pt    # PyTorch state dict for 24-hour failure prediction TCN-Transformer
│   ├── hmstt_best_48h.pt    # PyTorch state dict for 48-hour failure prediction TCN-Transformer
│   └── hmstt_best_72h.pt    # PyTorch state dict for 72-hour failure prediction TCN-Transformer
└── hgnn/
    └── hgnn_best.pt         # PyTorch-Geometric state dict for Heterogeneous GNN Root Cause Analyzer
```

### How to download the model bundle?

#### Method 1: Download from Google Drive Web Interface (Recommended)
1. Go to your [Google Drive](https://drive.google.com).
2. Navigate to `rakshak_v1` -> `models`.
3. Right-click on the `rakshak_v1` folder and click **Download**.
4. Google Drive will compress the folder into a `.zip` archive and download it to your local computer.

#### Method 2: Compress and Download directly from Google Colab
Run the following cells inside your Colab notebook to zip the bundle and trigger a direct browser download:

```python
# 1. Zip the model bundle directory
!zip -r /content/rakshak_trained_models.zip /content/drive/MyDrive/rakshak_v1/models/rakshak_v1

# 2. Trigger browser download
from google.colab import files
files.download('/content/rakshak_trained_models.zip')
```

---
*You are now ready to train the Project Rakshak AI models on Google Colab!*
