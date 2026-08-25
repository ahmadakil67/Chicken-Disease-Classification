# Chicken Disease Classification

An end-to-end deep learning project for classifying chicken fecal images into three health categories using **VGG16 transfer learning**, **TensorFlow/Keras**, **Azure Blob Storage**, and a **Flask web application**.

The project is organized as a reusable machine learning pipeline covering data ingestion, base-model preparation, model training, evaluation, and image prediction.

---

## Overview

Poultry health monitoring can benefit from automated image-based screening tools. This project trains a convolutional neural network to classify chicken fecal images into:

- **Coccidiosis**
- **Healthy**
- **Salmonella**

The model uses a pretrained **VGG16** backbone with ImageNet weights. The convolutional base is frozen and a custom three-class classification layer is trained on the project dataset.

> **Important:** This project is intended for educational/research use and is not a substitute for veterinary diagnosis.

---

## Key Features

- Three-class chicken disease classification
- Transfer learning with pretrained **VGG16**
- Azure Blob Storage based dataset ingestion
- Automatic ZIP download and extraction
- Image augmentation during training
- Reusable configuration-driven training pipeline
- Model evaluation with loss and accuracy reporting
- JSON-based evaluation output
- Flask web application for image prediction
- Modular project structure suitable for MLOps-style workflows

---

## Model Classes

The project currently uses the following class mapping:

| Class Index | Class |
|---:|---|
| 0 | Coccidiosis |
| 1 | Healthy |
| 2 | Salmonella |

The class order is produced alphabetically by Keras `flow_from_directory()`.

---

## Project Architecture

```text
Azure Blob Storage
        |
        v
   Data Ingestion
        |
        v
Download data.zip
        |
        v
 Extract Dataset
        |
        v
Prepare VGG16 Base Model
        |
        v
   Model Training
        |
        v
 Model Evaluation
        |
        v
 artifacts/training/model.h5
        |
        v
 Prediction Pipeline
        |
        v
    Flask Web App
```

---

## Project Structure

```text
Chicken-Disease-Classification/
|
|-- .github/
|
|-- artifacts/
|   |-- data_ingestion/
|   |   |-- data.zip
|   |   `-- Chicken Disease/
|   |       |-- Coccidiosis/
|   |       |-- Healthy/
|   |       `-- Salmonella/
|   |
|   |-- prepare_base_model/
|   |   |-- base_model.h5
|   |   `-- base_model_updated.h5
|   |
|   `-- training/
|       `-- model.h5
|
|-- config/
|   `-- config.yaml
|
|-- logs/
|   `-- running_logs.log
|
|-- research/
|   |-- 01_data_ingestion.ipynb
|   |-- 02_prepare_base_model.ipynb
|   |-- 03_model_trainer.ipynb
|   `-- 04_model_evaluation.ipynb
|
|-- src/
|   `-- cnnClassifier/
|       |-- components/
|       |   |-- data_ingestion.py
|       |   |-- prepare_base_model.py
|       |   |-- model_trainer.py
|       |   `-- evaluation.py
|       |
|       |-- config/
|       |   `-- configuration.py
|       |
|       |-- entity/
|       |   `-- config_entity.py
|       |
|       |-- pipeline/
|       |   |-- stage_01_data_ingestion.py
|       |   |-- stage_02_prepare_base_model.py
|       |   |-- stage_03_model_trainer.py
|       |   |-- stage_04_model_evaluation.py
|       |   `-- predict.py
|       |
|       |-- utils/
|       |   `-- common.py
|       |
|       |-- constants/
|       `-- __init__.py
|
|-- templates/
|   `-- index.html
|
|-- .env
|-- .gitignore
|-- app.py
|-- main.py
|-- params.yaml
|-- scores.json
`-- README.md
```

> Some generated files and folders may only appear after the corresponding pipeline stage has run.

---

## Dataset Structure

After data ingestion, the extracted dataset is expected to follow this structure:

```text
artifacts/data_ingestion/Chicken Disease/
|
|-- Coccidiosis/
|-- Healthy/
`-- Salmonella/
```

Keras uses these directory names to infer the target classes.

---

## Technology Stack

- **Python 3.8**
- **TensorFlow / Keras**
- **VGG16**
- **NumPy**
- **Flask**
- **Flask-CORS**
- **Azure Blob Storage**
- **PyYAML**
- **python-dotenv**
- **Conda**
- **Git / GitHub**

---

## Machine Learning Pipeline

### 1. Data Ingestion

The data ingestion stage:

1. Reads Azure Blob Storage configuration.
2. Connects to the configured Azure Storage container.
3. Downloads `data.zip`.
4. Stores it under `artifacts/data_ingestion/`.
5. Extracts the dataset for subsequent stages.

Azure may download the blob in multiple byte ranges. HTTP `206 Partial Content` responses in the logs are normal during chunked downloads.

### 2. Prepare Base Model

The project uses:

```text
VGG16
+ ImageNet pretrained weights
+ include_top=False
+ input size 224 x 224 x 3
```

The pretrained convolutional layers are frozen and a new softmax classification layer is added for the three target classes.

Current model output:

```text
Dense(3, activation="softmax")
```

### 3. Training

Images are resized to:

```text
224 x 224
```

The training pipeline supports image augmentation such as:

- Rotation
- Horizontal flipping
- Width shifting
- Height shifting
- Shearing
- Zooming

The corrected VGG16 training pipeline uses:

```python
tf.keras.applications.vgg16.preprocess_input
```

for image preprocessing.

A validation split of **20%** is used.

### 4. Evaluation

The evaluation stage loads:

```text
artifacts/training/model.h5
```

and evaluates it using the same VGG16 preprocessing and validation split configuration used during training.

Evaluation results are saved to:

```text
scores.json
```

Latest recorded result:

```json
{
  "loss": 1.4880330562591553,
  "accuracy": 0.9486780762672424
}
```

This corresponds to approximately **94.87% validation accuracy**.

> This score is currently based on a validation split from the same dataset rather than a fully independent held-out test set. A separate test set is recommended for final model assessment.

---

## Configuration

### `params.yaml`

Example configuration:

```yaml
AUGMENTATION: True
IMAGE_SIZE: [224, 224, 3]
BATCH_SIZE: 16
INCLUDE_TOP: False
EPOCHS: 1
CLASSES: 3
WEIGHTS: imagenet
LEARNING_RATE: 0.001
```

For machines with limited RAM/VRAM, reducing `BATCH_SIZE` to `4` or `8` may help.

### `config/config.yaml`

The main configuration defines paths for:

- Data ingestion
- Azure Blob data
- Base model
- Updated base model
- Trained model

Example:

```yaml
artifacts_root: artifacts

data_ingestion:
  root_dir: artifacts/data_ingestion
  container_name: chicken-data-2025
  blob_name: data.zip
  local_data_file: artifacts/data_ingestion/data.zip
  unzip_dir: artifacts/data_ingestion

prepare_base_model:
  root_dir: artifacts/prepare_base_model
  base_model_path: artifacts/prepare_base_model/base_model.h5
  updated_base_model_path: artifacts/prepare_base_model/base_model_updated.h5

training:
  root_dir: artifacts/training
  trained_model_path: artifacts/training/model.h5
```

---

## Azure Blob Storage Setup

The project reads the Azure Storage connection string from an environment variable.

Create a `.env` file in the project root:

```env
AZURE_STORAGE_CONNECTION_STRING="YOUR_AZURE_STORAGE_CONNECTION_STRING"
```

### Security

Never commit the `.env` file or Azure credentials to GitHub.

Make sure `.gitignore` contains:

```gitignore
.env
```

The current Azure resources used by the project are configured around:

```text
Storage account: chickenmlstorage2026
Container:       chicken-data-2025
Blob:            data.zip
```

If you fork or reuse the project, replace these values with your own Azure resources.

---

## Installation

### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd Chicken-Disease-Classification
```

### 2. Create a Conda Environment

```bash
conda create -n cnncls python=3.8 -y
conda activate cnncls
```

### 3. Install Dependencies

Install the project requirements according to your dependency file, or install the core packages:

```bash
pip install tensorflow flask flask-cors azure-storage-blob python-dotenv pyyaml
```

If the repository contains `requirements.txt`, prefer:

```bash
pip install -r requirements.txt
```

---

## Running the Full Pipeline

Activate the environment:

```bash
conda activate cnncls
```

Then run:

```bash
python main.py
```

The pipeline runs the stages in sequence:

```text
Data Ingestion
    ->
Prepare Base Model
    ->
Training
    ->
Evaluation
```

A successful run should produce the trained model at:

```text
artifacts/training/model.h5
```

and evaluation metrics at:

```text
scores.json
```

---

## Running the Web Application

After the trained model exists, start Flask:

```bash
python app.py
```

The application runs locally on:

```text
http://127.0.0.1:8080
```

Open the address in a browser and upload an image for classification.

---

## Prediction Pipeline

The prediction pipeline:

1. Loads the trained model.
2. Loads and resizes the uploaded image to `224 x 224`.
3. Applies VGG16 preprocessing.
4. Runs model inference.
5. Selects the class with the highest probability.
6. Returns the predicted disease class.

Expected mapping:

```python
class_names = {
    0: "Coccidiosis",
    1: "Healthy",
    2: "Salmonella"
}
```

Example response:

```json
[
  {
    "prediction": "Healthy",
    "confidence": 0.96
  }
]
```

The exact response structure depends on the implementation in `predict.py`.

---

## Flask Endpoints

### Home

```http
GET /
```

Renders the web interface.

### Prediction

```http
POST /predict
```

Receives an encoded image, runs inference, and returns the predicted class.

### Training

```http
POST /train
```

Runs the training pipeline through `main.py`.

> Triggering model training from a web route is useful for demonstration but should be replaced by a controlled training workflow or job system in a production application.

---

## Current Performance

| Metric | Value |
|---|---:|
| Validation Accuracy | **94.87%** |
| Validation Loss | **1.4880** |
| Number of Classes | **3** |
| Input Shape | **224 x 224 x 3** |
| Backbone | **VGG16** |
| Pretrained Weights | **ImageNet** |

Accuracy alone should not be treated as the complete measure of model quality. Future evaluation should also include per-class precision, recall, F1-score, and a confusion matrix.

---

## Recommended Next Improvements

- Create an independent train/validation/test split
- Add confusion matrix visualization
- Report precision, recall, and F1-score for each disease class
- Check class imbalance
- Add confidence scores to web predictions
- Load the prediction model once at application startup instead of reloading it for every request
- Add model versioning
- Add experiment tracking
- Add automated tests
- Add Docker support
- Add CI/CD
- Deploy the Flask application to Azure
- Move long-running training away from a synchronous Flask request
- Consider a lighter architecture such as MobileNetV2/EfficientNet for faster inference
- Evaluate the model on unseen real-world poultry images

---

## Notes on GPU Usage

TensorFlow GPU availability depends on the operating system, TensorFlow version, NVIDIA driver, CUDA, cuDNN, and the GPU itself.

The project does not require a GPU to run, but training can be significantly faster on a compatible modern GPU. Cloud GPU environments such as Azure Machine Learning, Kaggle, or Google Colab can be useful for larger training runs.

---

## Reproducibility

The project uses a fixed random seed in the image generators:

```python
seed=42
```

This helps keep training/validation splitting more reproducible across runs.

---

## Troubleshooting

### TensorFlow shows no GPU

Check:

```bash
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

An empty list means TensorFlow is currently using the CPU.

### Dataset directory not found

Verify that this directory exists:

```text
artifacts/data_ingestion/Chicken Disease
```

and contains:

```text
Coccidiosis/
Healthy/
Salmonella/
```

### Model not found

Verify:

```text
artifacts/training/model.h5
```

exists before starting prediction.

### Azure logs show HTTP 206

This is normally expected. Azure Blob Storage can download large files in chunks, and `206 Partial Content` indicates that a requested byte range was successfully returned.

---

## Disclaimer

This classifier is a machine learning research/educational project. Its predictions should not be used as a definitive veterinary diagnosis or as the sole basis for animal treatment decisions.

---

## License

No license has been specified yet.

If this project will be published publicly, add an appropriate open-source license such as MIT, Apache-2.0, or another license that matches the intended use.

---

## Author

**Project:** Chicken Disease Classification  
**Focus:** Deep Learning, Transfer Learning, MLOps-style Pipeline, Azure Blob Storage, Flask

Add your preferred name, GitHub profile, LinkedIn profile, and contact information here before publishing.
