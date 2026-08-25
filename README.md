# Chicken Disease Classification

[![Python](https://img.shields.io/badge/Python-3.8-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-orange.svg)](https://www.tensorflow.org/)
[![Flask](https://img.shields.io/badge/Flask-Web%20App-black.svg)](https://flask.palletsprojects.com/)
[![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED.svg)](https://www.docker.com/)
[![Azure](https://img.shields.io/badge/Microsoft%20Azure-Deployed-0078D4.svg)](https://azure.microsoft.com/)
[![CI/CD](https://github.com/ahmadakil67/Chicken-Disease-Classification/actions/workflows/cicd.yaml/badge.svg)](https://github.com/ahmadakil67/Chicken-Disease-Classification/actions/workflows/cicd.yaml)

An end-to-end deep learning project for classifying chicken fecal images into three health categories using **VGG16 transfer learning**, **TensorFlow/Keras**, **Azure Blob Storage**, **Flask**, **Docker**, **Azure Container Registry**, **Azure App Service**, and **GitHub Actions CI/CD**.

The repository follows a modular, configuration-driven machine learning pipeline covering data ingestion, base-model preparation, training, evaluation, prediction, containerization, and cloud deployment.

> **Disclaimer:** This project is intended for educational and research purposes. It is not a substitute for veterinary diagnosis or professional animal-health advice.

---

## Live Application

The application is deployed on Microsoft Azure App Service:

**Live URL:**  
https://chicken-disease-2026-aqbug7egfnb6dydd.indiasouthcentral-01.azurewebsites.net

---

## Overview

Automated image classification can support research into faster poultry-health screening. This project trains a convolutional neural network to classify chicken fecal images into three categories:

| Class Index | Class |
|---:|---|
| 0 | Coccidiosis |
| 1 | Healthy |
| 2 | Salmonella |

The model uses a pretrained **VGG16** backbone with ImageNet weights. The convolutional feature extractor is frozen and a custom three-class softmax classifier is trained on the project dataset.

The class order is derived alphabetically by Keras `flow_from_directory()`.

---

## Key Features

- Three-class chicken disease classification
- Transfer learning with pretrained **VGG16**
- TensorFlow/Keras training and inference pipeline
- Azure Blob Storage based dataset ingestion
- Automatic ZIP download and extraction
- Image augmentation during training
- Configuration-driven ML workflow
- Model evaluation with loss and accuracy reporting
- JSON-based evaluation output
- Flask web application for image upload and prediction
- Dockerized application
- Azure Container Registry image hosting
- Azure App Service container deployment
- GitHub Actions based CI/CD
- Passwordless Azure authentication through GitHub OIDC
- Commit-specific Docker image tagging for traceable deployments
- Modular project structure suitable for MLOps-style workflows

---

## System Architecture

```text
                         TRAINING PIPELINE
                         -----------------

                    Azure Blob Storage
                            |
                            v
                      Data Ingestion
                            |
                            v
                 Download + Extract ZIP
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


                         DEPLOYMENT PIPELINE
                         -------------------

Developer
   |
   | git push origin main
   v
GitHub Repository
   |
   v
GitHub Actions
   |
   +--> Continuous Integration
   |
   +--> Docker Image Build
   |
   +--> Push Image to Azure Container Registry
   |
   +--> Update Azure App Service Site Container
   |        target port: 8080
   |
   v
Azure App Service
   |
   v
Live Web Application
```

---

## Technology Stack

| Area | Technology |
|---|---|
| Language | Python 3.8 |
| Deep Learning | TensorFlow / Keras |
| Backbone | VGG16 |
| Image Processing | TensorFlow / NumPy |
| Web Framework | Flask |
| API Support | Flask-CORS |
| Data Storage | Azure Blob Storage |
| Configuration | YAML / PyYAML |
| Environment Variables | python-dotenv |
| Containerization | Docker |
| Container Registry | Azure Container Registry |
| Cloud Hosting | Azure App Service for Linux |
| CI/CD | GitHub Actions |
| Cloud Authentication | GitHub OIDC + Microsoft Entra ID |
| Version Control | Git / GitHub |
| Environment Management | Conda |

---

## Project Structure

```text
Chicken-Disease-Classification/
|
|-- .github/
|   `-- workflows/
|       `-- cicd.yaml
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
|-- Dockerfile
|-- .gitignore
|-- app.py
|-- main.py
|-- params.yaml
|-- requirements.txt
|-- scores.json
`-- README.md
```

> Some generated artifacts only appear after the corresponding pipeline stage has run. Local `.env` files must remain untracked.

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

Keras uses these folder names to infer target classes.

---

# Machine Learning Pipeline

## 1. Data Ingestion

The data ingestion stage:

1. Reads the project configuration.
2. Connects to the configured Azure Blob Storage container.
3. Downloads `data.zip`.
4. Saves it under `artifacts/data_ingestion/`.
5. Extracts the dataset for downstream stages.

Azure Blob Storage may transfer the file in byte ranges. HTTP `206 Partial Content` responses can therefore appear during chunked downloads and are not necessarily errors.

---

## 2. Prepare Base Model

The model is based on:

```text
VGG16
+ ImageNet pretrained weights
+ include_top=False
+ input shape = 224 x 224 x 3
```

The pretrained convolutional layers are frozen and a three-class softmax output layer is added:

```python
Dense(3, activation="softmax")
```

---

## 3. Training

Images are resized to:

```text
224 x 224
```

The training pipeline supports augmentation such as:

- Rotation
- Horizontal flipping
- Width shifting
- Height shifting
- Shearing
- Zooming

VGG16 preprocessing is performed with:

```python
tf.keras.applications.vgg16.preprocess_input
```

The current workflow uses a **20% validation split**.

---

## 4. Evaluation

The evaluation stage loads:

```text
artifacts/training/model.h5
```

and evaluates the model using the same VGG16 preprocessing configuration.

Results are stored in:

```text
scores.json
```

Latest recorded result from the project:

```json
{
  "loss": 1.4880330562591553,
  "accuracy": 0.9486780762672424
}
```

This corresponds to approximately **94.87% validation accuracy**.

> The recorded score is based on a validation split from the same dataset, not a fully independent held-out test set. A separate test set should be used for final model assessment.

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

Accuracy should not be treated as the only measure of model quality. Future evaluation should also include class-wise precision, recall, F1-score, a confusion matrix, and testing on unseen real-world images.

---

# Configuration

## `params.yaml`

Example:

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

For systems with limited memory, reducing `BATCH_SIZE` to `4` or `8` may help.

---

## `config/config.yaml`

The main configuration defines paths and data-ingestion settings.

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

# Azure Blob Storage Setup

The application reads the Azure Storage connection string from an environment variable.

Create a local `.env` file in the project root:

```env
AZURE_STORAGE_CONNECTION_STRING="YOUR_AZURE_STORAGE_CONNECTION_STRING"
```

Never commit `.env` or cloud credentials to GitHub.

Make sure `.gitignore` includes:

```gitignore
.env
```

The current data-ingestion configuration uses:

```text
Container: chicken-data-2025
Blob:      data.zip
```

When reusing the project, replace the Azure configuration with your own resources.

---

# Installation

## 1. Clone the Repository

```bash
git clone https://github.com/ahmadakil67/Chicken-Disease-Classification.git
cd Chicken-Disease-Classification
```

## 2. Create a Conda Environment

```bash
conda create -n cnncls python=3.8 -y
conda activate cnncls
```

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Running the ML Pipeline

Activate the environment:

```bash
conda activate cnncls
```

Run:

```bash
python main.py
```

Pipeline sequence:

```text
Data Ingestion
      |
      v
Prepare Base Model
      |
      v
Training
      |
      v
Evaluation
```

Successful execution should produce:

```text
artifacts/training/model.h5
scores.json
```

---

# Running the Web Application Locally

After the trained model exists:

```bash
python app.py
```

The application listens on port `8080`.

Open:

```text
http://127.0.0.1:8080
```

Then upload an image to obtain a prediction.

---

# Prediction Pipeline

The prediction pipeline:

1. Loads the trained model.
2. Loads the uploaded image.
3. Resizes it to `224 x 224`.
4. Applies VGG16 preprocessing.
5. Performs inference.
6. Selects the class with the highest probability.
7. Returns the predicted class.

Expected mapping:

```python
class_names = {
    0: "Coccidiosis",
    1: "Healthy",
    2: "Salmonella"
}
```

An example response can look like:

```json
[
  {
    "prediction": "Healthy",
    "confidence": 0.96
  }
]
```

The exact response format depends on the implementation in `predict.py`.

---

# Flask Endpoints

## Home

```http
GET /
```

Renders the web interface.

## Prediction

```http
POST /predict
```

Receives an encoded image, runs inference, and returns a prediction.

## Training

```http
POST /train
```

Runs the training pipeline through `main.py`.

> Triggering training from a synchronous web route is suitable for demonstration, but a controlled training job or workflow is preferable for production systems.

---

# Docker

The application is containerized and exposes port `8080`.

Build the image locally:

```bash
docker build -t chicken-disease-app .
```

Run it:

```bash
docker run -p 8080:8080 chicken-disease-app
```

Then open:

```text
http://localhost:8080
```

For Azure deployment, the image is published to Azure Container Registry using the repository:

```text
chickenmlacr2026.azurecr.io/chicken-disease-app
```

---

# Azure Deployment

The production application runs as a Linux container in **Azure App Service**.

Current deployment flow:

```text
Docker Image
    |
    v
Azure Container Registry
    |
    v
Azure App Service
    |
    v
Site Container: main
Target Port: 8080
```

The application listens internally on port `8080`, so the Azure App Service site container must also use:

```text
TargetPort = 8080
```

This is important because a mismatch between Azure's target port and the application's listening port can produce `503 Service Unavailable` or startup-probe timeouts.

---

# CI/CD with GitHub Actions

The repository contains:

```text
.github/workflows/cicd.yaml
```

The workflow runs automatically when code is pushed to the `main` branch, except when only `README.md` changes are ignored by the configured workflow.

The pipeline contains three jobs:

```text
Continuous Integration
        |
        v
Build and Push Docker Image
        |
        v
Continuous Deployment
```

## CI/CD Flow

1. Checkout the repository.
2. Run the integration stage.
3. Authenticate to Azure using GitHub OIDC.
4. Authenticate to Azure Container Registry.
5. Build the Docker image.
6. Tag the image with both:
   - the Git commit SHA
   - `latest`
7. Push the image to Azure Container Registry.
8. Update the Azure App Service `main` site container.
9. Keep the Azure target port at `8080`.
10. Restart the Azure Web App.
11. Verify the deployed site-container configuration.

Using the Git commit SHA as the deployment tag makes each deployment traceable to a specific source-code revision.

---

## GitHub Actions Authentication

The workflow uses passwordless **OpenID Connect (OIDC)** authentication instead of storing a long-lived Azure client secret.

Required GitHub repository secrets:

```text
AZURE_CLIENT_ID
AZURE_TENANT_ID
AZURE_SUBSCRIPTION_ID
ACR_NAME
ACR_LOGIN_SERVER
AZURE_WEBAPP_NAME
```

Do **not** commit any of these values directly into application source code.

The Azure identity used by GitHub Actions requires suitable permissions for:

- pushing images to Azure Container Registry
- updating and restarting the Azure Web App

---

## Automatic Deployment

After modifying the application:

```bash
git add .
git commit -m "Describe your change"
git push origin main
```

GitHub Actions then performs the Docker build, registry push, and Azure deployment automatically.

You can monitor each run from:

```text
GitHub Repository -> Actions -> Chicken Disease CI/CD
```

A successful run should show:

```text
Continuous Integration        ✅
Build and Push Docker Image   ✅
Continuous Deployment        ✅
```

---

# Security Notes

- Never commit `.env`.
- Never commit Azure connection strings.
- Never commit client secrets, passwords, access tokens, or registry credentials.
- Use GitHub repository secrets for workflow configuration.
- Prefer GitHub OIDC for Azure authentication.
- Rotate any credential immediately if it is accidentally exposed.
- Keep training data and model artifacts out of Git when they are too large or sensitive.
- Review Azure RBAC permissions and grant only the access required by the workflow.

---

# Reproducibility

The image generators use a fixed seed:

```python
seed = 42
```

This helps make train/validation splitting more reproducible across runs.

---

# Troubleshooting

## Azure App returns 503 or startup timeout

Verify the site-container target port:

```bash
az webapp sitecontainers list \
  --name chicken-disease-2026 \
  --resource-group chicken-ml-rg \
  -o table
```

The main container should use:

```text
Name        = main
TargetPort  = 8080
```

If necessary:

```bash
az webapp sitecontainers update \
  --name chicken-disease-2026 \
  --resource-group chicken-ml-rg \
  --container-name main \
  --target-port 8080
```

---

## Check Azure container logs

```bash
az webapp log tail \
  --name chicken-disease-2026 \
  --resource-group chicken-ml-rg
```

Look for messages such as:

```text
Container is running
Site startup probe succeeded
Running on ...:8080
```

---

## TensorFlow shows no GPU

Check:

```bash
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

An empty list means TensorFlow is using the CPU.

---

## Dataset directory not found

Verify:

```text
artifacts/data_ingestion/Chicken Disease/
```

contains:

```text
Coccidiosis/
Healthy/
Salmonella/
```

---

## Model not found

Verify:

```text
artifacts/training/model.h5
```

exists before prediction.

---

## Azure Blob logs show HTTP 206

`206 Partial Content` can be expected when Azure Blob Storage transfers a file using byte-range requests.

---

# Recommended Improvements

- Create an independent train/validation/test split
- Add precision, recall, F1-score, and confusion matrix reporting
- Analyze class imbalance
- Evaluate on unseen real-world poultry images
- Add prediction confidence calibration
- Load the model once at application startup when appropriate
- Add model versioning
- Add experiment tracking
- Replace placeholder integration checks with real linting and unit tests such as `flake8` and `pytest`
- Add deployment health checks
- Add automated rollback or deployment-slot strategy
- Move long-running training out of synchronous Flask requests
- Compare lighter backbones such as MobileNetV2 or EfficientNet for faster inference
- Add monitoring and application telemetry

---

# Notes on GPU Usage

The project can run without a GPU, although training can be significantly faster with compatible GPU acceleration.

TensorFlow GPU support depends on the operating system, TensorFlow version, NVIDIA driver, CUDA, cuDNN, and hardware compatibility.

Cloud GPU environments such as Azure Machine Learning, Kaggle, or Google Colab may be useful for larger experiments.

---

# License

No license is currently specified.

If the repository is intended for public reuse, add an appropriate open-source license such as MIT or Apache-2.0.

---

# Author

**Project:** Chicken Disease Classification  
**GitHub:** [@ahmadakil67](https://github.com/ahmadakil67)  
**Focus:** Deep Learning, Transfer Learning, Flask, Docker, Azure, MLOps, CI/CD

---

## Acknowledgements

This project uses:

- TensorFlow/Keras for deep learning
- VGG16 pretrained on ImageNet
- Flask for the web interface
- Microsoft Azure for storage, container registry, and hosting
- GitHub Actions for CI/CD automation

---

If you find the project useful, consider giving the repository a star.
