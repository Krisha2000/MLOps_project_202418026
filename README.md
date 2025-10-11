# End-to-End MLOps: Prudential Insurance Risk Prediction

## Project Overview

This repository contains a complete, production-grade MLOps pipeline designed to streamline the life insurance underwriting process. The system's core function is to analyze an applicant's data and generate a predictive risk classification, transforming a traditionally manual assessment into an automated, data-driven workflow.

This project utilizes an insurance applicant dataset, contained in the file train.csv, to predict a risk classification level. The dataset is composed of a mixture of continuous and discrete variables that describe each applicant. The core prediction target is an ordinal variable representing eight distinct risk levels, from 1 to 8. This data is versioned using DVC to ensure reproducibility without adding large files to the Git repository. Before training, the raw data is processed and stored in a feature store to maintain consistency between the model training and serving environments.

This system demonstrates a full, end-to-end MLOps lifecycle, covering every stage from data versioning and feature management to automated model training, serving, and continuous integration/continuous deployment (CI/CD).

---

## Project Highlights & Results

This project was successfully executed both locally for development and in the cloud via automated CI/CD pipelines. The key outcomes are demonstrated below.

### Local Experiment Tracking

**MLflow Experiment Tracking**: All local training runs were successfully logged, capturing parameters, metrics, and model artifacts in a reprodu
cible manner.

<img width="3716" height="1582" alt="Screenshot From 2025-10-11 14-39-41" src="https://github.com/user-attachments/assets/5864cbb6-0396-4734-96f2-f00f62728a75" />




### Live API Prediction Service

**Live Model Serving**: The trained model was served via a FastAPI endpoint, providing real-time, explainable predictions for new applicants.
<img width="1812" height="1961" alt="Screenshot From 2025-10-08 21-05-31" src="https://github.com/user-attachments/assets/0702444a-420a-45cf-bde4-7e4751738b72" />



### Automated CI/CD Pipeline

**Full CI/CD Automation**: The GitHub Actions workflow successfully automated the entire build, test, train, and package pipeline in the cloud.

<img width="3740" height="1970" alt="Screenshot From 2025-10-11 13-52-21" src="https://github.com/user-attachments/assets/0b7e02a2-4fe6-4055-8918-44e8151c7f89" />


---

## Project Structure

The project is organized into a modular structure that separates concerns, making it scalable and easy to maintain.

```
prudential-mlops/
├── .github/
│   └── workflows/
│       └── main.yml              # CI/CD automation pipeline for GitHub Actions
├── artifacts/                     # Directory for storing output files like the SHAP explainer
├── data/
│   ├── train.csv                  # Raw training data (tracked by DVC, not Git)
│   └── train.csv.dvc              # DVC pointer file for versioning the data
├── feature_repo/
│   ├── data/
│   │   └── train.parquet          # Processed data, optimized for Feast
│   ├── data_sources.py            # Defines the raw data sources for Feast
│   ├── feature_definitions.py     # Core logic for defining features
│   └── feature_store.yaml         # Main configuration file for the Feast feature store
├── scripts/
│   └── load_features.py           # Script to process raw data and populate the feature store
├── src/
│   ├── main.py                    # FastAPI application for serving the model
│   └── train.py                   # Script for training the model and logging to MLflow
├── .gitignore                     # Specifies files for Git to ignore
├── Dockerfile                     # Recipe to package the FastAPI app into a container
├── README.md                      # This documentation file
└── requirements.txt               # List of all Python dependencies for the project
```

### Rationale for Structure

- **`.github/workflows`**: Isolates all automation and CI/CD logic. This is the "Ops" engine of the project.
- **`data/`**: Contains raw, immutable data. Using DVC to track this directory ensures that our data is versioned without bloating the Git repository.
- **`feature_repo/`**: The heart of our feature management. By defining all feature logic here, we create a single source of truth that is used by both the training and serving code, thus eliminating train-serve skew.
- **`scripts/`**: Holds operational scripts that are not part of the main application, such as the one-time job to populate the feature store.
- **`src/`**: Contains the core Python source code for our application, separating the training logic (`train.py`) from the serving logic (`main.py`).

---

## MLOps Architecture and Philosophy

This project is built on a modern MLOps stack where each component serves a specific, critical function to ensure a robust and automated system.

| Component | Purpose |
|-----------|---------|
| **Code Versioning (Git)** | The single source of truth for all code |
| **Data Versioning (DVC)** | `train.csv` is tracked by DVC, ensuring data reproducibility |
| **Feature Store (Feast)** | Centralizes feature engineering to guarantee consistency between training and serving |
| **Experiment Tracking (MLflow)** | Logs every training run, creating an auditable history of model development |
| **Explainable AI (SHAP)** | Provides transparency by explaining why a model makes a specific prediction |
| **Model Serving (FastAPI)** | Serves the best model from the MLflow registry as a high-performance REST API |
| **Containerization (Docker)** | Packages the API into a portable image for consistent deployment |
| **CI/CD Automation (GitHub Actions)** | Automates the entire MLOps lifecycle from testing to deployment |

---

## Model & Evaluation

### Model Selection

A **LightGBM (Light Gradient Boosting Machine)** classifier was chosen for this task. It is a highly effective tree-based gradient boosting framework for tabular data, known for its performance, speed, and efficiency. It is well-suited to handle the mix of continuous and discrete variables present in the applicant dataset.

### Evaluation Metric

The primary evaluation metric for this project is the **Quadratic Weighted Kappa**. This metric was chosen for a critical reason: the prediction target is an ordinal variable. The risk levels (1-8) have a natural order, where a prediction of 7 for an actual risk of 8 is a much better prediction than a 2.

Simple accuracy would treat both of those errors as equally wrong. The Quadratic Weighted Kappa, however, measures the agreement between the model's predictions (Rater B) and the human ratings (Rater A) while penalizing predictions that are further away from the true value more heavily. This makes it a much more realistic and business-relevant measure of performance for this type of risk-stratification problem. A score of 1 indicates perfect agreement, while 0 indicates performance no better than random chance.

### Metric Calculation

The Quadratic Weighted Kappa is calculated based on three matrices:

1. **Observed Matrix (O)**: An N × N confusion matrix where O<sub>i,j</sub> is the number of applications that received rating *i* by the human and rating *j* by the model.

2. **Expected Matrix (E)**: Calculates the number of agreements that would be expected purely by chance.

3. **Weight Matrix (w)**: Assigns a penalty to disagreements. For the quadratic kappa, this penalty increases quadratically with the distance between the ratings. The formula is:

```
w(i,j) = (i - j)² / (N - 1)²
```

These matrices are then combined into the final kappa score formula:

```
κ = 1 - (Σ(i,j) w(i,j) × O(i,j)) / (Σ(i,j) w(i,j) × E(i,j))
```

This formula effectively calculates the observed agreement, corrects it for chance agreement, and applies a penalty for the severity of the disagreement.

---

## Project Output

The two primary outputs of this MLOps project are:

### 1. Live Prediction API

When running locally, the FastAPI server exposes a `/predict_risk` endpoint. When given an applicant ID, it returns a JSON object with the predicted risk level and an explanation.

**Example API Response:**

```json
{
  "applicant_id": 2,
  "predicted_risk_level": 5,
  "explanation": {
    "message": "Top 3 factors influencing the prediction (positive values increase risk).",
    "factors": {
      "Employment Info 6": -4.37,
      "BMI": 1.54,
      "Employment Info 1": -3.2
    }
  }
}
```

### 2. Versioned Docker Image

The CI/CD pipeline's final output is a Docker image pushed to Docker Hub. This image contains the complete, ready-to-deploy prediction service, tagged with the Git commit hash for perfect traceability.

---

## Local Development and Execution

Follow these steps to set up and run the project on a local machine.

### Prerequisites

- **Git**: For version control
- **Python 3.11**: It is highly recommended to manage Python versions with `pyenv`
- **Docker**: To run the Redis online store

### 1. Clone the Repository

```bash
git clone https://github.com/Krisha2000/MLOps_project_202418026.git
cd MLOps_project_202418026
```

### 2. Set Up the Environment

```bash
# Set the local Python version (if using pyenv)
pyenv local 3.11.9

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install all dependencies
pip install -r requirements.txt
```

### 3. Download and Version the Data

```bash
# Place the train.csv from the data source into the data/ directory

# Track the data with DVC
dvc add data/train.csv
git add data/train.csv.dvc
git commit -m "feat: Track training data with DVC"
```

---

## Running the Project Components Locally

This project has three main components that are run in separate terminals.

### Terminal 1: Start Background Services

```bash
# Start the Redis container for the Feast online store
docker start prudential-redis || docker run --name prudential-redis -p 6379:6379 -d redis

# Start the MLflow server to track experiments
mlflow ui
```

The MLflow dashboard is now available at **http://127.0.0.1:5000**

### Terminal 2: Prepare Feature Store & Train the Model

```bash
# Activate the environment
source venv/bin/activate  # On Windows: venv\Scripts\activate

# (First time only) Create the Parquet file and populate the feature store
python scripts/load_features.py

cd feature_repo
feast apply
feast materialize-incremental $(date -u +%Y-%m-%dT%H:%M:%S)
cd ..

# Run the training pipeline
python src/train.py
```

### Terminal 3: Run the Prediction API

```bash
# Activate the environment
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Start the FastAPI server
uvicorn src.main:app --reload
```

The API is now available at **http://127.0.0.1:8000**

Access the interactive API documentation at **http://127.0.0.1:8000/docs**

---

## Docker Deployment

### Build the Docker Image

```bash
docker build -t prudential-mlops:latest .
```

### Run the Container

```bash
docker run -p 8000:8000 prudential-mlops:latest
```

---

## CI/CD Pipeline

The GitHub Actions workflow (`.github/workflows/main.yml`) automates:

1. **Code checkout** from the repository
2. **Dependency installation**
3. **Data versioning** with DVC
4. **Feature store setup** with Feast
5. **Model training** and logging to MLflow
6. **Docker image building** and pushing to Docker Hub
7. **Automated testing** and validation

To trigger the pipeline, simply push to the main branch or create a pull request.

---

## Monitoring and Tracking

- **MLflow UI**: View experiment runs, compare metrics, and manage model registry at `http://127.0.0.1:5000`
- **FastAPI Docs**: Interactive API documentation at `http://127.0.0.1:8000/docs`
- **Logs**: Check application logs for debugging and monitoring

---

## Contributing

Contributions are welcome. Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

