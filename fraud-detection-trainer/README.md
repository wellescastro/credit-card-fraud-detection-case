# Fraud Detection Trainer

This project is a Python-based fraud detection trainer that uses machine learning techniques to train and evaluate models for detecting fraudulent transactions.

## Project Overview

The Fraud Detection Trainer is designed to:

1. Load and preprocess transaction data
2. Engineer relevant features
3. Train a machine learning model for fraud detection
4. Evaluate the model's performance
5. Log the results and model artifacts using MLflow

## Requirements

- Python 3.12+
- uv package manager
- Dependencies listed in `pyproject.toml`

## Project Structure

- `src/fraud_detection_trainer/`: Main package
  - `core/`: Core functionality
    - `feature_engineer.py`: Feature engineering
    - `preprocessing.py`: Data preprocessing
    - `schema.py`: Data schema definitions
    - `services/`: Service implementations
  - `infra/`: Infrastructure code
    - `config.py`: Configuration classes
    - `dataset.py`: Dataset handling
    - `services.py`: Service definitions
  - `jobs/`: Job definitions
    - `training_job.py`: Main training job
  - `__main__.py`: Entry point for the CLI

## Development

This project uses:
- [FastAPI](https://fastapi.tiangolo.com/) for the web framework
- [scikit-learn](https://scikit-learn.org/) for machine learning pipelines
- [MLflow](https://mlflow.org/) for model management and experiment tracking
- [Pydantic Settings](https://pydantic-docs.helpmanual.io/) for reading the job configuration settings
- [Pandas](https://pandas.pydata.org/) for data manipulation
- [uv](https://github.com/astral-sh/uv) for package management and virtual environments

## License

This project is licensed under the [MIT License](LICENSE).