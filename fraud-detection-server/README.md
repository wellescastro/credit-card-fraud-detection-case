# Fraud Detection Server

A FastAPI-based server for fraud detection using machine learning models.

## Overview

This project is a fraud detection server that uses machine learning models to predict whether a transaction is fraudulent. It leverages FastAPI for the web framework, MLflow for model management, and scikit-learn for the machine learning pipeline.

## Features

- FastAPI-based REST API
- Integration with MLflow for model versioning and loading
- Fraud prediction endpoint
- Model reloading capability
- Dockerized application for easy deployment

## Requirements

- Python 3.12+
- uv package manager
- Dependencies listed in `pyproject.toml`

## API Endpoints

- `GET /`: Root endpoint, provides basic API information
- `POST /predict`: Predicts whether a transaction is fraudulent
- `GET /reload`: Reloads the machine learning model from MLflow

For detailed Swagger-based documentation, visit `http://0.0.0.0:8000/docs` when the server is running.

## Development

This project uses:
- [FastAPI](https://fastapi.tiangolo.com/) for the web framework
- [MLflow](https://mlflow.org/) for model management
- [scikit-learn](https://scikit-learn.org/) for machine learning pipelines
- [Pydantic](https://pydantic-docs.helpmanual.io/) for data validation
- [Pandas](https://pandas.pydata.org/) for data manipulation

To set up the development environment:

1. Install the recommended VS Code extensions listed in `.vscode/extensions.json`
2. Use the settings in `.vscode/settings.json` for consistent formatting and linting
3. Run tests using pytest: `pytest tests`

## License

This project is licensed under the [MIT License](LICENSE).