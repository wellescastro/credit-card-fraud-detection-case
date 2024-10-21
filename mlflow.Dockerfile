# mlflow.Dockerfile

# Use the official MLflow image as the base image
FROM ghcr.io/mlflow/mlflow:latest

# Install curl to enable healthchecks
RUN apt-get update && apt-get install -y curl

# Clean up to reduce image size
RUN apt-get clean

