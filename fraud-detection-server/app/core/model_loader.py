from typing import Optional, Tuple

import mlflow
import mlflow.sklearn
from mlflow.entities.model_registry import ModelVersion
from sklearn.pipeline import Pipeline


class MlflowModelLoader:
    """
    A loader class for fetching machine learning models and their versions from MLflow's Model Registry.

    This class provides methods to load models based on their registered names and aliases
    within an MLflow tracking server. It leverages MLflow's client API to interact with
    the Model Registry and retrieve the desired model pipelines.
    """

    @staticmethod
    def load_model_pipeline(
        model_name: str = "fraud-detection-model-dev", model_alias: str = "Challenger"
    ) -> Tuple[Optional[Pipeline], Optional[ModelVersion]]:
        """
        Loads a machine learning pipeline and its corresponding model version from MLflow's Model Registry.

        This static method connects to the MLflow tracking server, retrieves the specified model
        version based on the provided model name and alias, and loads the model pipeline using
        MLflow's sklearn module.

        Args:
            model_name (str, optional): The registered name of the model in MLflow. Defaults to "fraud-detection-model-dev".
            model_alias (str, optional): The alias associated with the desired model version. Defaults to "Challenger".

        Returns:
            Tuple[Optional[Pipeline], Optional[ModelVersion]]:
                - The loaded scikit-learn Pipeline object if successful; otherwise, None.
                - The corresponding ModelVersion object from MLflow's Model Registry if successful; otherwise, None.
        """
        mlflow_client = mlflow.MlflowClient()
        try:
            model_version = mlflow_client.get_model_version_by_alias(
                name=model_name, alias=model_alias
            )
        except mlflow.exceptions.MlflowException as ex:
            print(f"Error retrieving model version: {ex}")
            return None, None

        model_uri = f"models:/{model_name}@{model_alias}"
        try:
            model: Pipeline = mlflow.sklearn.load_model(model_uri)
        except mlflow.exceptions.MlflowException as ex:
            print(f"Error loading model from URI '{model_uri}': {ex}")
            return None, model_version

        return model, model_version


if __name__ == "__main__":
    model, version = MlflowModelLoader.load_model_pipeline()
    if model and version:
        print(f"Successfully loaded model version: {version.version}")
    else:
        print("Failed to load the model pipeline.")
