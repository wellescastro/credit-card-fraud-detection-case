from typing import Any, Dict, Optional

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ClassifierConfig(BaseModel):
    model_type: str = Field(
        ...,
        description="The scikit-learn model type as a qualified namespace, e.g., 'linear_model.SGDClassifier'.",
    )
    params: Dict[str, Any] = Field(
        ..., description="Parameters for the classifier model."
    )


class JobConfiguration(BaseModel):
    data_path: str = Field(..., description="Path to the training data file.")
    classifier: ClassifierConfig = Field(
        ..., description="Configuration for the classifier."
    )


class EnvironmentSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env")

    mlflow_tracking_uri: Optional[str] = Field(default=None)
    mlflow_registry_uri: Optional[str] = Field(default=None)
