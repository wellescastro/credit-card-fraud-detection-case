import typing as T

import mlflow
from mlflow.models import infer_signature
from sklearn.pipeline import Pipeline

from fraud_detection_trainer.core.preprocessing import (
    DataPreparationTransformer,
    InvalidRowsCleaner,
)
from fraud_detection_trainer.core.schema import dataframe_to_schema
from fraud_detection_trainer.core.services.training_service import TrainingService
from fraud_detection_trainer.infra import dataset, services
from fraud_detection_trainer.infra.config import EnvironmentSettings, JobConfiguration
from fraud_detection_trainer.jobs.job import Job, Locals


class TrainingJob(Job):
    def __init__(
        self, job_config: JobConfiguration, environment_settings: EnvironmentSettings
    ) -> None:
        super().__init__(
            services.MlflowService(environment_settings=environment_settings)
        )
        self.csv_reader = dataset.CSVReader(job_config.data_path)
        self.job_config = job_config
        self.environment_settings = environment_settings
        self.run_config = services.MlflowService.RunConfig(
            name="Training",
            description="Train and register a single AI/ML model.",
            tags={"job": "training"},
        )

    @T.override
    def run(self) -> Locals:
        logger = self.logger_service.logger()
        logger.info("With logger: {}", logger)

        client = self.mlflow_service.client()
        logger.info("With client: {}", client.tracking_uri)

        logger.info("Reading the data...")
        df = self.csv_reader.read()

        self.preprocessor = Pipeline(
            [
                (
                    "data_preparation",
                    DataPreparationTransformer(duplicates_keep_strategy="first"),
                ),
                ("format", InvalidRowsCleaner("is_fraud", valid_values=[0, 1])),
            ]
        )

        logger.info("Preprocessing the data...")
        df = self.preprocessor.transform(df)
        features = df.drop(columns=["is_fraud"])
        targets = df[["is_fraud"]]

        # Start MLFlow run
        with self.mlflow_service.run_context(run_config=self.run_config) as run:
            logger.info("With run context: %s", run.info)

            logger.info("Log parameters...")
            for param, value in self.job_config.classifier.params.items():
                mlflow.log_param(
                    f"{self.job_config.classifier.model_type}_{param}", value
                )

            logger.info("Training model...")
            training_service = TrainingService(
                self.job_config.classifier, dataframe_to_schema(features)
            )
            training_result = training_service.train(features, targets)
            logger.info("Training finished...")

            # Get a sample of transformed data to save the final transformation schema
            sample_x, _ = training_result.X_test[:1], training_result.y_test[:1]
            transformer_pipeline = training_result.pipeline[:-1]
            sample_x_transformed = transformer_pipeline.transform(sample_x)
            mlflow.log_dict(
                sample_x_transformed.dtypes.to_dict(), "transformed_sample_types.json"
            )

            logger.info("Creating a MLFlow model signature...")
            signature = infer_signature(training_result.X_test, training_result.y_test)

            logger.info("Log model to MLFlow...")
            mlflow.sklearn.log_model(
                training_result.pipeline, "fraud-detection-model", signature=signature
            )
            model_uri = mlflow.get_artifact_uri("fraud-detection-model")

            logger.info("Evaluating the logged model using MLFlow")
            eval_data = training_result.X_test
            eval_data["label"] = training_result.y_test
            mlflow.evaluate(
                model_uri,
                eval_data,
                targets="label",
                model_type="classifier",
                evaluators=["default"],
            )

            model_registry_name = "fraud-detection-model-dev"
            model_version = mlflow.register_model(
                model_uri,
                model_registry_name,
            )

            self.mlflow_service.client().set_registered_model_alias(
                model_registry_name, "Challenger", model_version.version
            )

            logger.info("Training job completed")

        return locals()
