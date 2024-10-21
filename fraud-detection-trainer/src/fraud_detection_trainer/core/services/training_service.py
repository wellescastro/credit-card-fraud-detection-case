import importlib
from dataclasses import dataclass

import pandas as pd
from category_encoders.cat_boost import CatBoostEncoder
from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from fraud_detection_trainer.core.feature_engineer import FeatureEngineer
from fraud_detection_trainer.core.preprocessing import (
    ColumnDropper,
    NormalizationTransformer,
    normalize_text,
)
from fraud_detection_trainer.core.schema import Schema
from fraud_detection_trainer.infra.config import ClassifierConfig


@dataclass
class TrainingResult:
    pipeline: Pipeline
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series


class TrainingService:
    """Service to handle model training.

    Parameters:
        csv_reader: An instance of CSVReader to read the dataset.
    """

    def __init__(self, classifier_config: ClassifierConfig, schema: Schema):
        self.classifier_config = classifier_config

        # Features with small-to-medium cardinality
        categorical_features_one_hot = [
            "category",
            "state",
        ]
        # Features with high cardinality
        categorical_features_catboost = [
            "job",
            "city",
            "merchant"
        ]

        # Transformer that computes new features
        feature_engineer = FeatureEngineer()

        # List of columns that are excluded mid pipeline execution but are still required for feature engineering
        exclude_columns = {
            "trans_date_trans_time",
            "dob",
            "trans_num",
            "lat",
            "long",
            "merch_lat",
            "merch_long",
        }

        numerical_features = list(
            set(
                [
                    item["name"]
                    for item in schema.columns
                    if item["type"] in ["float64", "int64"]
                ]
                + feature_engineer.list_numeric_features()
            )
            - exclude_columns
        )

        encoding_transformer = ColumnTransformer(
            transformers=[
                (
                    "catboost",
                    CatBoostEncoder(return_df=True, cols=categorical_features_catboost),
                    categorical_features_catboost,
                ),
                (
                    "onehot",
                    OneHotEncoder(sparse_output=False),
                    categorical_features_one_hot,
                ),
                ("scaler", StandardScaler(), numerical_features),
            ],
            remainder="passthrough",
            verbose_feature_names_out=False,
        )

        try:
            module_path, class_name = self.classifier_config.model_type.rsplit(".", 1)
            model_class = getattr(importlib.import_module(module_path), class_name)
        except (ValueError, AttributeError, ImportError) as e:
            raise ValueError(
                f"Unsupported model type: {self.classifier_config.model_type}"
            ) from e

        classifier = model_class(**self.classifier_config.params)

        self.pipeline = Pipeline(
            [
                (
                    "normalizer",
                    NormalizationTransformer(
                        normalization_funcs={
                            "merchant": normalize_text,
                            "job": normalize_text,
                        }
                    ),
                ),
                ("feature_engineer", feature_engineer),
                ("column_dropper", ColumnDropper(exclude_columns)),
                ("encoding_transformer", encoding_transformer),
                ("classifier", classifier),
            ]
        )
        self.pipeline.set_output(transform="pandas")

    def train(self, features: pd.DataFrame, targets: pd.DataFrame) -> TrainingResult:
        X_train, X_test, y_train, y_test = train_test_split(
            features,
            targets,
            test_size=0.2,
            random_state=self.classifier_config.params["random_state"],
        )

        self.pipeline.fit(X_train, y_train)

        y_pred = self.pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1_score_outcome = f1_score(y_test, y_pred)
        print(f"Model Accuracy: {accuracy:.2f}")
        print(f"F1 Score: {f1_score_outcome:.2f}")
        y_pred_proba = self.pipeline.decision_function(X_test)
        auc_score_outcome = roc_auc_score(y_test, y_pred_proba)
        print(f"AUC ROC: {auc_score_outcome:.2f}")

        return TrainingResult(
            pipeline=self.pipeline,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
        )
