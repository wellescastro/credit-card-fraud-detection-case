import pandas as pd
from geopy.distance import geodesic
from sklearn.base import BaseEstimator, TransformerMixin


class FeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X["trans_date_trans_time"] = pd.to_datetime(
            X["trans_date_trans_time"], format="%d-%m-%Y %H:%M"
        )

        # Create new features
        X["transaction_hour"] = X["trans_date_trans_time"].dt.hour
        X["transaction_day"] = X["trans_date_trans_time"].dt.day
        X["transaction_month"] = X["trans_date_trans_time"].dt.month
        X["transaction_day_of_week"] = X["trans_date_trans_time"].dt.dayofweek
        X["transaction_is_weekend"] = (
            X["transaction_day_of_week"].isin([5, 6]).astype(int)
        )
        X["dob"] = pd.to_datetime(X["dob"], format="%d-%m-%Y")
        X["age"] = (X["trans_date_trans_time"] - X["dob"]).dt.days // 365

        # Distance between customer location and merchant location
        X["distance_to_merchant"] = X.apply(
            lambda row: geodesic(
                (row["lat"], row["long"]), (row["merch_lat"], row["merch_long"])
            ).km,
            axis=1,
        )

        return X

    def set_output(self, *, transform=None):
        """Set output container."""
        return self

    def list_numeric_features(self):
        return [
            "transaction_hour",
            "transaction_day",
            "transaction_month",
            "transaction_day_of_week",
            "transaction_is_weekend",
            "age",
            "distance_to_merchant",
        ]

    def _normalize_data(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize specified columns in the DataFrame using provided functions.

        :param X: DataFrame to normalize.
        :return: Normalized DataFrame.
        """
        for column, func in self.normalization_funcs.items():
            if column in X.columns:
                X[column] = func(X[column])
            else:
                raise ValueError(f"Column '{column}' does not exist in the DataFrame.")
        return X
