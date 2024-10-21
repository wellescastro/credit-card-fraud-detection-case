from typing import Callable, Dict

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class NormalizationTransformer(BaseEstimator, TransformerMixin):
    def __init__(
        self, normalization_funcs: Dict[str, Callable[[pd.Series], pd.Series]] = None
    ):
        """
        Initializes the NormalizationTransformer with normalization functions.

        :param normalization_funcs: A dictionary where keys are column names and values are functions to apply for normalization.
        """
        self.normalization_funcs = normalization_funcs or {}

    def fit(self, X, y=None):
        """
        Fit method does nothing as this transformer does not learn from data.

        :param X: pandas DataFrame to fit.
        :param y: Ignored.
        :return: self
        """
        return self

    def transform(self, X, y=None):
        """
        Apply normalization transformations to the DataFrame.

        :param X: pandas DataFrame to be transformed.
        :param y: Ignored.
        :return: Normalized pandas DataFrame.
        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input should be a pandas DataFrame.")

        X_normalized = X.copy()
        for column, func in self.normalization_funcs.items():
            if column in X_normalized.columns:
                X_normalized[column] = func(X_normalized[column])
            else:
                raise ValueError(f"Column '{column}' does not exist in the DataFrame.")

        return X_normalized

    def set_output(self, *, transform=None):
        """Set output container."""
        return self


class DataPreparationTransformer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        duplicates_keep_strategy="first",
    ):
        """
        Initializes the DataPreparationTransformer with parameters for preprocessing.

        :param keep: Determines which duplicates to keep.
                     'first' (default) keeps the first occurrence,
                     'last' keeps the last occurrence, and
                     False drops all duplicates.
        """
        self.keep = duplicates_keep_strategy

    def fit(self, X, y=None):
        """
        Fit method does nothing as this transformer does not learn from data.

        :param X: pandas DataFrame to fit.
        :param y: Ignored.
        :return: self
        """
        return self

    def transform(self, X, y=None):
        """
        Apply data cleaning transformations to the DataFrame.

        :param X: pandas DataFrame to be transformed.
        :param y: Ignored.
        :return: Cleaned pandas DataFrame.
        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input should be a pandas DataFrame.")

        X_cleaned = X.copy()
        X_cleaned = self._clear_duplicate_rows(X_cleaned)

        return X_cleaned

    def _clear_duplicate_rows(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Remove duplicate rows based on all columns.

        :param X: DataFrame to remove duplicates from.
        :return: DataFrame without duplicate rows.
        """
        X_cleaned = X.drop_duplicates(keep=self.keep).reset_index(drop=True)
        return X_cleaned


class InvalidRowsCleaner(BaseEstimator, TransformerMixin):
    def __init__(self, column_name, valid_values=[0, 1]):
        """
        Initialize the transformer with the column name to check for invalid values.
        :param column_name: Name of the column to clean.
        :param valid_values: List of valid values to retain (default is [0, 1]).
        """
        self.column_name = column_name
        self.valid_values = valid_values

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame):
        """
        Remove rows where the column has invalid values.
        :param X: DataFrame to be transformed.
        :return: DataFrame with rows containing only valid values in the specified column.
        """

        X_filtered = X[X[self.column_name].isin([str(v) for v in self.valid_values])]
        X_filtered[self.column_name] = X_filtered[self.column_name].astype(
            type(self.valid_values[0])
        )

        return X_filtered


class ColumnDropper(BaseEstimator, TransformerMixin):
    def __init__(self, columns_to_drop):
        """
        Initialize the transformer with the columns to drop.
        :param columns_to_drop: List of column names to drop from the DataFrame.
        """
        self.columns_to_drop = columns_to_drop

    def fit(self, X, y=None):
        # This transformer doesn't require fitting
        return self

    def transform(self, X):
        """
        Drop specified columns from the DataFrame.
        :param X: DataFrame to transform.
        :return: DataFrame with specified columns dropped.
        """
        X = pd.DataFrame(X)
        X = X.drop(columns=self.columns_to_drop, errors="ignore")
        return X

    def set_output(self, *, transform=None):
        """Set output container."""
        return self


def normalize_text(series: pd.Series) -> pd.Series:
    return series.str.strip('"').str.lower()
