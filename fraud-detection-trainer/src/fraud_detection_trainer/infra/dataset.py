import abc
from typing import Optional

import pandas as pd


class Reader(abc.ABC):
    """Base class for a dataset reader.

    Use a reader to load a dataset in memory.
    e.g., to read file, database, cloud storage, ...

    Parameters:
        limit (int, optional): maximum number of rows to read. Defaults to None.
    """

    @abc.abstractmethod
    def read(self) -> pd.DataFrame:
        """Read a dataframe from a dataset.

        Returns:
            pd.DataFrame: dataframe representation.
        """


class CSVReader(Reader):
    """CSV Reader class to load dataset from a CSV file.

    Parameters:
        filepath (str): Path to the CSV file.
        limit (int, optional): Maximum number of rows to read. Defaults to None.
        **kwargs: Additional arguments to pass to pandas.read_csv().
    """

    def __init__(self, filepath: str, limit: Optional[int] = None, **kwargs):
        self.filepath = filepath
        self.limit = limit
        self.kwargs = kwargs

    def read(self) -> pd.DataFrame:
        """Read a dataframe from a CSV file.

        Returns:
            pd.DataFrame: Dataframe representation of the CSV file.
        """
        df = pd.read_csv(self.filepath, **self.kwargs)
        if self.limit:
            df = df.head(self.limit)
        return df
