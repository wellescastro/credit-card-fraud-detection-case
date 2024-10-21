from dataclasses import dataclass
from typing import Any, Dict, List

import pandas as pd


@dataclass
class Schema:
    columns: List[Dict[str, Any]]


def dataframe_to_schema(df: pd.DataFrame) -> Schema:
    """Converts a pandas DataFrame to a schema representation.

    Parameters:
        df (pd.DataFrame): The DataFrame to be converted.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries representing column schema.
    """
    columns = []
    for column_name, dtype in df.dtypes.items():
        column_schema = {"name": column_name, "type": str(dtype)}
        columns.append(column_schema)
    return Schema(columns)
