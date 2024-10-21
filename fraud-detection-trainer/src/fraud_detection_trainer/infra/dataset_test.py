from io import StringIO

import pandas as pd

from fraud_detection_trainer.infra.dataset import CSVReader


def test_csv_reader_read_all_rows():
    csv_data = """col1,col2,col3
    1,2,3
    4,5,6
    7,8,9"""
    filepath = StringIO(csv_data)
    reader = CSVReader(filepath)
    df = reader.read()

    expected_data = [(1, 2, 3), (4, 5, 6), (7, 8, 9)]
    expected_df = pd.DataFrame(expected_data, columns=["col1", "col2", "col3"])

    pd.testing.assert_frame_equal(df, expected_df)


def test_csv_reader_read_with_limit():
    csv_data = """col1,col2,col3
    1,2,3
    4,5,6
    7,8,9"""
    filepath = StringIO(csv_data)
    reader = CSVReader(filepath, limit=2)
    df = reader.read()

    expected_data = [(1, 2, 3), (4, 5, 6)]
    expected_df = pd.DataFrame(expected_data, columns=["col1", "col2", "col3"])

    pd.testing.assert_frame_equal(df, expected_df)


def test_csv_reader_with_additional_kwargs():
    csv_data = """col1,col2,col3
    1,2,3
    4,5,6
    7,8,9"""
    filepath = StringIO(csv_data)
    reader = CSVReader(filepath, skipfooter=1)
    df = reader.read()

    expected_data = [(1, 2, 3), (4, 5, 6)]
    expected_df = pd.DataFrame(expected_data, columns=["col1", "col2", "col3"])

    print(df.head(10))
    print(expected_df.head(10))

    pd.testing.assert_frame_equal(df, expected_df)
