from __future__ import annotations
from typing import Any, Callable
import pandas as pd


class DependencyError(RuntimeError):
    pass


class MssqlConnector:
    """
    MSSQL connector that provides a run_sql callable after connecting.
    """

    def __init__(self) -> None:
        self.dialect: str | None = None
        self.run_sql: Callable[[str], pd.DataFrame] | None = None
        self.run_sql_is_set: bool = False

    def connect_to_mssql(self, odbc_conn_str: str, **kwargs: Any) -> None:
        """
        Connect to a Microsoft SQL Server database and set self.run_sql
        to execute queries returning pandas DataFrames.
        """
        try:
            import sqlalchemy as sa
            from sqlalchemy.engine import URL
        except ImportError as e:
            raise DependencyError(
                "You need to install required dependencies to execute this method, run command: pip install sqlalchemy"
            ) from e

        try:
            import pyodbc  # noqa: F401
        except ImportError as e:
            raise DependencyError(
                "You need to install required dependencies to execute this method, run command: pip install pyodbc"
            ) from e

        connection_url = URL.create(
            "mssql+pyodbc", query={"odbc_connect": odbc_conn_str}
        )

        from sqlalchemy import create_engine

        engine = create_engine(connection_url, **kwargs)

        def run_sql_mssql(sql: str) -> pd.DataFrame:
            # Execute the SQL statement and return the result as a pandas DataFrame
            with engine.begin() as conn:
                df = pd.read_sql_query(sa.text(sql), conn)
                return df

        self.dialect = "T-SQL / Microsoft SQL Server"
        self.run_sql = run_sql_mssql
        self.run_sql_is_set = True
