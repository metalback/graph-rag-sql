from __future__ import annotations
from typing import Any, Callable, Iterable
import os
import pandas as pd


class DependencyError(RuntimeError):
    pass


class MSSQLConnector:
    """
    MSSQL connector that provides a run_sql callable after connecting and
    an optional cache builder similar to the legacy DatabaseConnector.
    """

    def __init__(self) -> None:
        self.dialect: str | None = None
        self.run_sql: Callable[[str], pd.DataFrame] | None = None
        self.run_sql_is_set: bool = False
        self.engine = None

    def connect_to_mssql(self, odbc_conn_str: str | None = None, **kwargs: Any) -> None:
        """
        Connect to a Microsoft SQL Server database and set self.run_sql
        to execute queries returning pandas DataFrames.
        If odbc_conn_str is None, it will be read from env MSSQL_ODBC_CONN_STR.
        """
        try:
            import sqlalchemy as sa
            from sqlalchemy.engine import URL
            from sqlalchemy import create_engine
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

        odbc_conn_str = odbc_conn_str or os.getenv("MSSQL_ODBC_CONN_STR")
        if not odbc_conn_str:
            raise ValueError("MSSQL_ODBC_CONN_STR is not set and no odbc_conn_str was provided")

        connection_url = URL.create(
            "mssql+pyodbc", query={"odbc_connect": odbc_conn_str}
        )

        self.engine = create_engine(connection_url, **kwargs)

        def run_sql_mssql(sql: str) -> pd.DataFrame:
            # Execute the SQL statement and return the result as a pandas DataFrame
            with self.engine.begin() as conn:  # type: ignore
                df = pd.read_sql_query(sa.text(sql), conn)
                return df

        self.dialect = "T-SQL / Microsoft SQL Server"
        self.run_sql = run_sql_mssql
        self.run_sql_is_set = True

    def connect_and_cache(
        self,
        allowed_schemas: Iterable[str] | None = None,
        cache_dir: str = "cache",
        sample_rows: int = 300,
        max_common_values: int = 15,
    ) -> None:
        """Build a cache of common values per table for allowed schemas.

        Mirrors the legacy DatabaseConnector.connect_and_cache behavior but
        uses the SQLAlchemy engine established by this connector.
        """
        if self.engine is None:
            # Try to connect from env if not already connected
            self.connect_to_mssql()

        os.makedirs(cache_dir, exist_ok=True)

        try:
            import sqlalchemy as sa
        except ImportError as e:
            raise DependencyError(
                "sqlalchemy is required for caching. pip install sqlalchemy"
            ) from e

        insp = sa.inspect(self.engine)  # type: ignore[arg-type]

        # default allowed schemas (as in legacy)
        if allowed_schemas is None:
            allowed_schemas = {
                "bdi",
                "tesoreria_afc",
                "tesoreria_conciliacion_bancaria",
                "tesoreria_fondo_unico",
            }
        excluded_schemas = {"sys", "INFORMATION_SCHEMA"}

        def _is_name_safe(s: str) -> bool:
            try:
                s.encode("utf-8")
            except Exception:
                return False
            return s.isascii()

        # iterate schemas and tables
        for schema in insp.get_schema_names():
            if schema in excluded_schemas or schema not in set(allowed_schemas) or not _is_name_safe(schema):
                continue
            try:
                tables = insp.get_table_names(schema=schema)
            except Exception:
                continue
            for table_name in tables:
                if not _is_name_safe(table_name):
                    continue
                cache_file = os.path.join(cache_dir, f"{schema}.{table_name}.json")
                if os.path.exists(cache_file):
                    continue
                fqtn = f"[{schema}].[{table_name}]"
                try:
                    query = f"SELECT TOP {int(sample_rows)} * FROM {fqtn}"
                    with self.engine.begin() as conn:  # type: ignore
                        df = pd.read_sql_query(query, conn)
                except Exception as e:
                    print(f"WARN: No se pudo leer {fqtn}: {e}")
                    continue
                freq_dict: dict[str, list] = {}
                for col in df.columns:
                    vc = df[col].value_counts()
                    common_values = vc.keys().tolist()[: min(max_common_values, len(vc))]
                    freq_dict[col] = common_values
                with open(cache_file, "w") as f:
                    import json
                    json.dump(freq_dict, f, indent=2, default=str)
                print(f"Cached {cache_file}")
