from __future__ import annotations
from typing import Any, Callable, Iterable
import os
import pandas as pd
from urllib.parse import quote_plus
from ...config import settings


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
        If odbc_conn_str is not provided, it will be constructed from environment variables:
        DB_HOST, DB_PORT, DB_USER, DB_PASSWORD, DB_NAME, DB_DRIVER.
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

        # Prefer explicitly supplied connection string, otherwise construct from env vars
        if not odbc_conn_str:
            host = os.getenv("DB_HOST")
            port = os.getenv("DB_PORT", "1433")
            user = os.getenv("DB_USER")
            password = os.getenv("DB_PASSWORD")
            database = os.getenv("DB_NAME")
            driver = os.getenv("DB_DRIVER", "ODBC Driver 18 for SQL Server")

            missing = [name for name, val in (
                ("DB_HOST", host),
                ("DB_USER", user),
                ("DB_PASSWORD", password),
                ("DB_NAME", database),
            ) if not val]
            if missing:
                raise ValueError(
                    f"Missing required environment variables for MSSQL connection: {', '.join(missing)}"
                )

            # Build a safe ODBC connection string. Encrypt/trust can be adapted via env later if needed.
            raw_conn = (
                f"Driver={{{driver}}};Server={host},{port};Database={database};"
                f"UID={user};PWD={password};TrustServerCertificate=yes;Encrypt=no"
            )
            # URL-encode for SQLAlchemy's odbc_connect
            odbc_conn_str = quote_plus(raw_conn)
        else:
            # If a raw ODBC string was explicitly provided, ensure it is URL-encoded
            odbc_conn_str = quote_plus(odbc_conn_str)

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
        cache_dir: str | None = None,
        sample_rows: int | None = None,
        max_common_values: int | None = None,
        include_tables: Iterable[str] | None = None,
        exclude_tables: Iterable[str] | None = None,
    ) -> None:
        """Build a cache of common values per table for allowed schemas.

        Mirrors the legacy DatabaseConnector.connect_and_cache behavior but
        uses the SQLAlchemy engine established by this connector.
        """
        if self.engine is None:
            # Try to connect from env if not already connected
            self.connect_to_mssql()

        # Defaults from settings if not provided
        cache_dir = cache_dir or settings.GRAPH_CACHE_DIR
        sample_rows = int(sample_rows or settings.GRAPH_SAMPLE_ROWS)
        max_common_values = int(max_common_values or settings.GRAPH_MAX_COMMON_VALUES)

        os.makedirs(cache_dir, exist_ok=True)
        print(f"[connect_and_cache] cache_dir={cache_dir}, sample_rows={sample_rows}, max_common_values={max_common_values}")

        try:
            import sqlalchemy as sa
        except ImportError as e:
            raise DependencyError(
                "sqlalchemy is required for caching. pip install sqlalchemy"
            ) from e

        insp = sa.inspect(self.engine)  # type: ignore[arg-type]

        excluded_schemas = {"sys", "INFORMATION_SCHEMA"}
        # default allowed schemas (configurable)
        if allowed_schemas is None:
            if settings.GRAPH_ALLOWED_SCHEMAS:
                allowed_schemas = set(settings.GRAPH_ALLOWED_SCHEMAS)
            else:
                try:
                    all_schemas = set(insp.get_schema_names())
                except Exception:
                    all_schemas = set()
                allowed_schemas = {s for s in all_schemas if s not in excluded_schemas}
        print(f"[connect_and_cache] Schemas seleccionados para cache: {sorted(list(set(allowed_schemas)))}")

        # Normalize include/exclude sets of fully-qualified names schema.table
        include_set = set(include_tables) if include_tables else set(settings.GRAPH_INCLUDE_TABLES or [])
        exclude_set = set(exclude_tables) if exclude_tables else set(settings.GRAPH_EXCLUDE_TABLES or [])
        if include_set:
            print(f"[connect_and_cache] Filtro include (schema.table): {sorted(list(include_set))}")
        if exclude_set:
            print(f"[connect_and_cache] Filtro exclude (schema.table): {sorted(list(exclude_set))}")

        def _is_name_safe(s: str) -> bool:
            try:
                s.encode("utf-8")
            except Exception:
                return False
            return s.isascii()

        # iterate schemas and tables
        total_seen = 0
        total_cached = 0
        total_existing = 0
        total_filtered = 0
        for schema in insp.get_schema_names():
            if schema in excluded_schemas or schema not in set(allowed_schemas) or not _is_name_safe(schema):
                continue
            try:
                tables = insp.get_table_names(schema=schema)
            except Exception:
                continue
            for table_name in tables:
                if not _is_name_safe(table_name):
                    print(f"[connect_and_cache] Skip nombre no ASCII: {schema}.{table_name}")
                    total_filtered += 1
                    continue
                fq_name = f"{schema}.{table_name}"
                if include_set and fq_name not in include_set:
                    total_filtered += 1
                    continue
                if fq_name in exclude_set:
                    total_filtered += 1
                    continue
                cache_file = os.path.join(cache_dir, f"{schema}.{table_name}.json")
                if os.path.exists(cache_file):
                    total_existing += 1
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
                total_cached += 1
                print(f"[connect_and_cache] Cached {cache_file} (cols={len(df.columns)}, rows_sampled={len(df)})")
                total_seen += 1
        print(f"[connect_and_cache] Resumen: cached_nuevos={total_cached}, ya_existian={total_existing}, filtrados={total_filtered}")
