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
                    print(f"[connect_and_cache][SQL][sample] {query}")
                    with self.engine.begin() as conn:  # type: ignore
                        df = pd.read_sql_query(query, conn)
                    print(f"[connect_and_cache][SQL][sample] rows={0 if df is None else len(df)} cols={0 if df is None else len(df.columns)}")
                except Exception as e:
                    print(f"WARN: No se pudo leer {fqtn}: {e}")
                    continue
                freq_dict: dict[str, list] = {}
                for col in df.columns:
                    vc = df[col].value_counts()
                    common_values = vc.keys().tolist()[: min(max_common_values, len(vc))]
                    freq_dict[col] = common_values
                # --- Extra metadata: Columns, PKs, FKs, incoming FKs, constraints, procedures ---
                meta: dict[str, object] = {}
                try:
                    with self.engine.begin() as conn:  # type: ignore
                        # Columns info
                        cols_sql = (
                            "SELECT c.name AS column_name, t.name AS data_type, c.max_length, c.is_nullable "
                            "FROM sys.columns c "
                            "JOIN sys.types t ON c.user_type_id = t.user_type_id "
                            "JOIN sys.tables tb ON c.object_id   = tb.object_id "
                            "JOIN sys.schemas s ON tb.schema_id  = s.schema_id "
                            "WHERE s.name = ? AND tb.name = ?  "
                            "ORDER BY c.column_id"
                        )
                        print(f"[connect_and_cache][SQL][columns] fqtn={schema}.{table_name} -> {cols_sql}")
                        cols_df = pd.read_sql_query(cols_sql, conn, params=(schema, table_name))
                        print(f"[connect_and_cache][SQL][columns] rows={0 if cols_df is None else len(cols_df)}")
                        meta["columns_info"] = cols_df.to_dict(orient="records") if not cols_df.empty else []
                        print(f"[connect_and_cache][SQL][columns] meta={meta}")

                        # Primary keys
                        pk_sql = (
                            "SELECT i.name AS pk_name, c.name AS column_name "
                            "FROM sys.indexes i "
                            "JOIN sys.index_columns ic ON i.object_id = ic.object_id AND i.index_id = ic.index_id "
                            "JOIN sys.columns c ON ic.object_id = c.object_id AND ic.column_id = c.column_id "
                            "WHERE i.is_primary_key = 1 AND i.object_id = OBJECT_ID(?) "
                            "ORDER BY ic.key_ordinal"
                        )
                        print(f"[connect_and_cache][SQL][pk] fqtn={schema}.{table_name} -> {pk_sql}")
                        pk_df = pd.read_sql_query(pk_sql, conn, params=(f"{schema}.{table_name}"))
                        print(f"[connect_and_cache][SQL][pk] rows={0 if pk_df is None else len(pk_df)}")
                        meta["pk_columns"] = pk_df["column_name"].tolist() if not pk_df.empty else []

                        # Foreign keys (outgoing from this table)
                        fk_sql = (
                            "SELECT f.name AS fk_name, "
                            "SCHEMA_NAME(OBJECT_SCHEMA_ID(f.parent_object_id)) AS child_schema, "
                            "OBJECT_NAME(f.parent_object_id) AS child_table, "
                            "COL_NAME(fc.parent_object_id, fc.parent_column_id) AS child_column, "
                            "SCHEMA_NAME(OBJECT_SCHEMA_ID(f.referenced_object_id)) AS parent_schema, "
                            "OBJECT_NAME(f.referenced_object_id) AS parent_table, "
                            "COL_NAME(fc.referenced_object_id, fc.referenced_column_id) AS parent_column "
                            "FROM sys.foreign_keys AS f "
                            "INNER JOIN sys.foreign_key_columns AS fc ON f.object_id = fc.constraint_object_id "
                            "WHERE f.parent_object_id = OBJECT_ID(?)"
                        )
                        print(f"[connect_and_cache][SQL][fk_out] fqtn={schema}.{table_name} -> {fk_sql}")
                        fk_df = pd.read_sql_query(fk_sql, conn, params=(f"{schema}.{table_name}"))
                        print(f"[connect_and_cache][SQL][fk_out] rows={0 if fk_df is None else len(fk_df)}")
                        meta["foreign_keys"] = (
                            fk_df.to_dict(orient="records") if not fk_df.empty else []
                        )

                        # Incoming FKs (tables referencing this table)
                        rfk_sql = (
                            "SELECT f.name AS fk_name, "
                            "SCHEMA_NAME(OBJECT_SCHEMA_ID(f.parent_object_id)) AS child_schema, "
                            "OBJECT_NAME(f.parent_object_id) AS child_table, "
                            "COL_NAME(fc.parent_object_id, fc.parent_column_id) AS child_column, "
                            "SCHEMA_NAME(OBJECT_SCHEMA_ID(f.referenced_object_id)) AS parent_schema, "
                            "OBJECT_NAME(f.referenced_object_id) AS parent_table, "
                            "COL_NAME(fc.referenced_object_id, fc.referenced_column_id) AS parent_column "
                            "FROM sys.foreign_keys AS f "
                            "INNER JOIN sys.foreign_key_columns AS fc ON f.object_id = fc.constraint_object_id "
                            "WHERE f.referenced_object_id = OBJECT_ID(?)"
                        )
                        print(f"[connect_and_cache][SQL][fk_in] fqtn={schema}.{table_name} -> {rfk_sql}")
                        rfk_df = pd.read_sql_query(rfk_sql, conn, params=(f"{schema}.{table_name}"))
                        print(f"[connect_and_cache][SQL][fk_in] rows={0 if rfk_df is None else len(rfk_df)}")
                        meta["referenced_by"] = (
                            rfk_df.to_dict(orient="records") if not rfk_df.empty else []
                        )

                        # Unique indexes/constraints
                        uniq_sql = (
                            "SELECT i.name AS index_name, STRING_AGG(c.name, ',') WITHIN GROUP (ORDER BY ic.key_ordinal) AS columns "
                            "FROM sys.indexes i "
                            "JOIN sys.index_columns ic ON i.object_id = ic.object_id AND i.index_id = ic.index_id "
                            "JOIN sys.columns c ON ic.object_id = c.object_id AND ic.column_id = c.column_id "
                            "WHERE i.is_unique = 1 AND i.object_id = OBJECT_ID(?) "
                            "GROUP BY i.name"
                        )
                        print(f"[connect_and_cache][SQL][unique] fqtn={schema}.{table_name} -> {uniq_sql}")
                        uniq_df = pd.read_sql_query(uniq_sql, conn, params=(f"{schema}.{table_name}"))
                        print(f"[connect_and_cache][SQL][unique] rows={0 if uniq_df is None else len(uniq_df)}")
                        meta["unique_indexes"] = uniq_df.to_dict(orient="records") if not uniq_df.empty else []

                        # Check constraints (store definition)
                        check_sql = (
                            "SELECT cc.name AS constraint_name, TRY_CONVERT(NVARCHAR(MAX), cc.definition) AS definition "
                            "FROM sys.check_constraints cc "
                            "WHERE cc.parent_object_id = OBJECT_ID(?)"
                        )
                        print(f"[connect_and_cache][SQL][check] fqtn={schema}.{table_name} -> {check_sql}")
                        check_df = pd.read_sql_query(check_sql, conn, params=(f"{schema}.{table_name}"))
                        print(f"[connect_and_cache][SQL][check] rows={0 if check_df is None else len(check_df)}")
                        meta["check_constraints"] = check_df.to_dict(orient="records") if not check_df.empty else []

                        # Default constraints
                        def_sql = (
                            "SELECT dc.name AS constraint_name, c.name AS column_name, TRY_CONVERT(NVARCHAR(MAX), dc.definition) AS definition "
                            "FROM sys.default_constraints dc "
                            "JOIN sys.columns c ON dc.parent_object_id = c.object_id AND dc.parent_column_id = c.column_id "
                            "WHERE dc.parent_object_id = OBJECT_ID(?)"
                        )
                        print(f"[connect_and_cache][SQL][default] fqtn={schema}.{table_name} -> {def_sql}")
                        def_df = pd.read_sql_query(def_sql, conn, params=(f"{schema}.{table_name}"))
                        print(f"[connect_and_cache][SQL][default] rows={0 if def_df is None else len(def_df)}")
                        meta["default_constraints"] = def_df.to_dict(orient="records") if not def_df.empty else []

                        # Procedures referencing this table (simple LIKE match)
                        proc_sql = (
                            "DECLARE @TableName NVARCHAR(512) = ?; "
                            "SELECT o.name AS proc_name, SCHEMA_NAME(o.schema_id) AS proc_schema, o.type_desc AS obj_type "
                            ", TRY_CONVERT(NVARCHAR(MAX), m.definition) AS definition "
                            "FROM sys.sql_modules m "
                            "INNER JOIN sys.objects o ON m.object_id = o.object_id "
                            "WHERE m.definition LIKE '%' + @TableName + '%' AND o.type = 'P' "
                            "ORDER BY proc_schema, proc_name"
                        )
                        print(f"[connect_and_cache][SQL][procs] fqtn={schema}.{table_name} -> {proc_sql}")
                        proc_df = pd.read_sql_query(proc_sql, conn, params=(f"{schema}.{table_name}"))
                        print(f"[connect_and_cache][SQL][procs] rows={0 if proc_df is None else len(proc_df)}")
                        # To avoid huge JSONs, do not dump definitions by default
                        if not proc_df.empty:
                            meta["procedures"] = [
                                {
                                    "name": r["proc_name"],
                                    "schema": r["proc_schema"],
                                }
                                for _, r in proc_df.iterrows()
                            ]
                        else:
                            meta["procedures"] = []
                except Exception as e:
                    print(f"WARN: No se pudo obtener metadata de {fqtn}: {e}")
                    meta.setdefault("columns_info", [])
                    meta.setdefault("pk_columns", [])
                    meta.setdefault("foreign_keys", [])
                    meta.setdefault("referenced_by", [])
                    meta.setdefault("unique_indexes", [])
                    meta.setdefault("check_constraints", [])
                    meta.setdefault("default_constraints", [])
                    meta.setdefault("procedures", [])

                # Embed metadata under a reserved key
                freq_dict["__meta"] = meta
                with open(cache_file, "w") as f:
                    import json
                    json.dump(freq_dict, f, indent=2, default=str)
                total_cached += 1
                print(f"[connect_and_cache] Cached {cache_file} (cols={len(df.columns)}, rows_sampled={len(df)})")
                total_seen += 1
        print(f"[connect_and_cache] Resumen: cached_nuevos={total_cached}, ya_existian={total_existing}, filtrados={total_filtered}")
