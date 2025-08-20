import os
import json
import pandas as pd
import pyodbc


class DatabaseConnector:
    """Connects to a SQL Server database and caches sample data."""

    def __init__(
        self,
        host: str | None = None,
        ip: str | None = None,
        port: int | str | None = None,
        user: str | None = None,
        password: str | None = None,
        database: str | None = None,
    ) -> None:
        self.host = host or os.environ.get("DB_HOST")
        self.ip = ip or os.environ.get("DB_IP")
        self.port = str(port or os.environ.get("DB_PORT", 1433))
        self.user = user or os.environ.get("DB_USER")
        self.password = password or os.environ.get("DB_PASSWORD")
        self.database = database or os.environ.get("DB_NAME")
        self.cache_dir = "cache"
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)

    def _connection_string(self) -> str:
        server = self.host or self.ip or ""
        return (
            "DRIVER={ODBC Driver 18 for SQL Server};"
            f"SERVER={server},{self.port};"
            f"DATABASE={self.database};"
            f"UID={self.user};PWD={self.password};"
            "TrustServerCertificate=yes;"
        )

    def connect_and_cache(self) -> None:
        """Connect to the database and cache sample values for each table."""
        conn = pyodbc.connect(self._connection_string())
        # Normaliza codificación para evitar errores de Unicode (e.g., UTF-16-LE de NVARCHAR)
        try:
            conn.setdecoding(pyodbc.SQL_CHAR, encoding="utf-8")
            conn.setdecoding(pyodbc.SQL_WCHAR, encoding="utf-8")
            conn.setencoding(encoding="utf-8")
        except Exception:
            # Si la versión de pyodbc no soporta estas APIs, continúa sin romper
            pass
        cursor = conn.cursor()
        # Helper para filtrar nombres no ASCII/UTF-8 seguros
        def _is_name_safe(s: str) -> bool:
            try:
                s.encode("utf-8")
            except Exception:
                return False
            return s.isascii()

        # Recupera solo TABLEs (excluye VIEWs)
        tables = [(row.table_schem, row.table_name) for row in cursor.tables(tableType="TABLE")]
        tables = list({(s, t) for (s, t) in tables})  # únicos
        # Excluir esquemas con objetos del sistema o metadatos
        excluded_schemas = {"sys", "INFORMATION_SCHEMA"}
        # Incluir solo esquemas permitidos para el caso de prueba
        allowed_schemas = {"bdi","tesoreria_afc","tesoreria_conciliacion_bancaria","tesoreria_fondo_unico"}
        # Filtra por seguridad de nombre y por listas de inclusión/exclusión
        tables = [
            (s, t)
            for (s, t) in tables
            if (s in allowed_schemas) and (s not in excluded_schemas) and _is_name_safe(s) and _is_name_safe(t)
        ]
        print([f"{schema}.{table}" for schema, table in tables])
        for schema, table_name in tables:
            cache_file = os.path.join(self.cache_dir, f"{schema}.{table_name}.json")
            if os.path.exists(cache_file):
                continue
            fqtn = f"[{schema}].[{table_name}]"
            try:
                df = pd.read_sql_query(f"SELECT TOP 300 * FROM {fqtn}", conn)
            except Exception as e:
                print(f"WARN: No se pudo leer {fqtn}: {e}")
                continue
            freq_dict: dict[str, list] = {}
            for col in df.columns:
                common_values = (
                    df[col].value_counts().keys().tolist()[: min(15, len(df[col].value_counts()))]
                )
                freq_dict[col] = common_values
            with open(cache_file, "w") as f:
                # Serializa valores no JSON nativos (p.ej., Timestamp, Decimal, numpy types)
                json.dump(freq_dict, f, indent=2, default=str)
            print(f"Cached {cache_file}")
        conn.close()
