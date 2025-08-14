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
            "DRIVER={ODBC Driver 17 for SQL Server};"
            f"SERVER={server},{self.port};"
            f"DATABASE={self.database};"
            f"UID={self.user};PWD={self.password};"
            "TrustServerCertificate=yes;"
        )

    def connect_and_cache(self) -> None:
        """Connect to the database and cache sample values for each table."""
        conn = pyodbc.connect(self._connection_string())
        cursor = conn.cursor()
        tables = [row.table_name for row in cursor.tables(tableType="TABLE")]
        for table_name in tables:
            cache_file = os.path.join(self.cache_dir, f"{table_name}.json")
            if os.path.exists(cache_file):
                continue
            df = pd.read_sql_query(f"SELECT TOP 300 * FROM {table_name}", conn)
            freq_dict: dict[str, list] = {}
            for col in df.columns:
                common_values = (
                    df[col].value_counts().keys().tolist()[: min(15, len(df[col].value_counts()))]
                )
                freq_dict[col] = common_values
            with open(cache_file, "w") as f:
                json.dump(freq_dict, f, indent=2)
            print(f"Cached {cache_file}")
        conn.close()
