# database package
# Expose a simple factory to obtain a database connector based on configuration.
import os
from typing import Optional

try:
    from .mssql.connector import MSSQLConnector
except Exception:  # pragma: no cover
    MSSQLConnector = None  # type: ignore


def get_database_connector(provider: Optional[str] = None):
    """Return a database connector instance based on provider.

    provider can be: 'mssql' (default). Future: 'postgresql', 'mariadb'.
    Configuration is read from environment variables.
    """
    provider = (provider or os.getenv("DB_PROVIDER") or os.getenv("DATABASE_PROVIDER") or "mssql").lower()

    if provider == "mssql":
        if MSSQLConnector is None:
            raise RuntimeError("MSSQL connector not available. Ensure dependencies (pyodbc, sqlalchemy) are installed.")
        # Build connection from DB_* environment variables inside the connector
        connector = MSSQLConnector()
        connector.connect_to_mssql()
        return connector

    raise ValueError(f"Unsupported DB provider: {provider}")


__all__ = [
    "get_database_connector",
]