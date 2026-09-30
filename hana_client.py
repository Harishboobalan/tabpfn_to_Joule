"""Safe SAP HANA table loader using HANA_* settings from .env."""

import re

import pandas as pd

from config import boolean, required

_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_$]*$")


def _quoted_identifier(value: str) -> str:
    if not _IDENTIFIER.fullmatch(value):
        raise ValueError("Schema and table names may contain only letters, digits, _, and $.")
    return f'"{value}"'


def load_table(table_name: str) -> pd.DataFrame:
    """Load all rows from HANA_SCHEMA.table_name into a DataFrame."""
    try:
        from hdbcli import dbapi
    except ImportError as exc:
        raise RuntimeError("Install the SAP HANA Python driver: pip install hdbcli") from exc

    connection = dbapi.connect(
        address=required("HANA_HOST"),
        port=int(required("HANA_PORT")),
        user=required("HANA_USER"),
        password=required("HANA_PASSWORD"),
        encrypt=boolean("HANA_ENCRYPT", True),
        sslValidateCertificate=boolean("HANA_SSL_VALIDATE_CERTIFICATE", True),
    )
    cursor = connection.cursor()
    try:
        query = f"SELECT * FROM {_quoted_identifier(required('HANA_SCHEMA'))}.{_quoted_identifier(table_name)}"
        cursor.execute(query)
        columns = [column[0] for column in cursor.description]
        return pd.DataFrame(cursor.fetchall(), columns=columns)
    finally:
        cursor.close()
        connection.close()
