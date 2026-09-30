# HANA-to-TabPFN Comparison

Fetch a labeled table from SAP HANA, predict through the SAP AI Core TabPFN
deployment, and compare accuracy with Random Forest, SVM, and Logistic
Regression.

## Project structure

- `config.py` — `.env` loading and configuration helpers.
- `hana_client.py` — secure HANA table reader.
- `sap_ai_core.py` — SAP AI Core OAuth and TabPFN deployment client.
- `comparison.py` — shared train/holdout comparison logic.
- `run_hana_comparison.py` — command-line entry point.
- `main.py` — one FastAPI comparison route.

## `.env`

```text
HANA_HOST=...
HANA_PORT=443
HANA_USER=...
HANA_PASSWORD=...
HANA_SCHEMA=...

AICORE_AUTH_URL=...
AICORE_CLIENT_ID=...
AICORE_CLIENT_SECRET=...
AICORE_API_URL=...
AICORE_RESOURCE_GROUP=default
TABPFN_DEPLOYMENT_ID=...
```

For HANA Cloud, encrypted connections are enabled by default. Set
`HANA_SSL_VALIDATE_CERTIFICATE=false` only when your environment requires it.

## Run

```powershell
pip install -r requirements.txt
python run_hana_comparison.py --table YOUR_TABLE --target YOUR_TARGET_COLUMN
```

The command reads `HANA_SCHEMA.YOUR_TABLE`, creates one reproducible 80/20
holdout split, writes predictions and all accuracy scores to `comparison.json`,
and prints the model comparison.

## FastAPI route

```powershell
uvicorn main:app --host 0.0.0.0 --port 8080
```

Call `POST /compare` with only the target column. The table name comes from
`HANA_TABLE` in `.env`:

```json
{"target_column": "Survived"}
```
