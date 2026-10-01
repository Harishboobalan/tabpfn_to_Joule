# HANA TabPFN A2A Agent

The application is in `app/` and exposes the SAP HANA-to-TabPFN comparison as an A2A agent.

## Files

- `app/agent.py` — reads HANA data, invokes TabPFN, and scores the baseline models.
- `app/agent_executor.py` — maps A2A tasks to the comparison agent and streams a result artifact.
- `app/app.py` — creates the Agent Card and FastAPI/A2A routes.
- `app/manifest.yaml` — Cloud Foundry deployment configuration.
- `app/test_client.py` — sends a sample A2A request.
- `app/requirements.txt` — Python dependencies (also used by the Cloud Foundry buildpack).
- `da.sapdas.yaml`, `hana_tabpfn_capability/` — Joule capability that calls the agent through the `HANA_TABPFN_AGENT_A2A` destination.

## Local run

Create `app/.env` (or keep `.env` at the repository root) with the existing `HANA_*`, `AICORE_*`, `TABPFN_DEPLOYMENT_ID`, and optional `HANA_TABLE` values.

```powershell
cd app
pip install -r requirements.txt
python app.py   # listens on 0.0.0.0:1000 (override with HOST / PORT)
```

In another terminal:

```powershell
cd app
$env:TARGET_COLUMN = "Survived"
python test_client.py
```

The request is either JSON or plain English. In JSON, `target_column` is required; `table_name` and `test_size` are optional:

```json
{"target_column":"Survived","table_name":"PASSENGERS","test_size":0.2}
```

Plain-English requests (as sent by Joule) are matched against the table's columns, for example
"Compare TabPFN with SVM for the Survived target column" or
"Run a model comparison for the PASSENGERS table using Survived as the target with a 30% test size".
Without a table name, `HANA_TABLE` is used. The final task status message contains a Markdown
accuracy summary (shown by Joule); the full result, including predictions, is in the `comparison-result` artifact.

The Agent Card is at `/.well-known/agent-card.json`; JSON-RPC is at `/`, with A2A REST endpoints also enabled.

## Cloud Foundry

Set `A2A_PUBLIC_URL` in `manifest.yaml` to the public route and uncomment the `services` section with the bound SAP AI Core service instance. Then run `cf push` from the `app` directory.
