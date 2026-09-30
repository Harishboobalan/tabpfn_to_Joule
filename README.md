# TabPFN Titanic A2A agent

This service exposes a Titanic-survival classifier through the A2A JSON-RPC
protocol.

## Run locally

Use Python 3.10 or newer. Create and activate a virtual environment, then:

```powershell
pip install -r requirements.txt
python main.py
```

The included `Titanic-Dataset.csv` is used automatically. The first prediction
still downloads and caches TabPFN's model checkpoint, so it can take longer
than later requests. To use another dataset, set `TITANIC_DATA_PATH` to a CSV
with `Survived`, `Pclass`, `Sex`, `Age`, `SibSp`, `Parch`, and `Fare` columns.

## Test the A2A agent

```powershell
Invoke-RestMethod http://127.0.0.1:8080/health
Invoke-RestMethod http://127.0.0.1:8080/.well-known/agent-card.json
```

```powershell
$body = @{
  jsonrpc = "2.0"; id = "demo-1"; method = "message/send"
  params = @{ message = @{
    kind = "message"; messageId = "message-1"; role = "user"
    parts = @(@{ kind = "text"; text = "Would a 30 year old woman in first class survive?" })
  }}
} | ConvertTo-Json -Depth 10

Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8080/ `
  -ContentType "application/json" -Body $body
```

## Use the REST prediction route

For applications that do not need A2A, send the same passenger fields to
`POST /predict`:

```powershell
$passengers = @{
  passengers = @(@{ Pclass = 1; Sex = "female"; Age = 30; SibSp = 0; Parch = 0; Fare = 80 })
} | ConvertTo-Json -Depth 5

Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8080/predict `
  -ContentType "application/json" -Body $passengers
```

`Pclass` (1–3) and `Sex` (`male` or `female`) are required. `Age`, `SibSp`,
`Parch`, and `Fare` are optional; reasonable values from the training data are
used when they are omitted.

Set `A2A_API_KEY` before startup to protect calls (use `X-API-Key` on the
request), and set `A2A_PUBLIC_URL` when the service is deployed behind a
public HTTPS route.
