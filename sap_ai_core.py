"""SAP AI Core TabPFN deployment client."""

import os
import time
from threading import Lock

import requests

from config import required

_lock = Lock()
_token: str | None = None
_expires_at = 0.0
_deployment_url: str | None = None


def _access_token() -> str:
    global _token, _expires_at
    if _token and time.time() < _expires_at:
        return _token
    with _lock:
        if _token and time.time() < _expires_at:
            return _token
        auth_url = required("AICORE_AUTH_URL").rstrip("/")
        token_url = auth_url if auth_url.endswith("/oauth/token") else f"{auth_url}/oauth/token"
        response = requests.post(
            token_url,
            data={"grant_type": "client_credentials"},
            auth=(required("AICORE_CLIENT_ID"), required("AICORE_CLIENT_SECRET")),
            timeout=30,
        )
        response.raise_for_status()
        body = response.json()
        _token = body["access_token"]
        _expires_at = time.time() + max(int(body.get("expires_in", 300)) - 60, 30)
        return _token


def deployment_url() -> str:
    global _deployment_url
    if _deployment_url:
        return _deployment_url
    api_url = required("AICORE_API_URL").rstrip("/")
    api_url = api_url if api_url.endswith("/v2") else f"{api_url}/v2"
    response = requests.get(
        f"{api_url}/lm/deployments/{required('TABPFN_DEPLOYMENT_ID')}",
        headers={
            "Authorization": f"Bearer {_access_token()}",
            "AI-Resource-Group": os.getenv("AICORE_RESOURCE_GROUP", "default"),
        },
        timeout=30,
    )
    response.raise_for_status()
    _deployment_url = response.json()["deploymentUrl"].rstrip("/")
    return _deployment_url


def predict(payload: dict) -> dict:
    """Send a prepared TabPFN payload to the SAP AI Core deployment."""
    response = requests.post(
        f"{deployment_url()}/predict",
        headers={
            "Authorization": f"Bearer {_access_token()}",
            "AI-Resource-Group": os.getenv("AICORE_RESOURCE_GROUP", "default"),
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=180,
    )
    response.raise_for_status()
    return response.json()
