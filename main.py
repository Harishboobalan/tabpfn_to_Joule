"""Single FastAPI route for HANA-to-TabPFN model comparison."""

import asyncio
import os

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from comparison import compare
from config import required
from hana_client import load_table

app = FastAPI(title="HANA TabPFN Comparison")


class ComparisonRequest(BaseModel):
    target_column: str


@app.post("/compare")
async def compare_hana_table(request: ComparisonRequest):
    """Compare TabPFN, Random Forest, SVM, and Logistic Regression."""
    try:
        table = required("HANA_TABLE")
        data = await asyncio.to_thread(load_table, table)
        return await asyncio.to_thread(compare, data, request.target_column)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


if __name__ == "__main__":
    uvicorn.run(
        app,
        host=os.getenv("HOST", "127.0.0.1"),
        port=int(os.getenv("PORT", "8080")),
    )
