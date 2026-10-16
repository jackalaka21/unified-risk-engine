"""FastAPI scoring service. One lock serialises scoring: the neighbour index is mutable and order-sensitive."""
from __future__ import annotations

import os
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from serving.graph_index import OutOfOrderError


class Transaction(BaseModel):
    model_config = ConfigDict(extra="allow")                 # any other raw column is optional and passed through
    TransactionDT: int = Field(ge=0, description="seconds on the dataset's clock; must not go backwards")
    TransactionAmt: float = Field(ge=0, allow_inf_nan=False)


class Batch(BaseModel):
    transactions: list[Transaction] = Field(min_length=1, max_length=256)


def create_app(scorer=None, artifacts=None) -> FastAPI:
    state = {"scorer": scorer}
    lock = threading.Lock()

    @asynccontextmanager
    async def lifespan(app):
        if state["scorer"] is None:
            from serving.bundle import load_bundle

            state["scorer"] = load_bundle(artifacts or os.environ.get("URE_ARTIFACTS", "artifacts"))
        yield

    app = FastAPI(title="Unified risk engine scorer", lifespan=lifespan)

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request, exc):
        # FastAPI's default handler echoes the offending input, which cannot be serialised for inf or nan
        return JSONResponse(status_code=422, content={"detail": [{"loc": list(e["loc"]), "msg": e["msg"], "type": e["type"]}
                                                                  for e in exc.errors()]})

    def current():
        if state["scorer"] is None:
            raise HTTPException(503, "model not loaded")
        return state["scorer"]

    def run(payloads):
        s = current()
        try:
            with lock:
                return s.score_batch(payloads), s.version
        except OutOfOrderError as e:
            raise HTTPException(409, str(e))
        except ValueError as e:
            raise HTTPException(422, str(e))

    @app.get("/health")
    def health():
        s = current()
        return {"status": "ok", "indexed_transactions": len(s.index), "model_version": s.version}

    @app.get("/version")
    def version():
        s = current()
        return {"model_version": s.version, **s.info}

    @app.post("/score")
    def score(txn: Transaction):
        probs, v = run([txn.model_dump()])
        return {"fraud_probability": probs[0], "model_version": v}

    @app.post("/score_batch")
    def score_batch(batch: Batch):
        probs, v = run([t.model_dump() for t in batch.transactions])
        return {"fraud_probabilities": probs, "model_version": v}

    return app


app = create_app()
