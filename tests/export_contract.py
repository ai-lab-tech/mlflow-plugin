"""Export upstream routes and real serialized GET responses without startup.

Run with Triton Control's dependencies and triton-backend on PYTHONPATH.
Only the readiness transport and model record are simulated.
"""

import asyncio
import json
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from app.api.deployment_api import router
from app.services.deployment import mlflow
from fastapi import FastAPI

app = FastAPI()
app.include_router(router)
row = SimpleNamespace(
    name="iris",
    id=42,
    mlflow_model_uri="s3://models/repository/iris",
    mlflow_model_name="iris",
    health_error=None,
    url="http://triton.test",
    triton_verify_ssl=True,
    triton_ca_certificate=None,
)


async def export():
    responses = {}
    with patch.object(mlflow, "TritonService") as service:
        service.return_value.is_ready = AsyncMock(return_value=False)
        responses["starting"] = await mlflow.describe(row)
        service.return_value.is_ready = AsyncMock(return_value=True)
        service.return_value.get_repository_index = AsyncMock(
            return_value=[{"name": row.mlflow_model_name, "state": "READY"}]
        )
        responses["ready"] = await mlflow.describe(row)
    return {"openapi": app.openapi(), "descriptions": responses}


if __name__ == "__main__":
    with open(sys.argv[1], "w") as output:
        json.dump(asyncio.run(export()), output)
