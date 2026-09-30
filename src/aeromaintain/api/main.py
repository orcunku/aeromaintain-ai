from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException

from aeromaintain.agents.maintenance_agent import (
    MaintenanceAgent,
)


agent: MaintenanceAgent | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Initialize shared application resources when the API starts.
    """
    global agent

    agent = MaintenanceAgent()

    yield

    agent = None


app = FastAPI(
    title="AeroMaintain AI API",
    description=(
        "Synthetic aircraft maintenance and reliability "
        "decision-support API."
    ),
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/")
def root() -> dict[str, str]:
    """
    Basic API information.
    """
    return {
        "name": "AeroMaintain AI",
        "status": "running",
        "data_type": "synthetic",
    }


@app.get("/health")
def health() -> dict[str, str]:
    """
    Lightweight health-check endpoint.
    """
    return {
        "status": "healthy",
    }


@app.get(
    "/investigate/{component_id}",
)
def investigate_component(
    component_id: str,
    top_k_documents: int = 3,
    history_limit: int = 5,
) -> dict[str, Any]:
    """
    Run an evidence-driven maintenance investigation.

    The response combines:
    - predictive-maintenance risk
    - historical maintenance events
    - retrieved synthetic maintenance evidence

    This endpoint provides synthetic decision-support output only.
    """
    if agent is None:
        raise HTTPException(
            status_code=503,
            detail="Maintenance agent is not initialized.",
        )

    try:
        return agent.investigate_component(
            component_id=component_id,
            top_k_documents=top_k_documents,
            history_limit=history_limit,
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=503,
            detail=(
                "Predictive model artifact is unavailable. "
                f"{exc}"
            ),
        ) from exc


@app.get(
    "/components/{component_id}/risk",
)
def component_risk(
    component_id: str,
) -> dict[str, Any]:
    """
    Return only the predictive-maintenance risk result.
    """
    if agent is None:
        raise HTTPException(
            status_code=503,
            detail="Maintenance agent is not initialized.",
        )

    try:
        return agent.tools.calculate_failure_risk(
            component_id
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=503,
            detail=(
                "Predictive model artifact is unavailable. "
                f"{exc}"
            ),
        ) from exc