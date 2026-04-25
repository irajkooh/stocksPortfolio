from fastapi import APIRouter
from pydantic import BaseModel
from services.optimizer import optimize_portfolio

router = APIRouter(prefix="/optimizer", tags=["optimizer"])


class OptimizeRequest(BaseModel):
    tickers:   list[str]
    budget:    float = 100_000.0
    period:    str   = "2y"
    timesteps: int   = 10_000


@router.post("/run")
def run_optimizer(body: OptimizeRequest):
    result = optimize_portfolio(
        tickers   = [t.upper() for t in body.tickers],
        budget    = body.budget,
        period    = body.period,
        timesteps = body.timesteps,
    )
    if "error" in result:
        return result
    # Drop non-serialisable fields before returning
    return {k: v for k, v in result.items()
            if k not in ("returns_df", "prices_df", "final_weights")}
