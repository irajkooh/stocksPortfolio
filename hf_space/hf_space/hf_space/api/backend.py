from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from core.database import init_db
from api.routes import portfolio, stocks, optimizer, chat


def create_api() -> FastAPI:
    app = FastAPI(title="Portfolio Manager API", version="1.0.0", docs_url="/api/docs")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(portfolio.router, prefix="/api")
    app.include_router(stocks.router,    prefix="/api")
    app.include_router(optimizer.router, prefix="/api")
    app.include_router(chat.router,      prefix="/api")

    @app.on_event("startup")
    async def _startup():
        init_db()

    @app.get("/health")
    def health():
        return {"status": "ok"}

    return app
