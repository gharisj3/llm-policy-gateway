"""HTTP application entry point."""

from fastapi import FastAPI

app = FastAPI(title="LLM Policy Gateway")


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}
