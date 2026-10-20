"""Console entry point for serving the research API (``tradingagents-api``)."""

import os


def main() -> None:
    import uvicorn

    host = os.getenv("TRADINGAGENTS_API_HOST", "0.0.0.0")
    port = int(os.getenv("TRADINGAGENTS_API_PORT", "8000"))
    uvicorn.run("tradingagents.api.app:app", host=host, port=port)


if __name__ == "__main__":
    main()
