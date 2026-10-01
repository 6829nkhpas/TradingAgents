"""Console entry point for serving the optional research API."""

import os


def main() -> None:
    try:
        import uvicorn
    except ImportError as exc:
        message = 'Install the API dependencies with `pip install "tradingagents[api]"`.'
        raise SystemExit(message) from exc

    host = os.getenv("TRADINGAGENTS_API_HOST", "0.0.0.0")
    port = int(os.getenv("TRADINGAGENTS_API_PORT", "8000"))
    uvicorn.run("tradingagents.api.app:app", host=host, port=port)


if __name__ == "__main__":
    main()
