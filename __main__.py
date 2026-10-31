"""Launch with:  python -m backend   (from the brain_ai directory)"""
import logging
import sys

from .config import Settings


def main() -> None:
    settings = Settings.from_env()
    if not settings.is_loopback() and not settings.allow_non_local_bind:
        sys.exit(f"Refusing to bind to {settings.host!r}: this prototype has no authentication. "
                 "Use 127.0.0.1, or set BRAIN_ALLOW_NON_LOCAL_BIND=1 if you really know what you are doing.")
    import uvicorn
    from .app import create_app
    logging.basicConfig(level=logging.INFO)
    mode = "MOCK model (no real reasoning)" if settings.model_backend == "mock" else f"HTTP model '{settings.model_name}'"
    print(f"Brain AI prototype → http://{settings.host}:{settings.port}   [{mode}]")
    uvicorn.run(create_app(settings), host=settings.host, port=settings.port, log_level="info")


if __name__ == "__main__":
    main()
