import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from sensai.core.config import Config
from sensai.core.export.shares import SHARED_PATH, ShareStore, public_base

# A shared page runs nothing and loads nothing. The link is the only secret: it must not reach a search engine, a cache
# or the site of the next click.
PAGE_HEADERS = {
    "Content-Security-Policy": "default-src 'none'; style-src 'unsafe-inline'; base-uri 'none'; form-action 'none'",
    "Referrer-Policy": "no-referrer",
    "X-Robots-Tag": "noindex, nofollow",
    "X-Content-Type-Options": "nosniff",
    "Cache-Control": "no-store",
}
NOT_FOUND = "<!doctype html><title>Sensai</title><p>This page does not exist, or is no longer shared.</p>"


def create_public_app(shares: ShareStore) -> FastAPI:
    """
    A server of its own, and not a route of the chat: this one faces the network and has no login, so it holds no session,
    no model and no document, only the folder of the pages.
    """
    app = FastAPI(title="Sensai", docs_url=None, redoc_url=None, openapi_url=None)

    @app.get(f"{SHARED_PATH}/{{token}}")
    def shared(token: str) -> HTMLResponse:
        page = shares.page(token)
        if page is None:
            return HTMLResponse(NOT_FOUND, status_code=404)
        return HTMLResponse(page, headers=PAGE_HEADERS)

    return app


def run_public(config: Config, host: str) -> int:
    shares = ShareStore(config.shares_dir)
    print(f"Sensai public pages on {public_base(config)}: {len(shares.list())} (Ctrl+C to stop)", flush=True)
    print("Anyone who has a link can read its page. Nothing else is served.", flush=True)
    # No access log: a line per request would write the links down.
    uvicorn.run(create_public_app(shares), host=host, port=config.public_port, log_level="warning", access_log=False)
    return 0
