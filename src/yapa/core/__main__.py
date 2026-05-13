"""Application entry point (`python -m yapa.core`)."""

import uvicorn

uvicorn.run(
    "yapa.core.app:app",
    host="127.0.0.1",
    port=8000,
    reload=True,
)
