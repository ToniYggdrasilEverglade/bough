# frontend/

`backend/app.py` mounts this directory as static files at `/` (with `html=True`, so `index.html` is served at the root).

The original dashboard files were not included in the uploaded backup. Drop `index.html` and its assets here to restore the UI; the API (`/api/*`) and WebSocket (`/ws`) routes work without it.
