"""
app.py
------
Flask application factory.
Run with:  python app.py
Or with:   flask --app app run
"""

import os
import logging
import socket
from flask import Flask, send_from_directory

try:
    from .backend.routes import api_bp
except ImportError:
    from backend.routes import api_bp

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


def create_app() -> Flask:
    app = Flask(
        __name__,
        static_folder=os.path.join(os.path.dirname(__file__), "frontend"),
        static_url_path="",
    )
    app.debug = True
    # ── CORS (allow frontend dev server during development) ───────
    try:
        from flask_cors import CORS
        CORS(app, resources={r"/api/*": {"origins": "*"}})
    except ImportError:
        pass  # flask-cors is optional

    # ── Register API blueprint ────────────────────────────────────
    app.register_blueprint(api_bp)

    # ── Serve the frontend SPA ────────────────────────────────────
    @app.route("/", defaults={"path": ""})
    @app.route("/<path:path>")
    def serve_frontend(path):
        frontend_dir = os.path.join(os.path.dirname(__file__), "frontend")
        if path and os.path.exists(os.path.join(frontend_dir, path)):
            return send_from_directory(frontend_dir, path)
        return send_from_directory(frontend_dir, "index.html")

    return app


def resolve_port(preferred_port: int = 5000, max_attempts: int = 20) -> int:
    """Return the first available port at or above the preferred port."""
    port = int(os.environ.get("PORT", preferred_port))

    for _ in range(max_attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            if sock.connect_ex(("127.0.0.1", port)) != 0:
                return port
        port += 1

    return int(os.environ.get("PORT", preferred_port))


# Expose the WSGI app object for Vercel
app = create_app()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
