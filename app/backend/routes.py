"""
routes.py
---------
Flask Blueprint with all API endpoints.
"""

from flask import Blueprint, request, jsonify, current_app
from .predict import predict
from .model_loader import get_model
from .rss import get_cached_feed, invalidate_cache
import logging
import time

logger = logging.getLogger(__name__)
api_bp = Blueprint("api", __name__, url_prefix="/api")


@api_bp.route("/predict", methods=["POST"])
def predict_route():
    """
    POST /api/predict
    Body: { "text": "...", "headline": "..." }

    Returns prediction JSON or error JSON.
    """
    start = time.time()

    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Request body must be JSON."}), 400

    # Combine headline + article body if both provided
    headline = (data.get("headline") or "").strip()
    body     = (data.get("text") or "").strip()

    if not headline and not body:
        return jsonify({"error": "Please provide a headline or article text."}), 400

    # Prepend headline to body for richer context
    combined = f"{headline} {body}".strip() if headline else body

    try:
        model = get_model()
        result = predict(combined, model)

        if "error" in result:
            return jsonify(result), 422

        result["latency_ms"] = round((time.time() - start) * 1000, 1)
        return jsonify(result), 200

    except Exception as e:
        logger.exception("Prediction error")
        return jsonify({"error": f"Server error: {str(e)}"}), 500


@api_bp.route("/health", methods=["GET"])
def health():
    """GET /api/health — liveness check."""
    try:
        model = get_model()
        return jsonify({
            "status"    : "ok",
            "model"     : "Multinomial Naïve Bayes",
            "vocab_size": len(model.feature_names),
            "metadata"  : model.metadata,
        }), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 503


@api_bp.route("/examples", methods=["GET"])
def examples():
    """GET /api/examples — return sample articles for demo."""
    return jsonify({
        "examples": [
            {
                "label": "Real",
                "headline": "Senate confirms new Secretary of State",
                "text": (
                    "The Senate confirmed the new Secretary of State by a "
                    "bipartisan vote of 72 to 28, officials announced on Monday. "
                    "The nominee received support from both Republican and "
                    "Democratic lawmakers following weeks of hearings."
                ),
            },
            {
                "label": "Fake",
                "headline": "DEEP STATE CAUGHT destroying evidence!!!",
                "text": (
                    "BREAKING!!! The deep state was CAUGHT destroying evidence!!! "
                    "Share this before it gets DELETED!! Anonymous sources confirm "
                    "the mainstream media is hiding the truth from the people. "
                    "They cannot silence us anymore!! WAKE UP AMERICA!!!"
                ),
            },
        ]
    }), 200


@api_bp.route("/feed", methods=["GET"])
def feed_route():
    """GET /api/feed — return live RSS articles with predictions."""
    force = request.args.get("refresh") == "1"
    if force:
        invalidate_cache()
    try:
        model = get_model()
        data  = get_cached_feed(model)
        return jsonify(data), 200
    except Exception as e:
        logger.exception("Feed error")
        return jsonify({"error": str(e), "articles": []}), 500
