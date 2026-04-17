"""
model_loader.py
---------------
Loads the trained TF-IDF vectoriser and Multinomial Naïve Bayes model
from disk (joblib bundles or separate pkl files).

Supports two layouts:
  1. Single bundle: models/model_bundle.pkl  (output of your Cell 13)
  2. Separate files: models/vectorizer.pkl + models/model.pkl
"""

import os
import sys
import joblib
import numpy as np
import logging

from .predict import clean_text

logger = logging.getLogger(__name__)


def _register_pickle_compatibility() -> None:
    """Expose notebook-pickled callables under __main__ for joblib loads."""
    main_module = sys.modules.get("__main__")
    if main_module is not None and not hasattr(main_module, "clean_text"):
        setattr(main_module, "clean_text", clean_text)

# ── Paths ─────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")

BUNDLE_PATH     = os.path.join(MODELS_DIR, "model_bundle.pkl")
VECTORIZER_PATH = os.path.join(MODELS_DIR, "vectorizer.pkl")
MODEL_PATH      = os.path.join(MODELS_DIR, "model.pkl")


class ModelLoader:
    """Singleton wrapper around the trained model + vectoriser."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._loaded = False
        return cls._instance

    def load(self):
        if self._loaded:
            return self

        _register_pickle_compatibility()

        # ── Try bundle first ────────────────────────────────────
        if os.path.exists(BUNDLE_PATH):
            logger.info(f"Loading model bundle from {BUNDLE_PATH}")
            bundle = joblib.load(BUNDLE_PATH)
            self.vectorizer  = bundle["vectorizer"]
            self.model       = bundle["model"]
            self.label_map   = bundle.get("label_map", {0: "Fake", 1: "Real"})
            self.metadata    = {
                "accuracy"   : bundle.get("accuracy"),
                "f1_macro"   : bundle.get("f1_macro"),
                "vocab_size" : bundle.get("vocab_size"),
                "n_train"    : bundle.get("n_train"),
            }
            logger.info("Bundle loaded successfully.")

        # ── Fall back to separate files ──────────────────────────
        elif os.path.exists(VECTORIZER_PATH) and os.path.exists(MODEL_PATH):
            logger.info("Loading vectorizer + model from separate files.")
            self.vectorizer = joblib.load(VECTORIZER_PATH)
            self.model      = joblib.load(MODEL_PATH)
            self.label_map  = {0: "Fake", 1: "Real"}
            self.metadata   = {}
        else:
            raise FileNotFoundError(
                f"No model files found.\n"
                f"Expected one of:\n"
                f"  {BUNDLE_PATH}\n"
                f"  {VECTORIZER_PATH} + {MODEL_PATH}\n\n"
                f"Run your notebook Cell 13 to generate model_bundle.pkl, "
                f"then copy it to the models/ folder."
            )

        # Pre-compute feature names for importance extraction
        self.feature_names = np.array(
            self.vectorizer.get_feature_names_out()
        )

        # Pre-compute log-prob ratios (Fake − Real) for feature importance
        lp = self.model.feature_log_prob_  # shape (2, n_features)
        self.diff_fake = lp[0] - lp[1]    # positive → fake indicator
        self.diff_real = lp[1] - lp[0]    # positive → real indicator

        self._loaded = True
        logger.info(f"Vocabulary size: {len(self.feature_names):,}")
        return self

    def predict_fn(self, text: str) -> dict:
        """Convenience wrapper so rss.py doesn't need to import predict()."""
        from .predict import predict as _predict
        return _predict(text, self)


def get_model() -> ModelLoader:
    """Return the loaded singleton model instance."""
    return ModelLoader().load()
