"""
classification_service.py
==========================
ClassificationService for the ReviewGraph project.

Trains and serves a fake-review classifier.  The score returned by
``predict`` / ``predict_single`` is **not a calibrated probability** –
it is the raw decision-function / predict_proba score from the underlying
sklearn model and is deliberately labelled ``model_score`` to prevent
misinterpretation as a fraud probability.

Label convention (mirrors DatasetService):
    0 = genuine
    1 = suspicious / fake
"""

import json
import logging
import os
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from scipy.sparse import csr_matrix, hstack
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

_MODEL_FILENAME = "review_classifier.pkl"
_META_FILENAME = "classifier_meta.json"
# Alternate filenames tried during load (for backward compat)
_MODEL_FILENAME_FALLBACKS = ["classifier.joblib", "review_classifier.joblib"]

# Feature names for the behavioural block (must match extract_features order)
_BEHAVIOURAL_FEATURE_NAMES = [
    "review_length",
    "word_count",
    "avg_word_length",
    "rating_deviation_from_product_avg",
    "reviewer_review_count",
    "distinct_products_reviewed",
]


class ClassificationService:
    """
    Wraps TF-IDF + behavioural feature extraction and sklearn classifiers
    for fake-review detection.
    """

    def __init__(self, models_path: str) -> None:
        self.models_path = models_path
        os.makedirs(models_path, exist_ok=True)

        self.model = None          # best fitted sklearn estimator
        self.tfidf: Optional[TfidfVectorizer] = None
        self.metadata: Dict = {}
        self._feature_names: List[str] = []

        self.load_model()

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    def extract_features(
        self,
        df: pd.DataFrame,
        fit_tfidf: bool = False,
    ) -> Tuple[csr_matrix, List[str]]:
        """
        Build a sparse feature matrix from a (possibly preprocessed) dataframe.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain at least ``review_text``.  Behavioural columns
            (review_length, word_count, rating_deviation, etc.) are used
            when present; otherwise zeros are substituted.
        fit_tfidf : bool
            When True the internal TF-IDF vectoriser is fitted on the current
            texts (training phase).  When False the already-fitted vectoriser
            is used (inference phase).

        Returns
        -------
        X : scipy.sparse.csr_matrix
            Combined feature matrix (TF-IDF + behavioural).
        feature_names : list[str]
            Names matching the columns of X.
        """
        texts = df["review_text"].fillna("").astype(str).tolist()

        # ---- TF-IDF -----------------------------------------------------------
        if fit_tfidf or self.tfidf is None:
            self.tfidf = TfidfVectorizer(
                max_features=10_000,
                ngram_range=(1, 2),
                sublinear_tf=True,
                min_df=2,
                strip_accents="unicode",
            )
            X_tfidf = self.tfidf.fit_transform(texts)
            logger.info("TF-IDF fitted on %d documents.", len(texts))
        else:
            X_tfidf = self.tfidf.transform(texts)

        tfidf_names = [f"tfidf_{t}" for t in self.tfidf.get_feature_names_out()]

        # ---- Behavioural features --------------------------------------------
        def _safe_col(col: str, default: float = 0.0) -> np.ndarray:
            if col in df.columns:
                return df[col].fillna(default).astype(float).values
            return np.full(len(df), default, dtype=float)

        review_length   = _safe_col("review_length")
        word_count      = _safe_col("word_count")
        char_count      = _safe_col("char_count", default=review_length.mean() if len(review_length) else 0)

        avg_word_length = np.where(
            word_count > 0,
            review_length / np.maximum(word_count, 1),
            0.0,
        )

        rating_dev           = _safe_col("rating_deviation")
        reviewer_review_cnt  = _safe_col("reviewer_review_count")
        distinct_products    = _safe_col("reviewer_distinct_products")

        behavioural = np.column_stack([
            review_length,
            word_count,
            avg_word_length,
            rating_dev,
            reviewer_review_cnt,
            distinct_products,
        ])

        X_beh = csr_matrix(behavioural)
        X = hstack([X_tfidf, X_beh], format="csr")

        feature_names = tfidf_names + _BEHAVIOURAL_FEATURE_NAMES
        self._feature_names = feature_names
        return X, feature_names

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, df: pd.DataFrame, model_type: str = "logistic") -> Dict:
        """
        Train a fake-review classifier on *df*.

        Requires a ``label`` column that is **not** entirely NaN.

        Returns a metrics dict containing results for both LogisticRegression
        and RandomForest so the caller can compare.
        """
        if "label" not in df.columns or df["label"].isna().all():
            raise ValueError("DataFrame must contain a non-null 'label' column for training.")

        # Keep only labeled rows
        labeled = df[df["label"].notna()].copy()
        labeled["label"] = labeled["label"].astype(int)

        n_genuine    = int((labeled["label"] == 0).sum())
        n_suspicious = int((labeled["label"] == 1).sum())
        logger.info("Training on %d labeled rows (genuine=%d, suspicious=%d).",
                    len(labeled), n_genuine, n_suspicious)

        if len(labeled) < 10:
            raise ValueError("Need at least 10 labeled rows to train.")

        # ---- Group-aware split (by reviewer_id) -------------------------------
        if "reviewer_id" in labeled.columns:
            unique_reviewers = labeled["reviewer_id"].unique()
            rng = np.random.default_rng(seed=42)
            rng.shuffle(unique_reviewers)

            n = len(unique_reviewers)
            n_train_rev = int(n * 0.70)
            n_val_rev   = int(n * 0.15)

            train_revs = set(unique_reviewers[:n_train_rev])
            val_revs   = set(unique_reviewers[n_train_rev: n_train_rev + n_val_rev])
            # remainder → test

            train_mask = labeled["reviewer_id"].isin(train_revs)
            val_mask   = labeled["reviewer_id"].isin(val_revs)
            test_mask  = ~(train_mask | val_mask)

            train_df = labeled[train_mask]
            val_df   = labeled[val_mask]
            test_df  = labeled[test_mask]
        else:
            # Fallback to random split
            train_df, temp_df = train_test_split(labeled, test_size=0.30, random_state=42,
                                                  stratify=labeled["label"])
            val_df, test_df   = train_test_split(temp_df, test_size=0.50, random_state=42,
                                                   stratify=temp_df["label"])

        logger.info("Split sizes — train: %d, val: %d, test: %d",
                    len(train_df), len(val_df), len(test_df))

        # ---- Extract features (fit TF-IDF on training text only) --------------
        X_train, feat_names = self.extract_features(train_df, fit_tfidf=True)
        y_train = train_df["label"].values

        X_val, _ = self.extract_features(val_df)
        y_val    = val_df["label"].values

        X_test, _ = self.extract_features(test_df)
        y_test    = test_df["label"].values

        # ---- SMOTE if imbalanced (ratio > 3:1) --------------------------------
        n_maj = int(np.sum(y_train == 0))
        n_min = int(np.sum(y_train == 1))
        if n_min > 0 and n_maj / n_min > 3.0 and n_min >= 2:
            logger.info("Class imbalance detected (%.1f:1); applying SMOTE.", n_maj / n_min)
            try:
                sm = SMOTE(random_state=42, k_neighbors=min(5, n_min - 1))
                X_train, y_train = sm.fit_resample(X_train, y_train)
                logger.info("After SMOTE: %d training samples.", len(y_train))
            except Exception as exc:
                logger.warning("SMOTE failed (%s); continuing without oversampling.", exc)

        # ---- Train candidates --------------------------------------------------
        metrics: Dict = {}

        lr = LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            C=1.0,
            solver="lbfgs",
            random_state=42,
        )
        lr.fit(X_train, y_train)
        metrics["logistic"] = self._evaluate(lr, X_test, y_test, name="LogisticRegression")

        # RandomForest — use dense slice for speed on large matrices
        try:
            rf = RandomForestClassifier(
                n_estimators=200,
                class_weight="balanced",
                max_features="sqrt",
                random_state=42,
                n_jobs=-1,
            )
            rf.fit(X_train, y_train)
            metrics["random_forest"] = self._evaluate(rf, X_test, y_test, name="RandomForest")
        except Exception as exc:
            logger.warning("RandomForest training failed: %s", exc)
            metrics["random_forest"] = {"error": str(exc)}

        # ---- Choose best model by macro-F1 ------------------------------------
        lr_f1 = metrics["logistic"].get("f1_macro", 0.0)
        rf_f1 = metrics.get("random_forest", {}).get("f1_macro", -1.0)

        if rf_f1 > lr_f1 and model_type != "logistic":
            best_model = rf
            best_name  = "random_forest"
        else:
            best_model = lr
            best_name  = "logistic"

        self.model = best_model
        logger.info("Best model: %s (F1=%.4f).", best_name,
                    metrics[best_name].get("f1_macro", 0.0))

        # ---- Persist -----------------------------------------------------------
        meta = {
            "model_type":     best_name,
            "training_date":  datetime.now(timezone.utc).isoformat(),
            "model_version":  f"v1_{int(time.time())}",
            "n_train":        len(train_df),
            "n_val":          len(val_df),
            "n_test":         len(test_df),
            "feature_count":  len(feat_names),
            "metrics":        metrics,
            "label_map":      {"0": "genuine", "1": "suspicious"},
        }
        self.metadata = meta
        self._save_model(best_model, meta)

        return metrics

    # ------------------------------------------------------------------
    # Evaluation helper
    # ------------------------------------------------------------------

    def _evaluate(self, model, X_test, y_test, name: str = "") -> Dict:
        y_pred = model.predict(X_test)

        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, "decision_function"):
            raw = model.decision_function(X_test)
            y_score = 1 / (1 + np.exp(-raw))   # sigmoid squash for PR curve
        else:
            y_score = y_pred.astype(float)

        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        cm     = confusion_matrix(y_test, y_pred).tolist()

        try:
            pr_auc = float(average_precision_score(y_test, y_score))
        except Exception:
            pr_auc = float("nan")

        result = {
            "model_name":  name,
            "f1_macro":    round(report.get("macro avg", {}).get("f1-score", 0.0), 4),
            "f1_class_0":  round(report.get("0", {}).get("f1-score", 0.0), 4),
            "f1_class_1":  round(report.get("1", {}).get("f1-score", 0.0), 4),
            "precision_1": round(report.get("1", {}).get("precision", 0.0), 4),
            "recall_1":    round(report.get("1", {}).get("recall", 0.0), 4),
            "pr_auc":      round(pr_auc, 4),
            "confusion_matrix": cm,
            "n_test":      len(y_test),
        }
        logger.info("[%s] F1-macro=%.4f, PR-AUC=%.4f", name, result["f1_macro"], result["pr_auc"])
        return result

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save_model(self, model, meta: Dict) -> None:
        model_path = os.path.join(self.models_path, _MODEL_FILENAME)
        meta_path  = os.path.join(self.models_path, _META_FILENAME)

        # Bundle TF-IDF with the model so they are always in sync
        bundle = {"model": model, "tfidf": self.tfidf, "metadata": meta}
        joblib.dump(bundle, model_path, compress=3)

        with open(meta_path, "w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2, default=str)

        logger.info("Model saved to '%s'.", model_path)

    def load_model(self) -> None:
        """Load a saved model bundle or sklearn Pipeline from ``models_path``."""
        candidate_files = [_MODEL_FILENAME] + _MODEL_FILENAME_FALLBACKS
        model_path = None
        for cand in candidate_files:
            p = os.path.join(self.models_path, cand)
            if os.path.exists(p):
                model_path = p
                break

        if model_path is None:
            logger.info("No saved model found in '%s'.", self.models_path)
            return

        try:
            bundle = joblib.load(model_path)
            if isinstance(bundle, dict) and "model" in bundle:
                self.model    = bundle["model"]
                self.tfidf    = bundle.get("tfidf")
                self.metadata = bundle.get("metadata", {})
            else:
                # Standalone Pipeline or estimator
                self.model = bundle
                if hasattr(bundle, "named_steps") and "tfidf" in bundle.named_steps:
                    self.tfidf = bundle.named_steps["tfidf"]
                self.metadata = {"model_version": "v1.0-pipeline", "model_type": type(bundle).__name__}

            # Also try to load classifier_meta.json if present
            meta_path = os.path.join(self.models_path, _META_FILENAME)
            if os.path.exists(meta_path):
                try:
                    with open(meta_path, "r", encoding="utf-8") as fh:
                        self.metadata.update(json.load(fh))
                except Exception:
                    pass

            logger.info("Model loaded from '%s' (version=%s).", model_path, self.metadata.get("model_version", "unknown"))
        except Exception as exc:
            logger.exception("Failed to load model from '%s': %s", model_path, exc)
            self.model = None

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def _build_inference_matrix(
        self,
        texts: List[str],
        behavioral_features: Optional[np.ndarray],
    ) -> csr_matrix:
        """Combine TF-IDF and (optional) behavioural features for inference."""
        if self.tfidf is None:
            raise RuntimeError("TF-IDF vectoriser not initialised.  Train the model first.")

        X_tfidf = self.tfidf.transform(texts)

        if behavioral_features is not None:
            beh = np.atleast_2d(behavioral_features)
            # Pad / truncate to expected width
            expected = len(_BEHAVIOURAL_FEATURE_NAMES)
            if beh.shape[1] < expected:
                pad = np.zeros((beh.shape[0], expected - beh.shape[1]))
                beh = np.hstack([beh, pad])
            elif beh.shape[1] > expected:
                beh = beh[:, :expected]
            X_beh = csr_matrix(beh.astype(float))
        else:
            X_beh = csr_matrix(np.zeros((len(texts), len(_BEHAVIOURAL_FEATURE_NAMES))))

        return hstack([X_tfidf, X_beh], format="csr")

    def predict(
        self,
        texts: List[str],
        behavioral_features: Optional[np.ndarray] = None,
    ) -> Dict:
        """
        Predict labels for a list of review texts.

        Returns
        -------
        dict with keys:
            predicted_class  : List[int]   (0=genuine, 1=suspicious)
            model_score      : List[float] (raw model output, NOT a calibrated probability)
            model_version    : str
            warning          : str
        """
        if not self.is_ready():
            raise RuntimeError("Model is not loaded.  Call train() or load_model() first.")

        # If model is a pipeline that directly consumes raw text
        if hasattr(self.model, "named_steps") and "tfidf" in self.model.named_steps:
            raw_input = pd.Series(texts) if isinstance(texts, list) else texts
            predicted_class = [int(c) for c in self.model.predict(raw_input)]
            if hasattr(self.model, "predict_proba"):
                scores = [float(s) for s in self.model.predict_proba(raw_input)[:, 1]]
            else:
                scores = [float(c) for c in predicted_class]
        else:
            X = self._build_inference_matrix(texts, behavioral_features)
            predicted_class = [int(c) for c in self.model.predict(X)]

            if hasattr(self.model, "predict_proba"):
                scores = [float(s) for s in self.model.predict_proba(X)[:, 1]]
            elif hasattr(self.model, "decision_function"):
                raw    = self.model.decision_function(X)
                scores = [float(s) for s in (1 / (1 + np.exp(-raw)))]
            else:
                scores = [float(c) for c in predicted_class]

        return {
            "predicted_class": predicted_class,
            "model_score":     scores,
            "model_version":   self.metadata.get("model_version", "unknown"),
            "warning":         "model_score is the raw classifier output and is NOT a calibrated probability.",
        }

    def predict_single(
        self,
        review_text: str,
        behavioral_dict: Optional[Dict] = None,
    ) -> Dict:
        """
        Predict a single review.
        """
        beh_array: Optional[np.ndarray] = None
        if behavioral_dict:
            beh_array = np.array([[
                behavioral_dict.get("review_length", 0),
                behavioral_dict.get("word_count", 0),
                behavioral_dict.get("avg_word_length", 0),
                behavioral_dict.get("rating_deviation_from_product_avg", 0),
                behavioral_dict.get("reviewer_review_count", 0),
                behavioral_dict.get("distinct_products_reviewed", 0),
            ]], dtype=float)

        result = self.predict([review_text], beh_array)
        return {
            "predicted_class": result["predicted_class"][0],
            "model_score":     result["model_score"][0],
            "model_version":   result["model_version"],
            "warning":         result["warning"],
        }

    # ------------------------------------------------------------------
    # Status helpers
    # ------------------------------------------------------------------

    def is_ready(self) -> bool:
        """Return True if a trained model is loaded and ready for inference."""
        return self.model is not None and (self.tfidf is not None or hasattr(self.model, "predict"))

    def get_metadata(self) -> Dict:
        """Return model metadata (version, training date, metrics, feature names)."""
        meta = dict(self.metadata)
        meta["feature_names_sample"] = _BEHAVIOURAL_FEATURE_NAMES
        meta["tfidf_ready"] = self.tfidf is not None
        meta["model_ready"] = self.model is not None
        return meta
