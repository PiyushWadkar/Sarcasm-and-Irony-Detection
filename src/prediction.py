"""
Inference and Explainability Module for Sarcasm & Irony Detection.

Provides model prediction, calibrated confidence calculations, and feature-weight
based explainability for single and batch text analyses.
"""

import os
import json
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import joblib

from src.preprocessing import clean_text, preprocess_for_tfidf, validate_input_text

# Default Models Directory
DEFAULT_MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")

# Global Cache
_CACHED_MODEL = None
_CACHED_VECTORIZER = None
_CACHED_METADATA = None


def load_artifacts(models_dir: str = DEFAULT_MODELS_DIR) -> Tuple[Any, Any, Dict[str, Any]]:
    """Load model, TF-IDF vectorizer, and metadata from disk (with in-memory caching)."""
    global _CACHED_MODEL, _CACHED_VECTORIZER, _CACHED_METADATA

    if _CACHED_MODEL is not None and _CACHED_VECTORIZER is not None:
        return _CACHED_MODEL, _CACHED_VECTORIZER, _CACHED_METADATA or {}

    model_path = os.path.join(models_dir, "best_model.joblib")
    vectorizer_path = os.path.join(models_dir, "tfidf_vectorizer.joblib")
    meta_path = os.path.join(models_dir, "model_metadata.json")

    if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
        raise FileNotFoundError(
            f"Trained model artifacts not found in '{models_dir}'. Please run 'python train_model.py' first."
        )

    _CACHED_MODEL = joblib.load(model_path)
    _CACHED_VECTORIZER = joblib.load(vectorizer_path)

    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            _CACHED_METADATA = json.load(f)
    else:
        _CACHED_METADATA = {}

    return _CACHED_MODEL, _CACHED_VECTORIZER, _CACHED_METADATA


def explain_prediction(
    cleaned_text: str,
    predicted_class: str,
    model: Any,
    vectorizer: Any,
    top_n: int = 5
) -> Tuple[List[Dict[str, Any]], str]:
    """
    Compute model-supported feature contributions for the predicted class.
    Calculates TF-IDF weight * model coefficient for tokens present in input text.
    """
    feature_names = vectorizer.get_feature_names_out()
    vocab = vectorizer.vocabulary_
    tokens = cleaned_text.split()

    contributions = []

    # Check if model has linear coefficients (LogisticRegression or calibrated base estimator)
    coef = None
    classes = getattr(model, "classes_", None)

    if hasattr(model, "coef_"):
        coef = model.coef_
    elif hasattr(model, "calibrated_classifiers_"):
        # For CalibratedClassifierCV with LinearSVC, average the base estimators' coefficients
        base_coefs = [clf.estimator.coef_ for clf in model.calibrated_classifiers_ if hasattr(clf.estimator, "coef_")]
        if base_coefs:
            coef = np.mean(base_coefs, axis=0)

    # Class index
    class_idx = None
    if classes is not None and predicted_class in classes:
        class_idx = list(classes).index(predicted_class)

    # Extract matching unigrams and bigrams
    all_ngrams = []
    for i in range(len(tokens)):
        all_ngrams.append(tokens[i])
        if i + 1 < len(tokens):
            all_ngrams.append(f"{tokens[i]} {tokens[i+1]}")

    for ng in set(all_ngrams):
        if ng in vocab:
            idx = vocab[ng]
            if coef is not None and class_idx is not None and class_idx < len(coef):
                weight = float(coef[class_idx][idx])
            else:
                # Fallback: TF-IDF value as importance proxy
                weight = 1.0

            contributions.append({
                "feature": ng,
                "weight": round(weight, 4),
                "is_positive": weight > 0
            })

    # Sort descending by weight
    contributions.sort(key=lambda x: x["weight"], reverse=True)
    top_features = contributions[:top_n]

    # Formulate natural language explanation
    top_words_str = ", ".join([f"'{f['feature']}'" for f in top_features if f['weight'] > 0])
    
    if predicted_class == "Sarcastic":
        if top_words_str:
            explanation = (
                f"The model detected sentiment contrast markers and verbal irony indicators ({top_words_str}) "
                f"that frequently accompany polarity clashes in the SemEval corpus."
            )
        else:
            explanation = "Sentence structure and vocabulary distribution correlate with sarcastic phrasing patterns."
    elif predicted_class == "Ironical":
        if top_words_str:
            explanation = (
                f"The text exhibits situational incongruity or figurative phrasing ({top_words_str}) "
                f"characteristic of non-polar irony."
            )
        else:
            explanation = "Phrasing matches figurative or situational irony patterns identified in training."
    else:  # Non-Sarcastic / Literal
        if top_words_str:
            explanation = (
                f"The text utilizes objective, direct vocabulary ({top_words_str}) "
                f"with an absence of strong figurative contradictions."
            )
        else:
            explanation = "The sentence exhibits standard literal diction without noticeable ironic markers."

    return top_features, explanation


def predict_single(text: str, models_dir: str = DEFAULT_MODELS_DIR) -> Dict[str, Any]:
    """
    Run full inference on a single text sentence.
    Returns predicted class, calibrated probability distribution, explanation, and top features.
    """
    # 1. Validation
    is_valid, error_msg = validate_input_text(text)
    if not is_valid:
        raise ValueError(error_msg)

    # 2. Load model and vectorizer
    model, vectorizer, metadata = load_artifacts(models_dir)

    # 3. Preprocess
    cleaned = preprocess_for_tfidf(text)
    if not cleaned.strip():
        # Edge case: text contained only stripped characters
        cleaned = clean_text(text)

    # 4. Transform
    X_vec = vectorizer.transform([cleaned])

    # 5. Predict & Probabilities
    classes = list(model.classes_)
    
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_vec)[0]
    elif hasattr(model, "decision_function"):
        # Calibrate decision scores with softmax
        df = model.decision_function(X_vec)[0]
        exp_df = np.exp(df - np.max(df))
        probs = exp_df / np.sum(exp_df)
    else:
        probs = [1.0 / len(classes)] * len(classes)

    max_idx = int(np.argmax(probs))
    predicted_class = classes[max_idx]
    confidence = float(probs[max_idx])

    # Probability dictionary mapped by class name
    prob_dict = {cls: round(float(probs[i]), 4) for i, cls in enumerate(classes)}

    # Ensure all target classes are present in dict
    for tc in ["Non-Sarcastic / Literal", "Sarcastic", "Ironical"]:
        if tc not in prob_dict:
            prob_dict[tc] = 0.0

    # 6. Explainability
    top_features, explanation = explain_prediction(
        cleaned_text=cleaned,
        predicted_class=predicted_class,
        model=model,
        vectorizer=vectorizer,
        top_n=5
    )

    return {
        "input_text": text,
        "cleaned_text": cleaned,
        "predicted_class": predicted_class,
        "confidence": round(confidence, 4),
        "confidence_percentage": round(confidence * 100, 1),
        "probabilities": prob_dict,
        "explanation": explanation,
        "top_features": top_features,
        "disclaimer": (
            "Model predictions reflect statistical n-gram patterns learned from SemEval-2018 "
            "and do not constitute guaranteed human emotional intent or context understanding."
        )
    }


def predict_batch(texts: List[str], models_dir: str = DEFAULT_MODELS_DIR) -> List[Dict[str, Any]]:
    """Batch prediction on a list of texts."""
    model, vectorizer, metadata = load_artifacts(models_dir)
    results = []

    for idx, raw_text in enumerate(texts):
        raw_text_str = str(raw_text).strip() if raw_text is not None else ""
        if not raw_text_str:
            continue

        try:
            pred = predict_single(raw_text_str, models_dir=models_dir)
            results.append({
                "row_index": idx + 1,
                "input_text": raw_text_str,
                "predicted_class": pred["predicted_class"],
                "confidence": pred["confidence"],
                "confidence_percentage": pred["confidence_percentage"],
                "explanation": pred["explanation"],
                "status": "success"
            })
        except Exception as e:
            results.append({
                "row_index": idx + 1,
                "input_text": raw_text_str,
                "predicted_class": "Error",
                "confidence": 0.0,
                "confidence_percentage": 0.0,
                "explanation": f"Processing error: {str(e)}",
                "status": "error"
            })

    return results
