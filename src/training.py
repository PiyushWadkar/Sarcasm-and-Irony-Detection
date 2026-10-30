"""
Model Training and Evaluation Pipeline.

Trains and evaluates Logistic Regression and Linear SVM with CalibratedClassifierCV
on the SemEval-2018 Task 3 benchmark dataset.
"""

import os
import json
import urllib.request
import io
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import joblib

from src.preprocessing import preprocess_for_tfidf

# Target Class Names
TARGET_CLASSES = ["Non-Sarcastic / Literal", "Sarcastic", "Ironical"]

LABEL_MAP = {
    0: "Non-Sarcastic / Literal",
    1: "Sarcastic",
    2: "Ironical",
    3: "Ironical"
}

RAW_TRAIN_URL = "https://raw.githubusercontent.com/Cyvhee/SemEval2018-Task3/master/datasets/train/SemEval2018-T3-train-taskB.txt"
RAW_TEST_URL = "https://raw.githubusercontent.com/Cyvhee/SemEval2018-Task3/master/datasets/goldtest_TaskB/SemEval2018-T3_gold_test_taskB_emoji.txt"


def fetch_or_load_raw_data(data_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load raw datasets from local disk, or download from official benchmark repo if absent.
    """
    raw_dir = os.path.join(data_dir, "raw")
    os.makedirs(raw_dir, exist_ok=True)

    train_file = os.path.join(raw_dir, "SemEval2018-T3-train-taskB.txt")
    test_file = os.path.join(raw_dir, "SemEval2018-T3-gold-test-taskB.txt")

    if not os.path.exists(train_file):
        print(f"[INFO] Downloading official SemEval-2018 Task B training dataset...")
        req = urllib.request.Request(RAW_TRAIN_URL, headers={"User-Agent": "Mozilla/5.0"})
        content = urllib.request.urlopen(req, timeout=30).read().decode("utf-8", errors="ignore")
        with open(train_file, "w", encoding="utf-8") as f:
            f.write(content)

    if not os.path.exists(test_file):
        print(f"[INFO] Downloading official SemEval-2018 Task B gold test dataset...")
        req = urllib.request.Request(RAW_TEST_URL, headers={"User-Agent": "Mozilla/5.0"})
        content = urllib.request.urlopen(req, timeout=30).read().decode("utf-8", errors="ignore")
        with open(test_file, "w", encoding="utf-8") as f:
            f.write(content)

    train_df = pd.read_csv(train_file, sep="\t", on_bad_lines="skip")
    test_df = pd.read_csv(test_file, sep="\t", on_bad_lines="skip")

    return train_df, test_df


def prepare_dataset(train_df: pd.DataFrame, test_df: pd.DataFrame, processed_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Map labels, preprocess text, drop invalid records, and save cleaned corpus.
    """
    os.makedirs(processed_dir, exist_ok=True)

    def standardize_cols(df: pd.DataFrame) -> pd.DataFrame:
        col_rename = {}
        for c in df.columns:
            if "text" in c.lower():
                col_rename[c] = "text"
            elif "label" in c.lower():
                col_rename[c] = "raw_label"
        return df.rename(columns=col_rename)

    train_df = standardize_cols(train_df)
    test_df = standardize_cols(test_df)

    for df in [train_df, test_df]:
        df.dropna(subset=["text", "raw_label"], inplace=True)
        # Convert numeric label
        df["raw_label"] = pd.to_numeric(df["raw_label"], errors="coerce")
        df.dropna(subset=["raw_label"], inplace=True)
        df["raw_label"] = df["raw_label"].astype(int)
        df["label"] = df["raw_label"].map(LABEL_MAP)
        df.dropna(subset=["label"], inplace=True)

    # Remove duplicates within each split
    train_df = train_df.drop_duplicates(subset=["text"]).copy()
    test_df = test_df.drop_duplicates(subset=["text"]).copy()

    # Preprocess text
    print("[INFO] Preprocessing training texts...")
    train_df["cleaned_text"] = train_df["text"].apply(preprocess_for_tfidf)
    print("[INFO] Preprocessing testing texts...")
    test_df["cleaned_text"] = test_df["text"].apply(preprocess_for_tfidf)

    # Filter out empty preprocessed strings
    train_df = train_df[train_df["cleaned_text"].str.strip().str.len() > 0].copy()
    test_df = test_df[test_df["cleaned_text"].str.strip().str.len() > 0].copy()

    # Save processed datasets
    train_df.to_csv(os.path.join(processed_dir, "train_cleaned.csv"), index=False)
    test_df.to_csv(os.path.join(processed_dir, "test_cleaned.csv"), index=False)
    
    combined = pd.concat([
        train_df.assign(split="train"),
        test_df.assign(split="test")
    ])
    combined.to_csv(os.path.join(processed_dir, "cleaned_corpus.csv"), index=False)

    return train_df, test_df


def evaluate_model(model: Any, X_test: Any, y_test: Any, class_names: list) -> Dict[str, Any]:
    """Calculate comprehensive evaluation metrics and confusion matrix."""
    y_pred = model.predict(X_test)

    acc = float(accuracy_score(y_test, y_pred))
    macro_prec = float(precision_score(y_test, y_pred, labels=class_names, average="macro", zero_division=0))
    macro_rec = float(recall_score(y_test, y_pred, labels=class_names, average="macro", zero_division=0))
    macro_f1 = float(f1_score(y_test, y_pred, labels=class_names, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_test, y_pred, labels=class_names, average="weighted", zero_division=0))

    cm = confusion_matrix(y_test, y_pred, labels=class_names)
    report_dict = classification_report(y_test, y_pred, labels=class_names, output_dict=True, zero_division=0)

    # Format per-class metrics
    per_class = {}
    for cls in class_names:
        if cls in report_dict:
            per_class[cls] = {
                "precision": round(report_dict[cls]["precision"] * 100, 2),
                "recall": round(report_dict[cls]["recall"] * 100, 2),
                "f1_score": round(report_dict[cls]["f1-score"] * 100, 2),
                "support": int(report_dict[cls]["support"])
            }

    return {
        "accuracy": round(acc * 100, 2),
        "macro_precision": round(macro_prec * 100, 2),
        "macro_recall": round(macro_rec * 100, 2),
        "macro_f1": round(macro_f1 * 100, 2),
        "weighted_f1": round(weighted_f1 * 100, 2),
        "per_class": per_class,
        "confusion_matrix": {
            "labels": class_names,
            "matrix": cm.tolist()
        }
    }


def train_and_compare(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    models_dir: str,
    reports_dir: str,
    random_seed: int = 42
) -> Dict[str, Any]:
    """
    Train Logistic Regression and Linear SVM with Calibrated probabilities,
    compare metrics, and save best model.
    """
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    # 1. Feature Extraction: TF-IDF Vectorizer
    print("[INFO] Fitting TF-IDF Vectorizer (unigrams + bigrams, max_features=5000)...")
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=2,
        token_pattern=r"(?u)\b\w+\b|[!?]{1,2}"
    )

    X_train = vectorizer.fit_transform(train_df["cleaned_text"])
    y_train = train_df["label"].values

    X_test = vectorizer.transform(test_df["cleaned_text"])
    y_test = test_df["label"].values

    # 2. Model 1: Multinomial Logistic Regression
    print("[INFO] Training Model 1: Multinomial Logistic Regression...")
    lr_model = LogisticRegression(
        C=1.5,
        max_iter=1000,
        class_weight="balanced",
        random_state=random_seed,
        solver="lbfgs"
    )
    lr_model.fit(X_train, y_train)
    lr_metrics = evaluate_model(lr_model, X_test, y_test, TARGET_CLASSES)

    # 3. Model 2: Linear SVM with CalibratedClassifierCV
    print("[INFO] Training Model 2: Linear SVM (Calibrated for reliable probabilities)...")
    base_svm = LinearSVC(
        C=0.5,
        class_weight="balanced",
        random_state=random_seed,
        dual="auto"
    )
    # CalibratedClassifierCV uses 3-fold cross-validation on train set to fit Platt sigmoid calibrators
    svm_calibrated = CalibratedClassifierCV(estimator=base_svm, cv=3)
    svm_calibrated.fit(X_train, y_train)
    svm_metrics = evaluate_model(svm_calibrated, X_test, y_test, TARGET_CLASSES)

    print("\n--- Model Evaluation Summary ---")
    print(f"Logistic Regression -> Acc: {lr_metrics['accuracy']}%, Macro F1: {lr_metrics['macro_f1']}%")
    print(f"Linear SVM (Calibrated) -> Acc: {svm_metrics['accuracy']}%, Macro F1: {svm_metrics['macro_f1']}%")

    # Select best model based on Macro F1-score
    if svm_metrics["macro_f1"] >= lr_metrics["macro_f1"]:
        best_name = "Linear SVM (Calibrated)"
        best_model = svm_calibrated
        best_metrics = svm_metrics
    else:
        best_name = "Logistic Regression"
        best_model = lr_model
        best_metrics = lr_metrics

    print(f"[INFO] Champion Model selected: {best_name}")

    # Save artifacts
    best_model_path = os.path.join(models_dir, "best_model.joblib")
    vectorizer_path = os.path.join(models_dir, "tfidf_vectorizer.joblib")
    lr_model_path = os.path.join(models_dir, "logistic_regression.joblib")
    svm_model_path = os.path.join(models_dir, "linear_svm_calibrated.joblib")

    joblib.dump(best_model, best_model_path)
    joblib.dump(vectorizer, vectorizer_path)
    joblib.dump(lr_model, lr_model_path)
    joblib.dump(svm_calibrated, svm_model_path)

    # Metadata & summary
    metadata = {
        "benchmark": "SemEval-2018 Task 3 Subtask B",
        "champion_model_name": best_name,
        "classes": TARGET_CLASSES,
        "train_samples": len(train_df),
        "test_samples": len(test_df),
        "vocabulary_size": len(vectorizer.vocabulary_),
        "random_seed": random_seed,
        "models_comparison": {
            "logistic_regression": {
                "name": "Multinomial Logistic Regression",
                "metrics": lr_metrics
            },
            "linear_svm": {
                "name": "Linear SVM (Platt Calibrated)",
                "metrics": svm_metrics
            }
        },
        "champion_metrics": best_metrics
    }

    metadata_path = os.path.join(models_dir, "model_metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    report_path = os.path.join(reports_dir, "evaluation_summary.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"[SUCCESS] Best model saved to {best_model_path}")
    print(f"[SUCCESS] Vectorizer saved to {vectorizer_path}")
    print(f"[SUCCESS] Evaluation summary saved to {report_path}")

    return metadata
