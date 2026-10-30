"""
CLI Entrypoint for Dataset Preparation, Training, and Evaluation.

Run this script to reproduce the complete machine learning pipeline:
    python train_model.py
"""

import os
import sys
import argparse

# Ensure src package is importable
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.training import fetch_or_load_raw_data, prepare_dataset, train_and_compare


def main():
    parser = argparse.ArgumentParser(description="Train Sarcasm & Irony NLP Models")
    parser.add_argument("--data-dir", default=os.path.join(PROJECT_ROOT, "data"), help="Directory for dataset storage")
    parser.add_argument("--models-dir", default=os.path.join(PROJECT_ROOT, "models"), help="Directory to save trained models")
    parser.add_argument("--reports-dir", default=os.path.join(PROJECT_ROOT, "reports"), help="Directory to save evaluation reports")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    print("=" * 70)
    print(" NLP-BASED SARCASM AND IRONY DETECTION IN TEXT ")
    print(" Complete Dataset Preparation & Training Pipeline")
    print("=" * 70)

    try:
        # Step 1: Load Raw Dataset
        print("\n[STEP 1/4] Loading benchmark dataset (SemEval-2018 Task 3 Subtask B)...")
        train_raw, test_raw = fetch_or_load_raw_data(args.data_dir)
        print(f"  -> Raw training samples loaded: {len(train_raw)}")
        print(f"  -> Raw test samples loaded: {len(test_raw)}")

        # Step 2: Preprocess & Clean
        print("\n[STEP 2/4] Preprocessing texts and standardizing class labels...")
        processed_dir = os.path.join(args.data_dir, "processed")
        train_df, test_df = prepare_dataset(train_raw, test_raw, processed_dir)
        print(f"  -> Processed train records: {len(train_df)}")
        print(f"  -> Processed test records: {len(test_df)}")
        print("  -> Class distribution in train set:")
        for label, count in train_df["label"].value_counts().items():
            print(f"     * {label}: {count} ({count/len(train_df)*100:.1f}%)")

        # Step 3 & 4: Train, Evaluate, and Select Champion Model
        print("\n[STEP 3/4] Extracting TF-IDF features and training baseline models...")
        metadata = train_and_compare(
            train_df=train_df,
            test_df=test_df,
            models_dir=args.models_dir,
            reports_dir=args.reports_dir,
            random_seed=args.seed
        )

        print("\n[STEP 4/4] Final Results & Champion Model Selection:")
        champion = metadata["champion_model_name"]
        champ_metrics = metadata["champion_metrics"]
        print(f"  -> Best Performing Model: {champion}")
        print(f"  -> Test Accuracy:         {champ_metrics['accuracy']}%")
        print(f"  -> Macro Precision:       {champ_metrics['macro_precision']}%")
        print(f"  -> Macro Recall:          {champ_metrics['macro_recall']}%")
        print(f"  -> Macro F1-Score:        {champ_metrics['macro_f1']}%")
        print("\n  Per-Class Performance:")
        for cls, metrics in champ_metrics["per_class"].items():
            print(f"    [{cls}] Precision: {metrics['precision']}%, Recall: {metrics['recall']}%, F1: {metrics['f1_score']}% (N={metrics['support']})")

        print("\n" + "=" * 70)
        print(" TRAINING & EVALUATION COMPLETED SUCCESSFULLY! ")
        print(" Model artifacts ready for Flask inference.")
        print("=" * 70)

    except Exception as e:
        print(f"\n[ERROR] Training pipeline failed: {str(e)}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
