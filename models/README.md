# Models Directory

This directory stores trained model binaries, vectorizers, and metadata.

## Artifacts
- **`best_model.joblib`**: The champion model selected based on the highest Macro F1-score across validation (Multinomial Logistic Regression with balanced class weights).
- **`tfidf_vectorizer.joblib`**: The fitted `TfidfVectorizer` configured with 5,000 unigram and bigram features, sublinear term frequency scaling, and irony-aware token extraction.
- **`logistic_regression.joblib`**: Trained baseline multinomial logistic regression model.
- **`linear_svm_calibrated.joblib`**: Trained Linear Support Vector Classifier wrapped with `CalibratedClassifierCV` (Platt Sigmoidal scaling via 3-Fold cross-validation).
- **`model_metadata.json`**: Machine-readable performance metrics, class labels, hyperparameters, and confusion matrix data used by the web UI.
