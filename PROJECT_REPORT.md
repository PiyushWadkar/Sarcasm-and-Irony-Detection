# B.Tech Project Report & Viva Guide
# NLP-Based Sarcasm and Irony Detection in Text

---

## Executive Summary
This document serves as the academic project report and defense guide for the undergraduate final-year engineering project titled **"NLP-Based Sarcasm and Irony Detection in Text"**. It covers theoretical foundations, linguistic categorization, mathematical formulations, data pipelines, comparative machine learning algorithms, implementation details, experimental findings, and examiner viva voce questions.

---

## Table of Contents
1. [Introduction & Problem Formulation](#1-introduction--problem-formulation)
2. [Linguistic Taxonomy: Sarcasm vs. Irony](#2-linguistic-taxonomy-sarcasm-vs-irony)
3. [System Architecture & Data Flow](#3-system-architecture--data-flow)
4. [Dataset Benchmark: SemEval-2018 Task 3](#4-dataset-benchmark-semeval-2018-task-3)
5. [Preprocessing & Feature Engineering Pipeline](#5-preprocessing--feature-engineering-pipeline)
6. [Mathematical Foundations & Algorithms](#6-mathematical-foundations--algorithms)
7. [Explainability via Linear Feature Attribution](#7-explainability-via-linear-feature-attribution)
8. [Module-by-Module Code Walkthrough](#8-module-by-module-code-walkthrough)
9. [Experimental Results & Evaluation Analysis](#9-experimental-results--evaluation-analysis)
10. [Viva Voce Questions & Model Answers](#10-viva-voce-questions--model-answers)
11. [Conclusion & Future Work](#11-conclusion--future-work)

---

## 1. Introduction & Problem Formulation

### 1.1 Background
In the era of social media, online product reviews, and digital communication, Natural Language Processing (NLP) plays a critical role in automated sentiment analysis, brand monitoring, customer feedback ingestion, and content moderation.

However, conventional sentiment classification algorithms assume **literal compositionality**—that the sentiment of a sentence is a direct, monotonic function of its constituent words. Under this assumption:
- Words like *"great"*, *"awesome"*, *"loved"*, *"splendid"* are positive ($+1$).
- Words like *"awful"*, *"bad"*, *"worst"*, *"terrible"* are negative ($-1$).

### 1.2 The Figurative Language Challenge
When humans employ **sarcasm** or **irony**, they deliberately introduce a divergence between literal meaning (*what is said*) and figurative meaning (*what is intended*):
> *"Oh great, another flat tire on Monday morning! Just what I needed!"*

A literal sentiment analyzer flags *"great"* and *"needed"* as strong positive signals, incorrectly classifying a frustrated complaint as positive customer feedback.

### 1.3 Project Goal
To engineer a functional, transparent, and reproducible NLP application that accurately classifies sentences into:
1. **Sarcastic** (Verbal irony characterized by polarity contrast)
2. **Ironical** (Situational incongruity without direct polarity clash)
3. **Non-Sarcastic / Literal** (Straightforward communication)

---

## 2. Linguistic Taxonomy: Sarcasm vs. Irony

| Dimension | Sarcasm | Situational Irony | Literal Statement |
|:---|:---|:---|:---|
| **Linguistic Mechanism** | Polarity inversion (praise to censure) | Incongruity between expectation and outcome | Direct semantic correspondence |
| **Speaker Target** | Often targeted at a person, service, or event | Targeted at an absurd circumstance or reality | Informative / declarative |
| **Typical Example** | *"Brilliant driving, you only hit two trash cans!"* | *"A fire station caught fire yesterday."* | *"The meeting begins at 10 AM."* |
| **SemEval-2018 Class** | Subtask B: Class 1 | Subtask B: Class 2 & 3 | Subtask B: Class 0 |

---

## 3. System Architecture & Data Flow

### 3.1 Architectural Pipeline
```
[Raw Text / CSV Upload]
         │
         ▼
[1. Normalization & Cleaning]
  - HTML unescaping
  - Contraction expansion ("don't" -> "do not")
  - Hashtag segmentation ("#WorstDayEver" -> "Worst Day Ever")
  - Punctuation cluster preservation ("!!", "??")
         │
         ▼
[2. Irony-Aware Stopword Filtering]
  - Filters standard stop words
  - Explicitly PRESERVES negations: 'not', 'no', 'never', 'cannot'
  - Explicitly PRESERVES intensifiers: 'so', 'too', 'very', 'really'
         │
         ▼
[3. Feature Extraction: TF-IDF Vectorizer]
  - N-gram range: (1, 2) unigrams & bigrams
  - Sublinear TF scaling: 1 + log(TF)
  - Max features: 5,000
         │
         ▼
[4. Model Inference & Probability Calibration]
  - Champion: Multinomial Logistic Regression (Balanced class weights)
  - Baseline: Linear SVM + 3-Fold Platt Sigmoidal Calibration
         │
         ▼
[5. Explainability Engine]
  - Calculates token weight * TF-IDF contribution
  - Extracts top positive n-grams pushing classification
         │
         ▼
[6. SQLite Persistence & Web Dashboard]
  - REST API (/api/predict, /api/batch-predict, /api/stats, /api/history)
  - Interactive UI: Dark navy & purple design with Chart.js
```

---

## 4. Dataset Benchmark: SemEval-2018 Task 3

This project uses the gold-standard academic dataset from **SemEval-2018 Task 3: "Irony Detection in English Tweets" (Subtask B: Multi-class Irony Detection)** created by Van Hee et al. (Ghent University, ACL Anthology 2018).

### Data Split Statistics:
- **Training Set:** 3,817 samples
  - Literal: 1,916 (50.2%)
  - Sarcastic: 1,383 (36.2%)
  - Ironical: 518 (13.6%)
- **Held-Out Gold Test Set:** 784 samples
  - Literal: 473
  - Sarcastic: 164
  - Ironical: 147
- **Total Samples:** 4,601 verified samples

---

## 5. Preprocessing & Feature Engineering Pipeline

### 5.1 Why Naive Preprocessing Fails in Sarcasm Detection
Standard NLP tutorials blindly strip punctuation and remove all NLTK stopwords. In sarcasm detection, this causes severe data destruction:
1. **Negation Deletion:** Removing *"not"* turns *"not happy"* into *"happy"*, reversing the sentiment signal.
2. **Loss of Exclamations:** Multiple exclamation marks (`!!`) or question-exclamation pairs (`?!`) are hallmark signals of hyperbolic emotion.
3. **Hashtag Clumping:** Tweets often convey sarcasm in hashtags like `#BestDayEver`. Stripping the `#` without splitting camelCase leaves unrecognizable long strings.

### 5.2 Our Irony-Aware Preprocessing Strategy
1. **Contraction Expansion:** Normalizes 40+ English contractions (e.g., *can't* $\rightarrow$ *cannot*, *won't* $\rightarrow$ *will not*).
2. **Regex Hashtag Splitting:** `#TrafficJam` $\rightarrow$ `Traffic Jam`.
3. **Punctuation Normalization:** `!!!!` $\rightarrow$ `!!`, `???` $\rightarrow$ `??`.
4. **Negation-Preserving Filter:** Retains `not`, `no`, `nor`, `never`, `cannot`, `without`, `but`, `however`, `so`, `too`, `very`, `really`.

---

## 6. Mathematical Foundations & Algorithms

### 6.1 TF-IDF with Sublinear Scaling
The Term Frequency-Inverse Document Frequency weight for a token $t$ in document $d$ within corpus $D$ is given by:

$$\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \text{IDF}(t, D)$$

Where sublinear term frequency is:
$$\text{TF}(t, d) = 1 + \ln(f_{t,d}) \quad \text{for } f_{t,d} > 0$$

And inverse document frequency with smooth weighting:
$$\text{IDF}(t, D) = \ln\left(\frac{1 + |D|}{1 + |\{d \in D : t \in d\}|}\right) + 1$$

### 6.2 Multinomial Logistic Regression
Given feature vector $\mathbf{x} \in \mathbb{R}^V$, the probability that the text belongs to class $c \in \{0, 1, 2\}$ is:

$$P(Y = c \mid \mathbf{x}) = \frac{\exp(\mathbf{w}_c^T \mathbf{x} + b_c)}{\sum_{j=0}^{2} \exp(\mathbf{w}_j^T \mathbf{x} + b_j)}$$

Optimization is performed using the L-BFGS solver with balanced class weighting:
$$w_c = \frac{N}{K \cdot N_c}$$
Where $N$ is total samples, $K=3$ is number of classes, and $N_c$ is class sample count.

### 6.3 Platt Scaling for Linear SVM
A standard Linear Support Vector Machine maximizes the margin between classes by solving:

$$\min_{\mathbf{w}, b} \frac{1}{2} \|\mathbf{w}\|^2 + C \sum_{i=1}^n \max(0, 1 - y_i(\mathbf{w}^T \mathbf{x}_i + b))$$

The SVM output $f(\mathbf{x}) = \mathbf{w}^T \mathbf{x} + b$ is an uncalibrated signed geometric distance, not a probability. To calibrate it, **Platt Scaling** fits a logistic sigmoid over cross-validation decision values:

$$P(Y = 1 \mid f) = \frac{1}{1 + \exp(A \cdot f + B)}$$

Parameters $A$ and $B$ are estimated using maximum likelihood on out-of-fold validation splits.

---

## 7. Explainability via Linear Feature Attribution

For an input sentence with preprocessed tokens, each active n-gram has a TF-IDF value $x_i > 0$. The model's decision score for class $c$ is:

$$\text{score}_c = \sum_{i \in \text{tokens}} w_{c, i} \cdot x_i + b_c$$

We compute the contribution of each token $i$ to class $c$:
$$\text{attribution}(i, c) = w_{c, i} \cdot x_i$$

By sorting tokens by positive attribution, the system identifies the exact keywords and phrases (e.g., *"fantastic"*, *"delayed"*, *"just"*) driving the classification.

---

## 8. Module-by-Module Code Walkthrough

1. **`src/preprocessing.py`**:
   - `expand_contractions()`: Replaces abbreviated forms.
   - `split_hashtag()`: Splits camelCase hashtags.
   - `clean_text()`: Strips URLs, standardizes user mentions, unescapes HTML entities.
   - `preprocess_for_tfidf()`: Applies irony-aware stopword removal.
   - `validate_input_text()`: Validates string length and emptiness.

2. **`src/training.py`**:
   - `fetch_or_load_raw_data()`: Downloads SemEval-2018 Task B datasets.
   - `prepare_dataset()`: Maps labels to the 3 classes and cleans data.
   - `train_and_compare()`: Fits TF-IDF vectorizer, trains Logistic Regression and Calibrated Linear SVM, evaluates against gold test data, and saves champion artifacts.

3. **`src/prediction.py`**:
   - `load_artifacts()`: Caches model and vectorizer in memory.
   - `explain_prediction()`: Extracts positive n-gram coefficients.
   - `predict_single()`: Returns predicted class, probability breakdown, and explanation.
   - `predict_batch()`: Performs batch inference over arrays or CSV rows.

4. **`src/database.py`**:
   - SQLite table `predictions` with indexed fields for fast querying.
   - Functions `insert_prediction()`, `get_history()`, `get_statistics()`, `clear_history()`, and `export_all_predictions()`.

5. **`app.py`**:
   - Flask application exposing REST endpoints:
     - `GET /api/health`
     - `POST /api/predict`
     - `POST /api/batch-predict`
     - `GET /api/stats`
     - `GET /api/history`
     - `DELETE /api/history`
     - `GET /api/export-history`
     - `GET /api/evaluation`
   - Serves 5 server-rendered dashboard pages: `/`, `/batch`, `/history`, `/evaluation`, `/about`.

---

## 9. Experimental Results & Evaluation Analysis

### Comparative Test Set Evaluation (SemEval-2018 Task 3 Gold Test Set, N=784)

| Model Name | Test Accuracy | Macro Precision | Macro Recall | Macro F1-Score |
|:---|:---:|:---:|:---:|:---:|
| **Multinomial Logistic Regression** | **60.84%** | **56.10%** | **59.94%** | **57.31%** |
| **Linear SVM (Platt Calibrated)** | 62.12% | 45.45% | 48.20% | 45.96% |

### Why Logistic Regression was Selected as Champion:
Although Linear SVM achieved slightly higher top-line accuracy (62.12%), its macro recall on the minority *Ironical* class was substantially lower. Logistic Regression with balanced class weights produced **superior Macro F1-Score (57.31% vs 45.96%)** and balanced recall across all three classes:
- Literal Recall: **62.16%**
- Sarcastic Recall: **59.15%**
- Ironical Recall: **58.50%**

---

## 10. Viva Voce Questions & Model Answers

### Q1: What makes sarcasm detection significantly more difficult than traditional sentiment analysis?
**Answer:** Traditional sentiment analysis assumes semantic compositionality—that positive words reflect positive feelings. Sarcasm violates this through polarity inversion: the speaker expresses a negative attitude using ostensibly positive vocabulary (*"I just love waiting in 2-hour lines"*). Detecting this requires identifying contrast markers, hyperbolic patterns, and context.

### Q2: Why did you preserve certain stopwords instead of stripping all of them?
**Answer:** Standard stopword lists include negation words (`not`, `never`, `no`) and intensifiers (`very`, `so`, `too`). In sentiment and sarcasm analysis, negations act as polarity inverters (*"not good"* vs *"good"*). Removing them destroys the polarity contrast essential for sarcasm detection. We implemented an irony-aware stopword filter that explicitly preserves these critical tokens.

### Q3: What is the difference between Sarcasm and Irony in your system?
**Answer:** Sarcasm is a subtype of verbal irony characterized by an explicit polarity clash between surface words and speaker intent, usually intended to mock or express derision. Irony encompasses broader situational incongruities where an outcome is contrary to natural expectation (e.g., *"A fire station burning down"*), without an explicit polarity flip. We mapped SemEval-2018 Task 3 Subtask B classes accordingly.

### Q4: Why did you calibrate the Linear SVM using Platt Scaling?
**Answer:** Standard Linear SVM outputs a geometric distance $f(x) \in (-\infty, \infty)$ from the separating hyperplane. This value is not a calibrated posterior probability. Platt scaling fits a logistic sigmoid function over cross-validation decision values to transform raw SVM distances into statistically valid probabilities $P(y=c|x) \in [0, 1]$.

### Q5: How do you prevent data leakage during preprocessing and feature extraction?
**Answer:** The `TfidfVectorizer` is fitted exclusively on the training partition (`X_train`). The validation and test partitions are transformed using the already-fitted vectorizer (`transform()`). Furthermore, Platt calibration uses 3-fold internal cross-validation on the training set only, preventing test data leakage.

### Q6: How does the model explain its predictions?
**Answer:** For any input sentence, the system extracts the n-grams present in the text and computes their attribution score: $\text{attribution}_i = w_{c, i} \cdot x_i$, where $w_{c,i}$ is the model's learned coefficient for class $c$ and $x_i$ is the TF-IDF weight. The tokens with the highest positive attribution are returned to the user as key contributing features.

---

## 11. Conclusion & Future Work

### 11.1 Conclusion
The project successfully delivers a complete, college-level NLP system for sarcasm and irony detection. Moving beyond shallow UI prototypes, it implements a genuine machine learning pipeline trained on an established academic benchmark (SemEval-2018 Task 3), provides mathematically calibrated probabilities, offers token-level explainability, and presents a modern dark navy/purple web application.

### 11.2 Future Enhancements
1. **Transformer Fine-Tuning:** Fine-tune pretrained bidirectional models like RoBERTa or DeBERTa-v3.
2. **Context-Aware Dialogue Processing:** Ingest preceding turns in a dialogue to detect sarcasm rooted in conversation history.
3. **Multimodal Analysis:** Incorporate acoustic pitch, speech elongation, and facial expressions from video datasets like MUStARD.
