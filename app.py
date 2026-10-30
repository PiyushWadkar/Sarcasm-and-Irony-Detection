"""
Flask Application for NLP-Based Sarcasm and Irony Detection.

Exposes REST APIs and interactive server-rendered web dashboard.
"""

import os
import sys
import io
import csv
import json
from datetime import datetime
from flask import Flask, request, jsonify, render_template, send_file, Response

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.preprocessing import validate_input_text
from src.prediction import predict_single, predict_batch, load_artifacts
from src.database import (
    init_db, insert_prediction, get_history, get_statistics,
    clear_history as db_clear_history, export_all_predictions
)

app = Flask(__name__)
app.config["SECRET_KEY"] = "sarcasm-irony-detection-secure-key-2026"
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024  # 8 MB max upload limit

# Initialize SQLite tables on app startup
init_db()

# Pre-load ML artifacts
try:
    load_artifacts()
    print("[INFO] Model and vectorizer successfully pre-loaded.")
except Exception as e:
    print(f"[WARNING] Model artifacts could not be pre-loaded: {e}")


# ==========================================
# PAGE ROUTES (WEB DASHBOARD)
# ==========================================

@app.route("/")
def index():
    """Render Main Interactive Dashboard."""
    stats = get_statistics()
    return render_template("index.html", stats=stats, active_page="dashboard")


@app.route("/batch")
def batch_page():
    """Render Batch Text and CSV Analysis Page."""
    return render_template("batch.html", active_page="batch")


@app.route("/history")
def history_page():
    """Render Prediction History Page."""
    return render_template("history.html", active_page="history")


@app.route("/evaluation")
def evaluation_page():
    """Render Model Evaluation and Benchmarking Page."""
    eval_file = os.path.join(PROJECT_ROOT, "reports", "evaluation_summary.json")
    eval_data = {}
    if os.path.exists(eval_file):
        with open(eval_file, "r", encoding="utf-8") as f:
            eval_data = json.load(f)
    return render_template("evaluation.html", eval_data=eval_data, active_page="evaluation")


@app.route("/about")
def about_page():
    """Render Project Architecture, Theory, and Viva Guide Page."""
    return render_template("about.html", active_page="about")


# ==========================================
# REST API ENDPOINTS
# ==========================================

@app.route("/api/health", methods=["GET"])
def health_check():
    """Check API and Model Status."""
    meta_path = os.path.join(PROJECT_ROOT, "models", "model_metadata.json")
    model_ready = False
    metadata = {}
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            model_ready = True
        except Exception:
            pass

    return jsonify({
        "status": "healthy" if model_ready else "degraded",
        "model_loaded": model_ready,
        "champion_model": metadata.get("champion_model_name", "None"),
        "benchmark": metadata.get("benchmark", "SemEval-2018 Task 3"),
        "timestamp": datetime.utcnow().isoformat()
    }), 200


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """Analyze a single text statement."""
    if not request.is_json:
        return jsonify({"status": "error", "message": "Request body must be valid JSON."}), 400

    data = request.get_json()
    text = data.get("text", "")

    # Input validation
    is_valid, err_msg = validate_input_text(text)
    if not is_valid:
        return jsonify({"status": "error", "message": err_msg}), 400

    try:
        result = predict_single(text)

        # Record prediction into SQLite
        insert_prediction(
            input_text=text,
            predicted_class=result["predicted_class"],
            confidence=result["confidence"],
            probabilities=result["probabilities"],
            explanation=result["explanation"],
            top_features=result["top_features"],
            source="single"
        )

        return jsonify({
            "status": "success",
            "data": result
        }), 200

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Inference failed: {str(e)}"
        }), 500


@app.route("/api/batch-predict", methods=["POST"])
def api_batch_predict():
    """Analyze multiple texts via JSON array or CSV file upload."""
    texts = []

    # Handle file upload
    if "file" in request.files:
        file = request.files["file"]
        if file.filename == "":
            return jsonify({"status": "error", "message": "No file selected."}), 400
        
        if not (file.filename.endswith(".csv") or file.filename.endswith(".txt")):
            return jsonify({"status": "error", "message": "Only .csv and .txt files are supported."}), 400

        try:
            stream = io.StringIO(file.stream.read().decode("utf-8", errors="ignore"))
            reader = csv.reader(stream)
            rows = list(reader)
            
            if not rows:
                return jsonify({"status": "error", "message": "Uploaded file is empty."}), 400

            # Detect if header exists
            first_row = rows[0]
            start_idx = 0
            text_col_idx = 0

            # Check for header names
            lower_headers = [c.lower() for c in first_row]
            for candidate in ["text", "sentence", "tweet", "headline", "content"]:
                if candidate in lower_headers:
                    text_col_idx = lower_headers.index(candidate)
                    start_idx = 1
                    break

            for row in rows[start_idx:]:
                if row and len(row) > text_col_idx:
                    val = row[text_col_idx].strip()
                    if val:
                        texts.append(val)

        except Exception as e:
            return jsonify({"status": "error", "message": f"Error parsing uploaded file: {str(e)}"}), 400

    # Handle JSON array
    elif request.is_json:
        data = request.get_json()
        raw_texts = data.get("texts", [])
        if isinstance(raw_texts, list):
            texts = [str(t).strip() for t in raw_texts if str(t).strip()]
        elif isinstance(raw_texts, str):
            # Split lines
            texts = [line.strip() for line in raw_texts.splitlines() if line.strip()]

    if not texts:
        return jsonify({
            "status": "error",
            "message": "No valid text entries provided for batch analysis."
        }), 400

    # Safety limit
    MAX_BATCH_SIZE = 500
    if len(texts) > MAX_BATCH_SIZE:
        texts = texts[:MAX_BATCH_SIZE]

    try:
        results = predict_batch(texts)

        # Batch record insertion into SQLite
        for r in results:
            if r["status"] == "success":
                insert_prediction(
                    input_text=r["input_text"],
                    predicted_class=r["predicted_class"],
                    confidence=r["confidence"],
                    probabilities={},
                    explanation=r["explanation"],
                    top_features=[],
                    source="batch"
                )

        # Aggregate counts
        summary = {
            "total": len(results),
            "sarcastic": sum(1 for r in results if r["predicted_class"] == "Sarcastic"),
            "ironical": sum(1 for r in results if r["predicted_class"] == "Ironical"),
            "literal": sum(1 for r in results if r["predicted_class"] in ["Non-Sarcastic / Literal", "Literal"]),
            "errors": sum(1 for r in results if r["status"] == "error")
        }

        return jsonify({
            "status": "success",
            "summary": summary,
            "results": results
        }), 200

    except Exception as e:
        return jsonify({"status": "error", "message": f"Batch inference failed: {str(e)}"}), 500


@app.route("/api/stats", methods=["GET"])
def api_stats():
    """Retrieve current dashboard prediction statistics."""
    stats = get_statistics()
    return jsonify({"status": "success", "data": stats}), 200


@app.route("/api/history", methods=["GET"])
def api_history():
    """Fetch paginated, filtered prediction history."""
    limit = request.args.get("limit", default=50, type=int)
    offset = request.args.get("offset", default=0, type=int)
    search = request.args.get("search", default=None, type=str)
    class_filter = request.args.get("class", default=None, type=str)

    # Sanitize limit
    limit = min(max(limit, 5), 200)

    items, total = get_history(limit=limit, offset=offset, search_query=search, class_filter=class_filter)

    return jsonify({
        "status": "success",
        "total": total,
        "limit": limit,
        "offset": offset,
        "data": items
    }), 200


@app.route("/api/history", methods=["DELETE"])
def api_clear_history():
    """Clear all records from prediction history."""
    deleted_count = db_clear_history()
    return jsonify({
        "status": "success",
        "message": f"Successfully deleted {deleted_count} prediction records.",
        "deleted_count": deleted_count
    }), 200


@app.route("/api/export-history", methods=["GET"])
def export_history_csv():
    """Export all prediction records as a downloadable CSV."""
    records = export_all_predictions()

    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=["id", "input_text", "predicted_class", "confidence", "source", "created_at"])
    writer.writeheader()
    writer.writerows(records)

    csv_data = output.getvalue()
    output.close()

    filename = f"sarcasm_irony_predictions_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.csv"
    return Response(
        csv_data,
        mimetype="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )


@app.route("/api/evaluation", methods=["GET"])
def api_evaluation():
    """Return model evaluation metrics, reports, and confusion matrices."""
    report_path = os.path.join(PROJECT_ROOT, "reports", "evaluation_summary.json")
    if not os.path.exists(report_path):
        return jsonify({"status": "error", "message": "Evaluation report not found."}), 404

    with open(report_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return jsonify({"status": "success", "data": data}), 200


# ==========================================
# APPLICATION ENTRYPOINT
# ==========================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"\n[INFO] Starting Sarcasm & Irony Detection Web Application on http://127.0.0.1:{port}")
    app.run(host="127.0.0.1", port=port, debug=True)
