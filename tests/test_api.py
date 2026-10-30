"""
Integration and API Endpoint Tests for Flask Application.
"""

import unittest
import json
from app import app


class TestFlaskAPI(unittest.TestCase):

    def setUp(self):
        """Set up test client."""
        app.config["TESTING"] = True
        self.client = app.test_client()

    def test_health_endpoint(self):
        """Test GET /api/health."""
        resp = self.client.get("/api/health")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data["status"], "healthy")
        self.assertTrue(data["model_loaded"])

    def test_predict_endpoint_valid(self):
        """Test POST /api/predict with valid text."""
        payload = {"text": "Oh fantastic, another flat tire on Monday morning!"}
        resp = self.client.post(
            "/api/predict",
            data=json.dumps(payload),
            content_type="application/json"
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data["status"], "success")
        res = data["data"]
        self.assertIn(res["predicted_class"], ["Sarcastic", "Ironical", "Non-Sarcastic / Literal"])
        self.assertGreater(res["confidence"], 0.0)
        self.assertLessEqual(res["confidence"], 1.0)
        self.assertIn("probabilities", res)
        self.assertIn("explanation", res)

    def test_predict_endpoint_empty(self):
        """Test POST /api/predict with empty text returns 400."""
        payload = {"text": "    "}
        resp = self.client.post(
            "/api/predict",
            data=json.dumps(payload),
            content_type="application/json"
        )
        self.assertEqual(resp.status_code, 400)
        data = resp.get_json()
        self.assertEqual(data["status"], "error")

    def test_batch_predict_endpoint(self):
        """Test POST /api/batch-predict with multiple sentences."""
        payload = {
            "texts": [
                "Oh fantastic, my flight is delayed by another 4 hours!",
                "A fire station literally burned down yesterday.",
                "The meeting is scheduled for Thursday at 10 AM."
            ]
        }
        resp = self.client.post(
            "/api/batch-predict",
            data=json.dumps(payload),
            content_type="application/json"
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data["status"], "success")
        self.assertEqual(data["summary"]["total"], 3)
        self.assertEqual(len(data["results"]), 3)

    def test_stats_endpoint(self):
        """Test GET /api/stats."""
        resp = self.client.get("/api/stats")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data["status"], "success")
        self.assertIn("total_predictions", data["data"])

    def test_history_endpoint(self):
        """Test GET /api/history."""
        resp = self.client.get("/api/history?limit=10")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data["status"], "success")
        self.assertIsInstance(data["data"], list)

    def test_export_history_endpoint(self):
        """Test GET /api/export-history returns CSV."""
        resp = self.client.get("/api/export-history")
        self.assertEqual(resp.status_code, 200)
        self.assertIn("text/csv", resp.headers["Content-Type"])

    def test_dashboard_routes(self):
        """Test HTML dashboard page views."""
        for path in ["/", "/batch", "/history", "/evaluation", "/about"]:
            resp = self.client.get(path)
            self.assertEqual(resp.status_code, 200, f"Path {path} returned status {resp.status_code}")


if __name__ == "__main__":
    unittest.main()
