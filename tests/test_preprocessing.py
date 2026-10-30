"""
Unit Tests for NLP Preprocessing Pipeline.
"""

import unittest
from src.preprocessing import (
    clean_text, split_hashtag, expand_contractions,
    preprocess_for_tfidf, validate_input_text
)


class TestPreprocessing(unittest.TestCase):

    def test_contraction_expansion(self):
        """Test expansion of common English contractions."""
        self.assertEqual(expand_contractions("I don't think she won't come"), "I do not think she will not come")
        self.assertEqual(expand_contractions("can't stop"), "cannot stop")
        self.assertEqual(expand_contractions("it's great"), "it is great")

    def test_hashtag_segmentation(self):
        """Test camelCase and standard hashtag splitting."""
        self.assertEqual(split_hashtag("#BestDayEver"), "Best Day Ever")
        self.assertEqual(split_hashtag("#flightDelayed"), "flight Delayed")
        self.assertEqual(split_hashtag("#irony"), "irony")

    def test_clean_text_removes_urls_and_mentions(self):
        """Test URL and user mention cleaning."""
        raw = "Check this out @elonmusk https://t.co/xyz123 &amp; enjoy!"
        cleaned = clean_text(raw)
        self.assertNotIn("https", cleaned)
        self.assertNotIn("t.co", cleaned)
        self.assertIn("@user", cleaned)
        self.assertIn("&", cleaned)

    def test_negation_preservation_in_tfidf_preprocessing(self):
        """Ensure negation and contrast words are NOT stripped as stopwords."""
        text = "This is not what I wanted, but thank you anyway!"
        processed = preprocess_for_tfidf(text)
        self.assertIn("not", processed.split())
        self.assertIn("but", processed.split())

    def test_input_validation(self):
        """Test validation rules for user input."""
        valid, msg = validate_input_text("Hello world")
        self.assertTrue(valid)
        self.assertEqual(msg, "")

        # Empty string
        valid, msg = validate_input_text("   ")
        self.assertFalse(valid)
        self.assertIn("empty", msg.lower())

        # Too short
        valid, msg = validate_input_text("a")
        self.assertFalse(valid)
        self.assertIn("too short", msg.lower())

        # Exceeds max length
        valid, msg = validate_input_text("a" * 2500, max_chars=2000)
        self.assertFalse(valid)
        self.assertIn("exceeds", msg.lower())


if __name__ == "__main__":
    unittest.main()
