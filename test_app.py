"""
SpamGuard AI - Standalone Application Unit & Integration Tests
Verifies local model loading, direct Python inference, and SQLite operations.
"""

import os
import unittest
import joblib
import database

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "spam_model.pkl")
TEST_DB_PATH = os.path.join(BASE_DIR, "test_spam_guard.db")


class TestSpamGuardStandalone(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # 1. Verify model file exists
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")
        cls.pipeline = joblib.load(MODEL_PATH)
        
        # 2. Setup isolated test database
        if os.path.exists(TEST_DB_PATH):
            os.remove(TEST_DB_PATH)
        database.init_db(TEST_DB_PATH)

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(TEST_DB_PATH):
            try:
                os.remove(TEST_DB_PATH)
            except Exception:
                pass

    def test_01_model_loading(self):
        """Test model pipeline object and classes."""
        self.assertIsNotNone(self.pipeline)
        self.assertTrue(hasattr(self.pipeline, 'predict'))
        self.assertTrue(hasattr(self.pipeline, 'predict_proba'))
        classes = list(self.pipeline.classes_)
        self.assertIn('ham', classes)
        self.assertIn('spam', classes)

    def test_02_spam_prediction(self):
        """Test direct spam inference on phishing message."""
        spam_msg = "WINNER!! You have won a £1000 prize or brand new car! Call 09061701461 to claim your reward immediately."
        pred = self.pipeline.predict([spam_msg])[0]
        probs = self.pipeline.predict_proba([spam_msg])[0]
        classes = list(self.pipeline.classes_)
        spam_idx = classes.index('spam')

        self.assertEqual(pred, 'spam')
        self.assertGreater(probs[spam_idx], 0.90)

    def test_03_ham_prediction(self):
        """Test direct ham inference on normal message."""
        ham_msg = "Hi Sarah, can you review the attached presentation slides before our sync meeting tomorrow at 10 AM? Thanks!"
        pred = self.pipeline.predict([ham_msg])[0]
        probs = self.pipeline.predict_proba([ham_msg])[0]
        classes = list(self.pipeline.classes_)
        ham_idx = classes.index('ham')

        self.assertEqual(pred, 'ham')
        self.assertGreater(probs[ham_idx], 0.85)

    def test_04_sanitize_preview(self):
        """Test message sanitization and truncation."""
        long_text = "<b>Urgent Notice:</b> " + ("A" * 200)
        preview = database.sanitize_preview(long_text, max_length=50)
        self.assertNotIn("<b>", preview)
        self.assertLessEqual(len(preview), 53)  # max_length + "..."
        self.assertTrue(preview.endswith("..."))

    def test_05_database_insert_and_history(self):
        """Test inserting predictions and querying history."""
        # Insert 1 spam and 1 ham
        id1 = database.insert_prediction("Spam lottery text", "spam", 0.99, db_path=TEST_DB_PATH)
        id2 = database.insert_prediction("Legitimate team meeting note", "ham", 0.98, db_path=TEST_DB_PATH)

        self.assertGreater(id1, 0)
        self.assertGreater(id2, id1)

        # Retrieve history
        items, total = database.get_history(db_path=TEST_DB_PATH)
        self.assertEqual(total, 2)
        self.assertEqual(len(items), 2)

        # Filter by spam
        spam_items, spam_total = database.get_history(filter_type='spam', db_path=TEST_DB_PATH)
        self.assertEqual(spam_total, 1)
        self.assertEqual(spam_items[0]['prediction'], 'spam')

        # Search
        searched, search_total = database.get_history(search_term='meeting', db_path=TEST_DB_PATH)
        self.assertEqual(search_total, 1)
        self.assertIn('meeting', searched[0]['message_preview'])

    def test_06_database_statistics(self):
        """Test database aggregate calculation."""
        stats = database.get_stats(db_path=TEST_DB_PATH)
        self.assertEqual(stats['total'], 2)
        self.assertEqual(stats['spam'], 1)
        self.assertEqual(stats['ham'], 1)
        self.assertEqual(stats['spam_percentage'], 50.0)
        self.assertEqual(stats['ham_percentage'], 50.0)

    def test_07_clear_history(self):
        """Test clearing history and resetting records."""
        database.clear_history(db_path=TEST_DB_PATH)
        stats = database.get_stats(db_path=TEST_DB_PATH)
        self.assertEqual(stats['total'], 0)
        self.assertEqual(stats['spam'], 0)
        self.assertEqual(stats['ham'], 0)


if __name__ == '__main__':
    unittest.main()
