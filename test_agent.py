import io
import unittest
from contextlib import redirect_stdout

from agent import AutoStreamAgent, detect_intent, retrieve_documents


class AutoStreamAgentTests(unittest.TestCase):
    def test_intent_detection_prioritizes_high_intent(self):
        self.assertEqual(
            detect_intent("That sounds good, I want to try the Pro plan."),
            "high_intent",
        )

    def test_retrieves_pricing_document(self):
        docs = retrieve_documents("What does the Pro plan cost?")
        self.assertEqual(docs[0]["id"], "pricing")

    def test_lead_capture_waits_for_all_fields(self):
        agent = AutoStreamAgent()
        captured_output = io.StringIO()

        with redirect_stdout(captured_output):
            first = agent.respond("I want to try the Pro plan for my YouTube channel.")
            second = agent.respond("Priya Sharma")
            third = agent.respond("priya@example.com")

        self.assertIn("name", first.lower())
        self.assertIn("email", second.lower())
        self.assertIn("Thanks Priya Sharma", third)
        self.assertIn(
            "Lead captured successfully: Priya Sharma, priya@example.com, YouTube",
            captured_output.getvalue(),
        )

    def test_invalid_email_does_not_capture_lead(self):
        agent = AutoStreamAgent()
        captured_output = io.StringIO()

        with redirect_stdout(captured_output):
            agent.respond("Sign me up for AutoStream.")
            agent.respond("Maya")
            response = agent.respond("not an email")

        self.assertIn("valid email", response.lower())
        self.assertNotIn("Lead captured successfully", captured_output.getvalue())


if __name__ == "__main__":
    unittest.main()
