import io
import unittest
from contextlib import redirect_stdout

from agent import AutoStreamAgent, detect_intent, retrieve_documents


class AutoStreamAgentTests(unittest.TestCase):
    def test_greeting_with_question_returns_answer_not_only_greeting(self):
        agent = AutoStreamAgent()
        response = agent.respond("Hi, tell me about your pricing.")
        self.assertIn("$29/month", response)
        self.assertIn("$79/month", response)

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

    def test_high_intent_message_can_answer_question_and_collect_lead(self):
        agent = AutoStreamAgent()
        response = agent.respond(
            "Hi, your Pro plan sounds good. I want to try it for my YouTube channel."
        )
        self.assertIn("$79/month", response)
        self.assertIn("name", response.lower())

    def test_agent_extracts_multiple_lead_fields_from_single_message(self):
        agent = AutoStreamAgent()
        captured_output = io.StringIO()

        with redirect_stdout(captured_output):
            first = agent.respond("I want to get started.")
            second = agent.respond("My name is Maya and my email is maya@example.com")
            third = agent.respond("Instagram")

        self.assertIn("name", first.lower())
        self.assertIn("platform", second.lower())
        self.assertIn("maya@example.com", third)
        self.assertIn("Lead captured successfully", captured_output.getvalue())

    def test_invalid_email_does_not_capture_lead(self):
        agent = AutoStreamAgent()
        captured_output = io.StringIO()

        with redirect_stdout(captured_output):
            agent.respond("Sign me up for AutoStream.")
            agent.respond("Maya")
            response = agent.respond("not an email")

        self.assertIn("valid email", response.lower())
        self.assertNotIn("Lead captured successfully", captured_output.getvalue())

    def test_high_intent_phrase_is_not_used_as_name(self):
        agent = AutoStreamAgent()
        captured_output = io.StringIO()

        with redirect_stdout(captured_output):
            first = agent.respond("ready to start")
            second = agent.respond("msr")
            third = agent.respond("Msr@gmail.com")
            fourth = agent.respond("insta")

        self.assertIn("name", first.lower())
        self.assertIn("email", second.lower())
        self.assertIn("platform", third.lower())
        self.assertIn("Lead captured successfully: Msr, Msr@gmail.com, insta", captured_output.getvalue())
        self.assertNotIn("ready to start", fourth.lower())


if __name__ == "__main__":
    unittest.main()
