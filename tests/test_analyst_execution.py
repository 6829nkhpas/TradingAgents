import unittest

from tradingagents.graph.analyst_execution import (
    build_analyst_execution_plan,
)


class AnalystExecutionPlanTests(unittest.TestCase):
    def test_build_plan_preserves_selected_order(self):
        plan = build_analyst_execution_plan(["news", "fundamentals"])

        self.assertEqual([spec.key for spec in plan.specs], ["news", "fundamentals"])
        self.assertEqual(plan.specs[0].agent_node, "News Analyst")
        self.assertEqual(plan.specs[0].report_key, "news_report")
        self.assertEqual(plan.specs[0].conviction_key, "news_conviction")

    def test_rejects_unknown_analyst_keys(self):
        with self.assertRaises(ValueError):
            build_analyst_execution_plan(["news", "macro"])

    def test_the_market_analyst_is_gone(self):
        with self.assertRaises(ValueError):
            build_analyst_execution_plan(["market"])

    def test_two_analysts_cannot_fill_one_section(self):
        with self.assertRaises(ValueError):
            build_analyst_execution_plan(["fundamentals", "crypto_fundamentals"])

    def test_social_key_displays_as_sentiment_analyst(self):
        # The selection key stays "social" for saved-config back-compat; the
        # section it fills is "sentiment".
        spec = build_analyst_execution_plan(["social"]).specs[0]
        self.assertEqual(spec.key, "social")
        self.assertEqual(spec.name, "sentiment")
        self.assertEqual(spec.agent_node, "Sentiment Analyst")
        self.assertEqual(spec.report_key, "sentiment_report")

    def test_crypto_fundamentals_fills_the_fundamentals_section(self):
        spec = build_analyst_execution_plan(["crypto_fundamentals"]).specs[0]
        self.assertEqual(spec.name, "fundamentals")
        self.assertEqual(spec.report_key, "fundamentals_report")
        self.assertEqual(spec.tools, ())
