from __future__ import annotations

import unittest

from company_research.judge import build_leaderboard, score_artifact


def _artifact(trial: int, *, correct: bool) -> dict:
    companies = (
        [
            {
                "name": "Acme",
                "domain": "https://www.acme.example/about",
                "cited_urls": ["https://acme.example/about"],
                "evidence": [],
            }
        ]
        if correct
        else []
    )
    return {
        "schema_version": "multi-turn-company-search-run-v1",
        "dataset": {"slug": "test", "content_sha256": "sha256:test"},
        "research_mode": "search_only",
        "vendor_key": "brave",
        "vendor": {"provider_name": "Brave Search", "config_slug": "brave"},
        "case": {
            "case_key": "q-1",
            "question": "Find Acme",
            "gold": [
                {
                    "entity_key": "entity:acme",
                    "name": "Acme, Inc.",
                    "domain": "acme.example",
                    "aliases": ["Acme"],
                }
            ],
        },
        "trial_index": trial,
        "result": {
            "status": "ok",
            "latency_ms": trial * 100,
            "model_turn_count": 2,
            "model_cost_usd": 0.01,
            "searches": [],
            "fetches": [],
        },
        "parsed_companies": companies,
        "parse_error": None,
    }


class JudgeTest(unittest.TestCase):
    def test_domain_and_alias_matching(self) -> None:
        score = score_artifact(_artifact(1, correct=True))
        self.assertEqual(score["true_positive"], 1)
        self.assertEqual(score["false_positive"], 0)
        self.assertEqual(score["f1"], 1.0)

    def test_mean_and_sample_sd_are_across_trial_means(self) -> None:
        result = build_leaderboard(
            [_artifact(1, correct=True), _artifact(2, correct=True), _artifact(3, correct=False)]
        )
        row = result["boards"]["search_only"][0]
        self.assertEqual(row["f1"], {"mean_pct": 66.67, "sd_pp": 57.74})
        self.assertEqual(row["rank"], 1)
        self.assertEqual(result["aggregation"]["standard_deviation_unit"], "percentage points")

    def test_incomplete_trials_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "trial indexes"):
            build_leaderboard([_artifact(1, correct=True)], expected_trials=3)


if __name__ == "__main__":
    unittest.main()
