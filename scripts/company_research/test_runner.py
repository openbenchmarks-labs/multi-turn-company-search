from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from company_research.agent import AgentConfig, AgentResult
from company_research.dataset import build_artifact
from company_research.runner import RunnerConfig, execute, runner_plan


def _dataset(path: Path) -> dict:
    payload = build_artifact(
        {
            "method": "test",
            "items": [
                {
                    "id": "q-1",
                    "question": "Find companies founded after 2020",
                    "family": "other",
                    "constraint_count": 1,
                    "clauses": ["founded after 2020"],
                    "answers": ["Acme"],
                    "answer_domains": {"Acme": "acme.example"},
                }
            ],
        },
        dataset_slug="test-dataset",
        dataset_name="Test dataset",
    )
    path.write_text(json.dumps(payload))
    return payload


class RunnerTest(unittest.TestCase):
    def test_plan_is_local_and_counts_runs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "dataset.json"
            _dataset(dataset)
            config = RunnerConfig(
                dataset_path=dataset,
                output_dir=root / "runs",
                vendor_keys=("brave",),
                trial_count=3,
                research_mode="search_only",
                agent=AgentConfig(max_fetches=0, max_fetches_per_turn=None),
            )
            plan = runner_plan(config)
            self.assertEqual(plan["planned_agent_runs"], 3)
            self.assertIn("local JSON only", plan["writes"])
            self.assertFalse((root / "runs").exists())

    def test_execute_writes_resumable_trial_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "dataset.json"
            _dataset(dataset)
            config = RunnerConfig(
                dataset_path=dataset,
                output_dir=root / "runs",
                vendor_keys=("brave",),
                trial_count=2,
                research_mode="search_only",
                agent=AgentConfig(max_fetches=0, max_fetches_per_turn=None),
            )
            response = json.dumps(
                {
                    "companies": [
                        {
                            "name": "Acme",
                            "domain": "acme.example",
                            "cited_urls": ["https://acme.example/about"],
                            "evidence": [
                                {"url": "https://acme.example/about", "claim": "Founded in 2021"}
                            ],
                        }
                    ]
                }
            )

            def fake_run(question, *, vendor_key, client, config):
                return AgentResult(question, vendor_key, response, "ok", None, 10)

            with patch("company_research.runner.run_agent", side_effect=fake_run) as run:
                execute(config, object())
                execute(config, object())
            self.assertEqual(run.call_count, 2)
            paths = sorted((root / "runs").rglob("trial-*.json"))
            self.assertEqual(len(paths), 2)
            artifact = json.loads(paths[0].read_text())
            self.assertEqual(artifact["dataset"]["slug"], "test-dataset")
            self.assertEqual(artifact["parsed_companies"][0]["name"], "Acme")

    def test_search_only_rejects_fetch_budget(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            dataset = Path(temporary) / "dataset.json"
            _dataset(dataset)
            with self.assertRaisesRegex(ValueError, "max_fetches=0"):
                runner_plan(
                    RunnerConfig(
                        dataset_path=dataset,
                        vendor_keys=("brave",),
                        research_mode="search_only",
                    )
                )


if __name__ == "__main__":
    unittest.main()
