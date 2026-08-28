from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from company_research.agent import AgentConfig, AgentResult
from company_research.dataset import build_artifact
from company_research.runner import RunnerConfig, execute, runner_plan
from company_research.vendors import (
    VENDORS,
    VendorCall,
    _parse_hits,
    parallel_site_policy,
    search,
)


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
    def test_parallel_modes_use_source_policy_for_site_queries(self) -> None:
        fixture = {"results": [{"url": "https://example.test", "title": "Example", "excerpts": ["Evidence"]}]}
        empty_call = VendorCall(
            status="ok", latency_ms=1, raw_request={}, raw_response=fixture,
            error=None, attempts=[], cost_usd=None,
        )
        query = "funding site:Crunchbase.com/org site:https://techcrunch.com/startups -site:reddit.com"
        self.assertEqual(
            parallel_site_policy(query),
            ("funding -site:reddit.com", ["crunchbase.com", "techcrunch.com"]),
        )
        for key, mode in (
            ("parallel_basic", "basic"),
            ("parallel_advanced", "advanced"),
            ("parallel_fast", "fast"),
            ("parallel_turbo", "turbo"),
        ):
            with self.subTest(key=key):
                spec = VENDORS[key]
                self.assertEqual(spec.request_config, {
                    "mode": mode,
                    "site_operator_policy": "source_policy",
                })
                self.assertEqual(
                    spec.search_unit_cost_usd,
                    0.001 if mode in {"fast", "turbo"} else 0.005,
                )
                self.assertTrue(spec.native_fetch)
                self.assertEqual(_parse_hits(key, fixture, 10)[0]["url"], "https://example.test")
                with (
                    patch.dict(os.environ, {"PARALLEL_API_KEY": "test-key"}),
                    patch("company_research.vendors._request", return_value=empty_call) as request,
                ):
                    search(key, query, max_results=7)
                    body = request.call_args.kwargs["body"]
                    self.assertEqual(body["mode"], mode)
                    self.assertEqual(body["objective"], "funding -site:reddit.com")
                    self.assertEqual(body["search_queries"], ["funding -site:reddit.com"])
                    self.assertEqual(body["advanced_settings"], {
                        "max_results": 7,
                        "source_policy": {
                            "include_domains": ["crunchbase.com", "techcrunch.com"],
                        },
                    })

    def test_parallel_query_without_site_filter_is_unchanged(self) -> None:
        empty_call = VendorCall(
            status="ok", latency_ms=1, raw_request={}, raw_response={"results": []},
            error=None, attempts=[], cost_usd=None,
        )
        with (
            patch.dict(os.environ, {"PARALLEL_API_KEY": "test-key"}),
            patch("company_research.vendors._request", return_value=empty_call) as request,
        ):
            search("parallel_fast", "ordinary company query")
        body = request.call_args.kwargs["body"]
        self.assertEqual(body["objective"], "ordinary company query")
        self.assertNotIn("source_policy", body["advanced_settings"])

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
