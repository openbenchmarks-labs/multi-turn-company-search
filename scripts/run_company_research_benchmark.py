#!/usr/bin/env python3
"""Plan or explicitly execute the file-backed company-search benchmark."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv

from company_research.agent import AgentConfig
from company_research.custom_fetch import validate_playwright_installation
from company_research.model_client import build_openai_client
from company_research.runner import RESEARCH_MODES, RunnerConfig, execute, runner_plan
from company_research.vendors import DEFAULT_VENDOR_KEYS, VENDORS, validate_vendor_keys


ROOT = Path(__file__).resolve().parents[1]


def _csv(raw: str) -> tuple[str, ...]:
    values = tuple(part.strip() for part in raw.split(",") if part.strip())
    validate_vendor_keys(values)
    return values


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--out-dir", type=Path, default=Path("runs"))
    parser.add_argument("--vendors", default=",".join(DEFAULT_VENDOR_KEYS))
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--vendor-concurrency", type=int, default=11)
    parser.add_argument("--trial-concurrency", type=int, default=3)
    parser.add_argument("--query-offset", type=int, default=0)
    parser.add_argument("--query-limit", type=int)
    parser.add_argument("--research-mode", choices=RESEARCH_MODES, default="search_and_fetch")
    parser.add_argument("--model", default="gpt-5.6-sol")
    parser.add_argument("--reasoning-effort", default="medium")
    parser.add_argument("--max-turns", type=int, default=8)
    parser.add_argument("--max-searches", type=int, default=14)
    parser.add_argument("--max-searches-per-turn", type=int, default=2)
    parser.add_argument("--max-results", type=int, default=10)
    parser.add_argument("--max-fetches", type=int, default=14)
    parser.add_argument("--max-fetches-per-turn", type=int, default=2)
    parser.add_argument("--max-output-tokens", type=int, default=6000)
    parser.add_argument("--model-input-usd-per-million", type=float, default=5.0)
    parser.add_argument("--model-cached-input-usd-per-million", type=float, default=0.5)
    parser.add_argument("--model-cache-write-usd-per-million", type=float, default=6.25)
    parser.add_argument("--model-output-usd-per-million", type=float, default=30.0)
    parser.add_argument("--retry-errors", action="store_true")
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm-paid", action="store_true")
    args = parser.parse_args()

    env_file = args.env_file or (ROOT / ".env.local")
    load_dotenv(env_file)
    if not args.env_file:
        load_dotenv(ROOT / ".env")

    vendors = _csv(args.vendors)
    agent = AgentConfig(
        model=args.model,
        reasoning_effort=args.reasoning_effort,
        max_turns=args.max_turns,
        max_searches=args.max_searches,
        max_results=args.max_results,
        max_searches_per_turn=args.max_searches_per_turn,
        max_fetches=0 if args.research_mode == "search_only" else args.max_fetches,
        max_fetches_per_turn=(
            None if args.research_mode == "search_only" else args.max_fetches_per_turn
        ),
        max_output_tokens=args.max_output_tokens,
        model_input_usd_per_million=args.model_input_usd_per_million,
        model_cached_input_usd_per_million=args.model_cached_input_usd_per_million,
        model_cache_write_usd_per_million=args.model_cache_write_usd_per_million,
        model_output_usd_per_million=args.model_output_usd_per_million,
    )
    config = RunnerConfig(
        dataset_path=args.dataset.resolve(),
        output_dir=args.out_dir.resolve(),
        vendor_keys=vendors,
        trial_count=args.trials,
        vendor_concurrency=args.vendor_concurrency,
        trial_concurrency=args.trial_concurrency,
        query_offset=args.query_offset,
        query_limit=args.query_limit,
        research_mode=args.research_mode,
        retry_errors=args.retry_errors,
        agent=agent,
    )
    plan = runner_plan(config)
    print(json.dumps(plan, indent=2))
    if not args.execute:
        return 0
    if not args.confirm_paid:
        parser.error("paid execution requires both --execute and --confirm-paid")

    missing = [name for name in plan["required_env_when_executed"] if not os.getenv(name, "").strip()]
    if missing:
        parser.error(f"missing environment variables: {', '.join(missing)}")
    if args.research_mode == "search_and_fetch" and any(VENDORS[key].custom_fetch for key in vendors):
        validate_playwright_installation()

    client = build_openai_client()
    result = execute(config, client)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
