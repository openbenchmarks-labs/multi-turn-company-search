#!/usr/bin/env python3
"""Build leaderboard JSON from local company-search trial artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from company_research.judge import build_leaderboard, load_artifacts, write_leaderboard


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=Path("runs"))
    parser.add_argument("--output", type=Path, default=Path("leaderboard.json"))
    parser.add_argument("--trials", type=int, default=3)
    args = parser.parse_args()

    artifacts = load_artifacts(args.runs)
    leaderboard = build_leaderboard(artifacts, expected_trials=args.trials)
    write_leaderboard(leaderboard, args.output)
    print(json.dumps({"artifacts": len(artifacts), "output": str(args.output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
