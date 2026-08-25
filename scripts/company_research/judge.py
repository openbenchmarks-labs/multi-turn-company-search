"""Deterministic, file-backed judge for saved company-search run artifacts."""

from __future__ import annotations

import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlparse

from .dataset import normalize_company_name


QUALITY_METRICS = ("precision", "recall", "f1", "exact_set_accuracy")


def _domain(value: Any) -> str:
    raw = str(value or "").strip().casefold()
    if not raw:
        return ""
    parsed = urlparse(raw if "://" in raw else f"https://{raw}")
    return (parsed.hostname or "").removeprefix("www.").rstrip(".")


def _gold_lookup(gold: list[dict[str, Any]]) -> tuple[set[str], dict[str, str]]:
    identities: set[str] = set()
    lookup: dict[str, str] = {}
    for member in gold:
        identity = str(member["entity_key"])
        identities.add(identity)
        domain = _domain(member.get("domain"))
        if domain:
            lookup[f"domain:{domain}"] = identity
        for name in [member.get("name"), *(member.get("aliases") or [])]:
            normalized = normalize_company_name(str(name or ""))
            if normalized:
                lookup[f"name:{normalized}"] = identity
    return identities, lookup


def score_artifact(artifact: dict[str, Any]) -> dict[str, Any]:
    """Score one immutable trial artifact without model or network calls."""
    gold, lookup = _gold_lookup(artifact["case"]["gold"])
    predicted: set[str] = set()
    supported = 0
    if artifact.get("result", {}).get("status") == "ok" and not artifact.get("parse_error"):
        for index, company in enumerate(artifact.get("parsed_companies") or []):
            domain_key = f"domain:{_domain(company.get('domain'))}"
            name_key = f"name:{normalize_company_name(str(company.get('name') or ''))}"
            identity = lookup.get(domain_key) if _domain(company.get("domain")) else None
            identity = identity or lookup.get(name_key) or f"prediction:{index}:{name_key}"
            if identity not in predicted and company.get("cited_urls"):
                supported += 1
            predicted.add(identity)

    true_positive = len(gold & predicted)
    false_positive = len(predicted - gold)
    false_negative = len(gold - predicted)
    precision = true_positive / len(predicted) if predicted else 0.0
    recall = true_positive / len(gold) if gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    result = artifact.get("result") or {}
    calls = [*(result.get("searches") or []), *(result.get("fetches") or [])]
    vendor_costs = [row.get("call", {}).get("cost_usd") for row in calls]
    known_vendor_costs = [float(value) for value in vendor_costs if value is not None]
    model_cost = result.get("model_cost_usd")
    total_cost = (
        sum(known_vendor_costs) + float(model_cost)
        if model_cost is not None and len(known_vendor_costs) == len(vendor_costs)
        else None
    )
    return {
        "research_mode": artifact["research_mode"],
        "vendor_key": artifact["vendor_key"],
        "vendor": artifact["vendor"],
        "case_key": artifact["case"]["case_key"],
        "trial_index": int(artifact["trial_index"]),
        "status": result.get("status"),
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "exact_set_accuracy": float(false_positive == 0 and false_negative == 0),
        "cited_return_rate": supported / len(predicted) if predicted else 0.0,
        "latency_ms": int(result.get("latency_ms") or 0),
        "model_turn_count": int(result.get("model_turn_count") or 0),
        "cost_usd": total_cost,
    }


def load_artifacts(root: Path) -> list[dict[str, Any]]:
    paths = sorted(root.rglob("trial-*.json"))
    if not paths:
        raise ValueError(f"no trial artifacts found beneath {root}")
    artifacts = []
    for path in paths:
        try:
            artifact = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"cannot read run artifact {path}: {exc}") from exc
        if artifact.get("schema_version") != "multi-turn-company-search-run-v1":
            raise ValueError(f"unsupported run artifact schema in {path}")
        artifacts.append(artifact)
    return artifacts


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    return statistics.fmean(float(row[key]) for row in rows)


def _percent_summary(values: list[float]) -> dict[str, float]:
    return {
        "mean_pct": round(statistics.fmean(values) * 100, 2),
        "sd_pp": round(statistics.stdev(values) * 100, 2) if len(values) > 1 else 0.0,
    }


def _assert_complete(
    grouped: dict[tuple[str, str], list[dict[str, Any]]], expected_trials: int
) -> None:
    expected_indexes = set(range(1, expected_trials + 1))
    cases_by_mode: dict[str, set[str]] = {}
    for (mode, vendor), rows in grouped.items():
        indexes = {row["trial_index"] for row in rows}
        if indexes != expected_indexes:
            raise ValueError(
                f"{mode}/{vendor} has trial indexes {sorted(indexes)}; "
                f"expected {sorted(expected_indexes)}"
            )
        per_trial = {
            trial: {row["case_key"] for row in rows if row["trial_index"] == trial}
            for trial in expected_indexes
        }
        first = per_trial[1]
        if not first or any(case_keys != first for case_keys in per_trial.values()):
            raise ValueError(f"{mode}/{vendor} does not have identical cases in every trial")
        identities = [(row["trial_index"], row["case_key"]) for row in rows]
        if len(identities) != len(set(identities)):
            raise ValueError(f"{mode}/{vendor} contains duplicate trial/case artifacts")
        if mode in cases_by_mode and cases_by_mode[mode] != first:
            raise ValueError(f"{mode}/{vendor} does not cover the same cases as other vendors")
        cases_by_mode.setdefault(mode, first)


def build_leaderboard(
    artifacts: Iterable[dict[str, Any]], *, expected_trials: int = 3
) -> dict[str, Any]:
    if expected_trials <= 0:
        raise ValueError("expected_trials must be positive")
    artifact_list = list(artifacts)
    receipts = {
        (
            artifact.get("dataset", {}).get("slug"),
            artifact.get("dataset", {}).get("content_sha256"),
        )
        for artifact in artifact_list
    }
    if len(receipts) != 1 or not all(next(iter(receipts), (None, None))):
        raise ValueError("all artifacts must carry one identical dataset slug and content hash")
    dataset_slug, dataset_sha256 = next(iter(receipts))
    scored = [score_artifact(artifact) for artifact in artifact_list]
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in scored:
        grouped[(row["research_mode"], row["vendor_key"])].append(row)
    if not grouped:
        raise ValueError("no artifacts supplied")
    _assert_complete(grouped, expected_trials)

    boards: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for (mode, vendor_key), rows in grouped.items():
        trial_values = {
            metric: [
                _mean([row for row in rows if row["trial_index"] == trial], metric)
                for trial in range(1, expected_trials + 1)
            ]
            for metric in QUALITY_METRICS
        }
        costs = [float(row["cost_usd"]) for row in rows if row["cost_usd"] is not None]
        latencies = sorted(row["latency_ms"] for row in rows)
        p95_index = max(0, int(0.95 * len(latencies) + 0.999999) - 1)
        boards[mode].append(
            {
                "vendor_key": vendor_key,
                "vendor": rows[0]["vendor"],
                "question_count": len({row["case_key"] for row in rows}),
                "trial_count": expected_trials,
                "agent_run_count": len(rows),
                **{metric: _percent_summary(values) for metric, values in trial_values.items()},
                "cited_return_rate_pct": round(_mean(rows, "cited_return_rate") * 100, 2),
                "mean_latency_ms": round(statistics.fmean(latencies)),
                "p95_latency_ms": latencies[p95_index],
                "mean_model_turn_count": round(_mean(rows, "model_turn_count"), 2),
                "mean_cost_usd": round(statistics.fmean(costs), 6) if costs else None,
            }
        )

    for rows in boards.values():
        rows.sort(
            key=lambda row: (-row["f1"]["mean_pct"], -row["precision"]["mean_pct"], row["vendor_key"])
        )
        for rank, row in enumerate(rows, start=1):
            row["rank"] = rank
    return {
        "schema_version": "multi-turn-company-search-leaderboard-v1",
        "dataset": {"slug": dataset_slug, "content_sha256": dataset_sha256},
        "aggregation": {
            "quality": "mean and sample standard deviation across trial-level question means",
            "expected_trials": expected_trials,
            "standard_deviation_unit": "percentage points",
        },
        "boards": dict(sorted(boards.items())),
    }


def write_leaderboard(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)
