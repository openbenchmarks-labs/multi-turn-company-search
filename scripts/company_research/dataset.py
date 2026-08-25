"""Frozen dataset contract for the multi-constraint company-research bench."""

from __future__ import annotations

import hashlib
import json
import re
from difflib import SequenceMatcher
from itertools import combinations
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "company-research-dataset-v1"
DEFAULT_DATASET_SLUG = "multi-constraint-company-research-v1"
DEFAULT_DATASET_NAME = "Multi-Constraint Company Research — 45 Query Set"
SOURCE_SYSTEM = "fiber_companies_name_v1"
FAMILIES = {"investor", "accelerator", "funding", "other"}


class DatasetValidationError(ValueError):
    """The local artifact is unsafe to publish."""


@dataclass(frozen=True)
class DatasetSummary:
    dataset_slug: str
    case_count: int
    company_count: int
    gold_membership_count: int
    content_sha256: str


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def content_sha256(payload: dict[str, Any]) -> str:
    without_receipt = dict(payload)
    without_receipt.pop("content_sha256", None)
    return "sha256:" + hashlib.sha256(canonical_json(without_receipt).encode("utf-8")).hexdigest()


def normalize_company_name(value: str) -> str:
    value = re.sub(r"\s*\([^)]*\)\s*$", "", value.casefold())
    return re.sub(r"[^a-z0-9]+", " ", value).strip()


def entity_key(name: str) -> str:
    normalized = normalize_company_name(name)
    return "name:" + hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:24]


def parse_curated_queries(text: str) -> list[dict[str, str]]:
    """Parse alternating question/arrow rows without splitting company commas."""
    rows: list[dict[str, str]] = []
    pending: tuple[int, str] | None = None
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("->"):
            if pending is None:
                raise DatasetValidationError(f"line {line_number}: answer has no question")
            answer_text = line[2:].strip()
            if not answer_text:
                raise DatasetValidationError(f"line {line_number}: answer list is empty")
            rows.append(
                {
                    "question": pending[1],
                    "question_line": str(pending[0]),
                    "answer_text": answer_text,
                    "answer_line": str(line_number),
                }
            )
            pending = None
        else:
            if pending is not None:
                raise DatasetValidationError(f"line {pending[0]}: question has no answer")
            pending = (line_number, line)
    if pending is not None:
        raise DatasetValidationError(f"line {pending[0]}: question has no answer")
    if not rows:
        raise DatasetValidationError("curated query file is empty")
    return rows


def _answer_subset(answers: list[str], rendered: str) -> list[str] | None:
    """Resolve a comma-rendered row against known names, including names with commas."""
    matches = [
        [answers[index] for index in indexes]
        for size in range(1, len(answers) + 1)
        for indexes in combinations(range(len(answers)), size)
        if ", ".join(answers[index] for index in indexes) == rendered
    ]
    if len(matches) > 1:
        raise DatasetValidationError(f"ambiguous curated answer list {rendered!r}")
    return matches[0] if matches else None


def _curated_clauses(source_clauses: list[str], question: str) -> list[str]:
    replacements = {
        "closed a 2026 growth-stage financing in the mid tens of millions":
            "closed a 2026 growth-stage financing in the $10-$90 million range",
        "build hardware rather than pure software": "build hardware",
    }
    clauses = []
    folded_question = question.casefold()
    for clause in source_clauses:
        if clause.casefold() in folded_question:
            clauses.append(clause)
            continue
        replacement = replacements.get(clause)
        if replacement and replacement.casefold() in folded_question:
            clauses.append(replacement)
            continue
        if clause == "closed a 2026 growth-stage financing in the mid tens of millions":
            funding_range = re.search(
                r"closed a 2026 growth-stage financing in the \$10-\$?90 million range",
                question,
                flags=re.IGNORECASE,
            )
            if funding_range:
                clauses.append(funding_range.group(0))
                continue
        raise DatasetValidationError(
            f"curated question no longer contains source clause {clause!r}: {question!r}"
        )
    return clauses


def build_curated_artifact(
    bank: dict[str, Any],
    curated_text: str,
    *,
    dataset_slug: str = DEFAULT_DATASET_SLUG,
    dataset_name: str = DEFAULT_DATASET_NAME,
    source_snapshot: str = "fiber_companies snapshot used by simulator bank-100",
) -> dict[str, Any]:
    """Select and safely override bank rows using the hand-curated text file."""
    bank_items = _require_list(bank.get("items"), "bank.items")
    curated_rows = parse_curated_queries(curated_text)
    selected_items: list[dict[str, Any]] = []
    selected_ids: set[str] = set()

    for row in curated_rows:
        candidates = []
        for raw in bank_items:
            answers = _require_list(raw.get("answers"), "bank item answers")
            selected_answers = _answer_subset(answers, row["answer_text"])
            if selected_answers is None:
                continue
            score = SequenceMatcher(
                None,
                re.sub(r"\s+", " ", row["question"].casefold()),
                re.sub(r"\s+", " ", str(raw.get("question") or "").casefold()),
            ).ratio()
            candidates.append((score, raw, selected_answers))
        if not candidates:
            raise DatasetValidationError(
                f"line {row['question_line']}: answers do not match any bank case"
            )
        score, source, selected_answers = max(candidates, key=lambda candidate: candidate[0])
        if score < 0.85:
            raise DatasetValidationError(
                f"line {row['question_line']}: best bank question match is only {score:.3f}"
            )
        source_id = _require_text(source.get("id"), "bank item id")
        if source_id in selected_ids:
            raise DatasetValidationError(f"curated file selects bank case {source_id!r} twice")
        selected_ids.add(source_id)

        selected = dict(source)
        selected["question"] = row["question"]
        selected["answers"] = selected_answers
        selected["answer_count"] = len(selected_answers)
        selected["answer_aliases"] = {
            answer: list((source.get("answer_aliases") or {}).get(answer) or [])
            for answer in selected_answers
        }
        selected["clauses"] = _curated_clauses(list(source.get("clauses") or []), row["question"])
        selected["curation"] = {
            "question_line": int(row["question_line"]),
            "answer_line": int(row["answer_line"]),
            "question_matches_source": row["question"] == source.get("question"),
            "answers_match_source": selected_answers == source.get("answers"),
        }
        selected_items.append(selected)

    selected_bank = {**bank, "items": selected_items, "n": len(selected_items)}
    payload = build_artifact(
        selected_bank,
        dataset_slug=dataset_slug,
        dataset_name=dataset_name,
        source_snapshot=source_snapshot,
    )
    payload.pop("content_sha256")
    payload["source"].update(
        {
            "selection": "main-bench/final_queries.txt",
            "bank": "simulator/bank-100.json",
            "gold_status": "provisional_pending_hand_labelling",
        }
    )
    payload["dataset"]["notes"] = (
        "Curated 45-query multi-constraint company discovery set. Initial gold is a "
        "provisional selection from the frozen Fiber census and will be superseded by "
        "a hand-labelled release after vendor runs."
    )
    payload["content_sha256"] = content_sha256(payload)
    validate_artifact(payload)
    return payload


def _require_text(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DatasetValidationError(f"{path} must be a non-empty string")
    return value.strip()


def _require_list(value: Any, path: str) -> list[Any]:
    if not isinstance(value, list):
        raise DatasetValidationError(f"{path} must be an array")
    return value


def build_artifact(
    bank: dict[str, Any],
    *,
    dataset_slug: str = DEFAULT_DATASET_SLUG,
    dataset_name: str = DEFAULT_DATASET_NAME,
    source_snapshot: str = "fiber_companies snapshot used by simulator bank-100",
) -> dict[str, Any]:
    items = _require_list(bank.get("items"), "bank.items")
    cases: list[dict[str, Any]] = []
    for index, raw in enumerate(items, start=1):
        if not isinstance(raw, dict):
            raise DatasetValidationError(f"bank.items[{index - 1}] must be an object")
        answers = _require_list(raw.get("answers"), f"bank.items[{index - 1}].answers")
        aliases = raw.get("answer_aliases") or {}
        domains = raw.get("answer_domains") or {}
        gold = []
        for answer in answers:
            name = _require_text(answer, f"bank.items[{index - 1}].answers")
            answer_aliases = aliases.get(answer) or []
            domain = str(domains.get(answer) or "").strip().lower() or None
            gold.append(
                {
                    "entity_key": entity_key(name),
                    "name": name,
                    "domain": domain,
                    "aliases": list(dict.fromkeys(
                        alias.strip()
                        for alias in answer_aliases
                        if isinstance(alias, str) and alias.strip() and alias.strip() != name
                    )),
                }
            )
        cases.append(
            {
                "case_key": _require_text(raw.get("id"), f"bank.items[{index - 1}].id"),
                "question": _require_text(raw.get("question"), f"bank.items[{index - 1}].question"),
                "family": _require_text(raw.get("family"), f"bank.items[{index - 1}].family"),
                "constraint_count": int(raw.get("constraint_count") or 0),
                "constraints": list(raw.get("clauses") or []),
                "predicates": {
                    "axes": list(raw.get("axes") or []),
                    "labels": dict(raw.get("labels") or {}),
                },
                "census_metadata": dict(raw.get("census") or {}),
                "source_metadata": {
                    "combination": raw.get("combination"),
                    "coverage": raw.get("coverage") or {},
                },
                "sort_order": index,
                "gold": gold,
            }
        )

    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "dataset": {
            "slug": dataset_slug,
            "name": dataset_name,
            "region": "Global",
            "notes": (
                "Deterministic 100-query multi-constraint company discovery set. "
                "Initial gold is generated from the frozen Fiber company census and "
                "is expected to receive a later hand-labelled gold release."
            ),
        },
        "source": {
            "type": "clickhouse_fiber_companies",
            "snapshot": source_snapshot,
            "generator": "simulator/dataset_simulations/web_search/final_bank.py",
            "method": bank.get("method"),
        },
        "gold_release": {"version": 1, "source": "generated"},
        "cases": cases,
    }
    payload["content_sha256"] = content_sha256(payload)
    validate_artifact(payload)
    return payload


def validate_artifact(payload: dict[str, Any]) -> DatasetSummary:
    if not isinstance(payload, dict):
        raise DatasetValidationError("dataset root must be an object")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise DatasetValidationError(
            f"schema_version must be {SCHEMA_VERSION!r}, got {payload.get('schema_version')!r}"
        )
    expected_hash = content_sha256(payload)
    if payload.get("content_sha256") != expected_hash:
        raise DatasetValidationError(
            f"content hash mismatch: stored={payload.get('content_sha256')!r} computed={expected_hash!r}"
        )

    dataset = payload.get("dataset")
    if not isinstance(dataset, dict):
        raise DatasetValidationError("dataset must be an object")
    dataset_slug = _require_text(dataset.get("slug"), "dataset.slug")
    _require_text(dataset.get("name"), "dataset.name")

    source = payload.get("source")
    if not isinstance(source, dict):
        raise DatasetValidationError("source must be an object")
    _require_text(source.get("type"), "source.type")

    release = payload.get("gold_release")
    if not isinstance(release, dict) or release.get("version") != 1:
        raise DatasetValidationError("gold_release.version must be 1")
    if release.get("source") != "generated":
        raise DatasetValidationError("the initial gold release must use source='generated'")

    cases = _require_list(payload.get("cases"), "cases")
    if not cases:
        raise DatasetValidationError("cases cannot be empty")
    case_keys: set[str] = set()
    sort_orders: set[int] = set()
    entity_names: dict[str, str] = {}
    memberships = 0
    for index, case in enumerate(cases):
        path = f"cases[{index}]"
        if not isinstance(case, dict):
            raise DatasetValidationError(f"{path} must be an object")
        key = _require_text(case.get("case_key"), f"{path}.case_key")
        if key in case_keys:
            raise DatasetValidationError(f"duplicate case_key {key!r}")
        case_keys.add(key)
        question = _require_text(case.get("question"), f"{path}.question")
        family = _require_text(case.get("family"), f"{path}.family")
        if family not in FAMILIES:
            raise DatasetValidationError(f"{path}.family is unsupported: {family!r}")
        constraint_count = case.get("constraint_count")
        constraints = _require_list(case.get("constraints"), f"{path}.constraints")
        if not isinstance(constraint_count, int) or constraint_count <= 0:
            raise DatasetValidationError(f"{path}.constraint_count must be positive")
        if len(constraints) != constraint_count:
            raise DatasetValidationError(
                f"{path} has constraint_count={constraint_count} but {len(constraints)} clauses"
            )
        if any(not isinstance(clause, str) or not clause.strip() for clause in constraints):
            raise DatasetValidationError(f"{path}.constraints contains an empty clause")
        if any(clause.casefold() not in question.casefold() for clause in constraints):
            raise DatasetValidationError(f"{path}.question does not contain every constraint clause")
        if not isinstance(case.get("predicates"), dict):
            raise DatasetValidationError(f"{path}.predicates must be an object")
        order = case.get("sort_order")
        if not isinstance(order, int) or order <= 0 or order in sort_orders:
            raise DatasetValidationError(f"{path}.sort_order must be unique and positive")
        sort_orders.add(order)

        gold = _require_list(case.get("gold"), f"{path}.gold")
        if not gold:
            raise DatasetValidationError(f"{path}.gold cannot be empty")
        member_keys: set[str] = set()
        for member_index, member in enumerate(gold):
            member_path = f"{path}.gold[{member_index}]"
            if not isinstance(member, dict):
                raise DatasetValidationError(f"{member_path} must be an object")
            name = _require_text(member.get("name"), f"{member_path}.name")
            key_value = _require_text(member.get("entity_key"), f"{member_path}.entity_key")
            if key_value != entity_key(name):
                raise DatasetValidationError(f"{member_path}.entity_key is not derived from its name")
            if key_value in member_keys:
                raise DatasetValidationError(f"{path} repeats entity {name!r}")
            member_keys.add(key_value)
            existing_name = entity_names.setdefault(key_value, name)
            if normalize_company_name(existing_name) != normalize_company_name(name):
                raise DatasetValidationError(f"entity key collision: {existing_name!r} and {name!r}")
            aliases = _require_list(member.get("aliases"), f"{member_path}.aliases")
            if any(not isinstance(alias, str) or not alias.strip() for alias in aliases):
                raise DatasetValidationError(f"{member_path}.aliases contains an empty value")
            memberships += 1

    if sort_orders != set(range(1, len(cases) + 1)):
        raise DatasetValidationError("case sort_order values must be contiguous from 1")
    return DatasetSummary(
        dataset_slug=dataset_slug,
        case_count=len(cases),
        company_count=len(entity_names),
        gold_membership_count=memberships,
        content_sha256=expected_hash,
    )


def load_artifact(path: Path) -> tuple[dict[str, Any], DatasetSummary]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload, validate_artifact(payload)


def write_artifact(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
