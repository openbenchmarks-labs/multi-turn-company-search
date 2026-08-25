"""Deterministic structured company-set parsing and conservative resolution."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

from .dataset import normalize_company_name


@dataclass(frozen=True)
class ParsedCompany:
    name: str
    domain: str | None
    cited_urls: list[str]
    evidence: list[dict[str, str]]


@dataclass(frozen=True)
class Resolution:
    company_id: str | None
    method: str
    confidence: float | None


def normalize_domain(value: str | None) -> str | None:
    raw = str(value or "").strip().lower()
    if not raw:
        return None
    if "://" in raw:
        raw = urlparse(raw).hostname or ""
    raw = raw.split("/")[0].split(":")[0].removeprefix("www.").strip(".")
    if not raw or "." not in raw or not re.fullmatch(r"[a-z0-9.-]+", raw):
        return None
    return raw


def _url(value: Any) -> str | None:
    raw = str(value or "").strip()
    parsed = urlparse(raw)
    return raw if parsed.scheme in {"http", "https"} and parsed.hostname else None


def parse_company_set(raw: str) -> list[ParsedCompany]:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"final response is not JSON: {exc}") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("companies"), list):
        raise ValueError("final response must be an object with a companies array")
    output: list[ParsedCompany] = []
    seen: set[str] = set()
    for index, item in enumerate(payload["companies"], 1):
        if not isinstance(item, dict):
            raise ValueError(f"company {index} is not an object")
        name = str(item.get("name") or "").strip()
        if not name:
            raise ValueError(f"company {index} has no name")
        domain = normalize_domain(item.get("domain"))
        urls = list(dict.fromkeys(filter(None, (_url(value) for value in item.get("cited_urls") or []))))
        evidence: list[dict[str, str]] = []
        for row in item.get("evidence") or []:
            if not isinstance(row, dict):
                continue
            url = _url(row.get("url"))
            claim = str(row.get("claim") or "").strip()
            if url and claim:
                evidence.append({"url": url, "claim": claim})
                if url not in urls:
                    urls.append(url)
        identity = domain or normalize_company_name(name)
        if not identity or identity in seen:
            continue
        seen.add(identity)
        output.append(ParsedCompany(name, domain, urls, evidence))
    return output


class CompanyResolver:
    def __init__(self, companies: list[dict[str, Any]], aliases: list[dict[str, Any]]) -> None:
        domains: dict[str, set[str]] = {}
        names: dict[str, set[str]] = {}
        for company in companies:
            company_id = str(company["id"])
            domain = normalize_domain(company.get("canonical_domain"))
            if domain:
                domains.setdefault(domain, set()).add(company_id)
            normalized = normalize_company_name(str(company.get("canonical_name") or ""))
            if normalized:
                names.setdefault(normalized, set()).add(company_id)
        for alias in aliases:
            normalized = normalize_company_name(str(alias.get("alias") or ""))
            if normalized:
                names.setdefault(normalized, set()).add(str(alias["company_id"]))
        self.domains = domains
        self.names = names

    def resolve(self, company: ParsedCompany) -> Resolution:
        if company.domain and len(self.domains.get(company.domain, set())) == 1:
            return Resolution(next(iter(self.domains[company.domain])), "domain", 1.0)
        normalized = normalize_company_name(company.name)
        if normalized and len(self.names.get(normalized, set())) == 1:
            return Resolution(next(iter(self.names[normalized])), "alias", 1.0)
        return Resolution(None, "unresolved", None)
