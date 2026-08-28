"""Vendor web-search and vendor-native fetch adapters.

There is deliberately no generic HTTP fallback in this module. A provider
without a documented native fetch endpoint exposes search only until a custom
fetch implementation is designed and reviewed separately.
"""

from __future__ import annotations

import ipaddress
import os
import re
import socket
import time
from dataclasses import asdict, dataclass
from typing import Any, Callable
from urllib.parse import urlparse

import requests

from .custom_fetch import fetch_page


REDACTED = "***REDACTED***"
RETRYABLE_STATUS = {429, 500, 502, 503, 504}


@dataclass(frozen=True)
class VendorSpec:
    key: str
    provider_slug: str
    provider_name: str
    config_slug: str
    endpoint: str
    env_keys: tuple[str, ...]
    search_unit_cost_usd: float | None
    native_fetch: bool
    fetch_endpoint: str | None
    fetch_kind: str | None
    request_config: dict[str, Any]
    fetch_unit_cost_usd: float | None = None
    custom_fetch: bool = False

    def public_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["env_keys"] = list(self.env_keys)
        return value


@dataclass
class VendorCall:
    status: str
    latency_ms: int
    raw_request: dict[str, Any]
    raw_response: Any
    error: str | None
    attempts: list[dict[str, Any]]
    cost_usd: float | None
    hits: list[dict[str, Any]] | None = None
    page: dict[str, Any] | None = None


VENDORS: dict[str, VendorSpec] = {
    "brave": VendorSpec(
        "brave", "brave-search", "Brave Search", "brave", "GET /res/v1/web/search",
        ("BRAVE_SEARCH_API_KEY",), 0.005, False, None, None,
        {"count": 10, "result_filter": "web"}, custom_fetch=True,
    ),
    "exa_deep": VendorSpec(
        "exa_deep", "exa-deep", "Exa deep", "exa-deep", "POST /search type=deep",
        ("EXA_API_KEY",), 0.012, True, "POST /contents", "exa_contents",
        {"type": "deep", "contents": {"highlights": True}}, fetch_unit_cost_usd=0.001,
    ),
    "exa_instant": VendorSpec(
        "exa_instant", "exa-instant", "Exa instant", "exa-instant", "POST /search type=instant",
        ("EXA_API_KEY",), 0.007, True, "POST /contents", "exa_contents",
        {"type": "instant", "contents": {"highlights": True}}, fetch_unit_cost_usd=0.001,
    ),
    "firecrawl": VendorSpec(
        "firecrawl", "firecrawl-search", "Firecrawl", "firecrawl", "POST /v2/search",
        ("FIRECRAWL_API_KEY",), 0.005, True, "POST /v2/scrape", "firecrawl_scrape",
        {}, fetch_unit_cost_usd=0.0025,
    ),
    "linkup_fast": VendorSpec(
        "linkup_fast", "linkup-fast", "Linkup fast", "linkup-fast", "POST /v1/search depth=fast",
        ("LINKUP_API_KEY",), 0.005, True, "POST /v1/fetch", "linkup_fetch",
        {"depth": "fast", "outputType": "searchResults"}, fetch_unit_cost_usd=0.001,
    ),
    "linkup_standard": VendorSpec(
        "linkup_standard", "linkup-standard", "Linkup standard", "linkup-standard",
        "POST /v1/search depth=standard", ("LINKUP_API_KEY",), 0.005, True,
        "POST /v1/fetch", "linkup_fetch",
        {"depth": "standard", "outputType": "searchResults"}, fetch_unit_cost_usd=0.001,
    ),
    "parallel_advanced": VendorSpec(
        "parallel_advanced", "parallel-advanced", "Parallel advanced", "parallel-advanced",
        "POST /v1/search mode=advanced", ("PARALLEL_API_KEY",), 0.005, True,
        "POST /v1/extract", "parallel_extract",
        {"mode": "advanced", "site_operator_policy": "source_policy"}, fetch_unit_cost_usd=0.001,
    ),
    "parallel_basic": VendorSpec(
        "parallel_basic", "parallel-basic", "Parallel basic", "parallel-basic",
        "POST /v1/search mode=basic", ("PARALLEL_API_KEY",), 0.005, True,
        "POST /v1/extract", "parallel_extract",
        {"mode": "basic", "site_operator_policy": "source_policy"}, fetch_unit_cost_usd=0.001,
    ),
    "parallel_fast": VendorSpec(
        "parallel_fast", "parallel-fast", "Parallel fast", "parallel-fast",
        "POST /v1/search mode=fast", ("PARALLEL_API_KEY",), 0.001, True,
        "POST /v1/extract", "parallel_extract",
        {"mode": "fast", "site_operator_policy": "source_policy"}, fetch_unit_cost_usd=0.001,
    ),
    "parallel_turbo": VendorSpec(
        "parallel_turbo", "parallel-turbo", "Parallel turbo", "parallel-turbo",
        "POST /v1/search mode=turbo", ("PARALLEL_API_KEY",), 0.001, True,
        "POST /v1/extract", "parallel_extract",
        {"mode": "turbo", "site_operator_policy": "source_policy"}, fetch_unit_cost_usd=0.001,
    ),
    "seltz_companies": VendorSpec(
        "seltz_companies", "seltz-companies", "Seltz companies", "seltz-companies",
        "POST /v1/search scope=companies", ("SELTZ_API_KEY",), 0.005, False, None, None,
        {"scope": "companies"}, custom_fetch=True,
    ),
    "serp": VendorSpec(
        "serp", "serp-rapidapi", "SERP (RapidAPI)", "serp", "GET google-search74.p.rapidapi.com",
        ("RAPIDAPI_KEY",), 0.003, False, None, None, {"limit": 10}, custom_fetch=True,
    ),
    "tavily": VendorSpec(
        "tavily", "tavily-advanced", "Tavily advanced", "tavily-advanced",
        "POST /search search_depth=advanced", ("TAVILY_API_KEY",), 0.016, True,
        "POST /extract", "tavily_extract",
        {"search_depth": "advanced", "chunks_per_source": 3}, fetch_unit_cost_usd=0.0032,
    ),
}

DEFAULT_VENDOR_KEYS = tuple(VENDORS)


def capability_inventory() -> list[dict[str, Any]]:
    return [
        {
            "vendor": spec.key,
            "search": True,
            "native_fetch": spec.native_fetch,
            "fetch_kind": spec.fetch_kind,
            "custom_fetch": spec.custom_fetch,
            "fetch_available": spec.native_fetch or spec.custom_fetch,
        }
        for spec in VENDORS.values()
    ]


def required_env(vendor_keys: list[str] | tuple[str, ...]) -> list[str]:
    return sorted({key for vendor in vendor_keys for key in VENDORS[vendor].env_keys})


def validate_vendor_keys(vendor_keys: list[str] | tuple[str, ...]) -> None:
    unknown = sorted(set(vendor_keys) - set(VENDORS))
    if unknown:
        raise ValueError(f"unknown company-research vendors: {unknown}")


def _redact_headers(headers: dict[str, str]) -> dict[str, str]:
    return {
        key: REDACTED if any(token in key.lower() for token in ("key", "token", "authorization", "secret")) else value
        for key, value in headers.items()
    }


def _json_or_text(response: Any) -> Any:
    try:
        return response.json()
    except ValueError:
        return {"_raw": str(getattr(response, "text", ""))[:100_000]}


def _request(
    *,
    method: str,
    url: str,
    headers: dict[str, str],
    body: dict[str, Any] | None = None,
    params: dict[str, Any] | None = None,
    timeout: int = 90,
    retries: int = 3,
    request_fn: Callable[..., Any] = requests.request,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> VendorCall:
    started = time.perf_counter()
    attempts: list[dict[str, Any]] = []
    payload: Any = None
    error: str | None = None
    for attempt in range(1, retries + 2):
        attempt_started = time.perf_counter()
        try:
            response = request_fn(
                method=method, url=url, headers=headers, json=body, params=params, timeout=timeout
            )
            payload = _json_or_text(response)
            attempts.append({
                "attempt": attempt,
                "status_code": response.status_code,
                "latency_ms": round((time.perf_counter() - attempt_started) * 1000),
            })
            if response.ok:
                error = None
                break
            error = f"HTTP {response.status_code}: {str(payload)[:500]}"
            if response.status_code not in RETRYABLE_STATUS or attempt > retries:
                break
        except Exception as exc:  # noqa: BLE001
            error = f"{type(exc).__name__}: {exc}"
            attempts.append({
                "attempt": attempt,
                "status_code": None,
                "latency_ms": round((time.perf_counter() - attempt_started) * 1000),
                "error": error,
            })
            if attempt > retries:
                break
        sleep_fn(min(2 ** (attempt - 1), 8))
    return VendorCall(
        status="ok" if error is None else "error",
        latency_ms=round((time.perf_counter() - started) * 1000),
        raw_request={
            "method": method,
            "url": url,
            "headers": _redact_headers(headers),
            "body": body,
            "params": params,
        },
        raw_response=payload,
        error=error,
        attempts=attempts,
        cost_usd=None,
    )


def _hit(url: Any, title: Any, snippet: Any, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "url": str(url or ""),
        "title": str(title or ""),
        "snippet": str(snippet or "")[:4000],
        "metadata": metadata or {},
    }


def _dedupe(hits: list[dict[str, Any]], max_results: int) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    seen: set[str] = set()
    for hit in hits:
        url = hit["url"].strip()
        if not url or url in seen:
            continue
        seen.add(url)
        output.append(hit)
        if len(output) >= max_results:
            break
    return output


def _reported_dollar_cost(payload: Any) -> float | None:
    if not isinstance(payload, dict):
        return None
    cost = payload.get("costDollars")
    if not isinstance(cost, dict):
        return None
    try:
        total = float(cost.get("total"))
    except (TypeError, ValueError):
        return None
    return total if total >= 0 else None


_SITE_INCLUDE_RE = re.compile(r"(?<![-])site:([^\s]+)", re.I)


def parallel_site_policy(query: str) -> tuple[str, list[str]]:
    """Move positive site: filters to Parallel's include_domains policy."""
    domains: list[str] = []
    seen: set[str] = set()

    def take(match: re.Match[str]) -> str:
        raw = match.group(1).strip("\"'").rstrip("/")
        raw = re.sub(r"^https?://", "", raw, flags=re.I).removeprefix("www.")
        host = raw.split("/", 1)[0].strip(".").lower()
        if host and host not in seen:
            seen.add(host)
            domains.append(host)
        return " "

    cleaned = re.sub(r"\s+", " ", _SITE_INCLUDE_RE.sub(take, query)).strip()
    return cleaned, domains


def _parse_hits(vendor_key: str, payload: Any, max_results: int) -> list[dict[str, Any]]:
    payload = payload if isinstance(payload, dict) else {}
    hits: list[dict[str, Any]] = []
    if vendor_key.startswith("parallel_"):
        for item in payload.get("results") or []:
            excerpts = item.get("excerpts") or []
            hits.append(_hit(item.get("url"), item.get("title"), "\n".join(x for x in excerpts if isinstance(x, str))))
    elif vendor_key.startswith("exa_"):
        for item in payload.get("results") or []:
            highlights = item.get("highlights") or []
            snippet = " ".join(x for x in highlights if isinstance(x, str)) or item.get("text")
            hits.append(_hit(item.get("url"), item.get("title"), snippet))
    elif vendor_key == "firecrawl":
        data = payload.get("data") or {}
        rows = (data.get("web") or data.get("results") or []) if isinstance(data, dict) else data
        for item in rows or []:
            hits.append(_hit(item.get("url") or item.get("link"), item.get("title"), item.get("description") or item.get("snippet")))
    elif vendor_key == "brave":
        for item in (payload.get("web") or {}).get("results") or []:
            hits.append(_hit(item.get("url"), item.get("title"), item.get("description") or item.get("snippet")))
    elif vendor_key.startswith("linkup_"):
        for item in payload.get("results") or []:
            if item.get("type") != "image":
                hits.append(_hit(item.get("url"), item.get("name") or item.get("title"), item.get("content") or item.get("snippet")))
    elif vendor_key == "seltz_companies":
        for item in payload.get("documents") or []:
            hits.append(_hit(item.get("url"), item.get("title") or item.get("name"), item.get("content") or item.get("snippet"), {"domain": item.get("domain")}))
    elif vendor_key == "serp":
        for item in payload.get("results") or []:
            hits.append(_hit(item.get("url") or item.get("link"), item.get("title"), item.get("description") or item.get("snippet")))
    elif vendor_key == "tavily":
        for item in payload.get("results") or []:
            hits.append(_hit(item.get("url"), item.get("title"), item.get("content"), {"score": item.get("score")}))
    return _dedupe(hits, max_results)


def search(vendor_key: str, query: str, *, max_results: int = 10) -> VendorCall:
    spec = VENDORS[vendor_key]
    if vendor_key.startswith("parallel_"):
        mode = str(spec.request_config["mode"])
        sent_query = query
        advanced_settings: dict[str, Any] = {"max_results": max_results}
        if spec.request_config.get("site_operator_policy") == "source_policy":
            sent_query, include_domains = parallel_site_policy(query)
            if not sent_query:
                sent_query = " ".join(include_domains) or query
            if include_domains:
                advanced_settings["source_policy"] = {"include_domains": include_domains}
        call = _request(
            method="POST", url="https://api.parallel.ai/v1/search",
            headers={"x-api-key": os.environ["PARALLEL_API_KEY"], "Content-Type": "application/json"},
            body={
                "objective": sent_query,
                "search_queries": [sent_query],
                "mode": mode,
                "advanced_settings": advanced_settings,
            },
            timeout=120 if mode == "advanced" else 60,
        )
    elif vendor_key.startswith("exa_"):
        kind = vendor_key.removeprefix("exa_")
        call = _request(
            method="POST", url="https://api.exa.ai/search",
            headers={"x-api-key": os.environ["EXA_API_KEY"], "Content-Type": "application/json"},
            body={"query": query, "type": kind, "numResults": max_results, "contents": {"highlights": True}},
            timeout=120 if kind == "deep" else 60,
        )
    elif vendor_key == "firecrawl":
        call = _request(
            method="POST", url="https://api.firecrawl.dev/v2/search",
            headers={"Authorization": f"Bearer {os.environ['FIRECRAWL_API_KEY']}", "Content-Type": "application/json"},
            body={"query": query, "limit": max_results},
        )
    elif vendor_key == "brave":
        call = _request(
            method="GET", url="https://api.search.brave.com/res/v1/web/search",
            headers={"X-Subscription-Token": os.environ["BRAVE_SEARCH_API_KEY"], "Accept": "application/json"},
            params={"q": query, "count": min(max_results, 20), "result_filter": "web"},
            timeout=45,
        )
    elif vendor_key.startswith("linkup_"):
        depth = vendor_key.removeprefix("linkup_")
        call = _request(
            method="POST", url="https://api.linkup.so/v1/search",
            headers={"Authorization": f"Bearer {os.environ['LINKUP_API_KEY']}", "Content-Type": "application/json"},
            body={"q": query, "depth": depth, "outputType": "searchResults", "maxResults": max_results},
        )
    elif vendor_key == "seltz_companies":
        call = _request(
            method="POST", url="https://api.seltz.ai/v1/search",
            headers={"x-api-key": os.environ["SELTZ_API_KEY"], "Content-Type": "application/json"},
            body={"query": query, "max_results": max_results, "scope": "companies"},
            timeout=45,
        )
    elif vendor_key == "serp":
        call = _request(
            method="GET", url="https://google-search74.p.rapidapi.com/",
            headers={"x-rapidapi-key": os.environ["RAPIDAPI_KEY"], "x-rapidapi-host": "google-search74.p.rapidapi.com"},
            params={"query": query, "limit": max_results}, timeout=45,
        )
    elif vendor_key == "tavily":
        call = _request(
            method="POST", url="https://api.tavily.com/search",
            headers={"Authorization": f"Bearer {os.environ['TAVILY_API_KEY']}", "Content-Type": "application/json"},
            body={"query": query, "search_depth": "advanced", "max_results": max_results, "chunks_per_source": 3},
        )
    else:  # pragma: no cover - registry and dispatcher change together
        raise KeyError(vendor_key)
    reported_cost = _reported_dollar_cost(call.raw_response)
    call.cost_usd = (
        reported_cost if reported_cost is not None else spec.search_unit_cost_usd
    ) if call.status == "ok" else 0.0
    call.hits = _parse_hits(vendor_key, call.raw_response, max_results) if call.status == "ok" else []
    return call


def _assert_public_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError("web_fetch requires a public absolute http(s) URL")
    if parsed.hostname.lower() == "localhost":
        raise ValueError("web_fetch blocks localhost and private networks")
    addresses = socket.getaddrinfo(parsed.hostname, parsed.port or (443 if parsed.scheme == "https" else 80))
    if not addresses or any(not ipaddress.ip_address(row[4][0]).is_global for row in addresses):
        raise ValueError("web_fetch blocks localhost and private networks")


def _page(vendor_key: str, url: str, payload: Any, max_chars: int) -> dict[str, Any]:
    payload = payload if isinstance(payload, dict) else {}
    if vendor_key.startswith("parallel_"):
        item = (payload.get("results") or [{}])[0]
        text = item.get("full_content") or "\n".join(item.get("excerpts") or [])
        final_url, title = item.get("url") or url, item.get("title") or ""
    elif vendor_key.startswith("exa_"):
        item = (payload.get("results") or [{}])[0]
        text, final_url, title = item.get("text") or "", item.get("url") or url, item.get("title") or ""
    elif vendor_key == "firecrawl":
        item = payload.get("data") or {}
        metadata = item.get("metadata") or {}
        text, final_url, title = item.get("markdown") or "", metadata.get("sourceURL") or url, metadata.get("title") or ""
    elif vendor_key == "tavily":
        item = (payload.get("results") or [{}])[0]
        text, final_url, title = item.get("raw_content") or item.get("content") or "", item.get("url") or url, ""
    elif vendor_key.startswith("linkup_"):
        text, final_url, title = payload.get("markdown") or "", url, ""
    else:
        raise RuntimeError(f"{vendor_key} has no vendor-native fetch adapter")
    if not str(text).strip():
        raise RuntimeError(f"{vendor_key} native fetch returned no page content")
    return {
        "requested_url": url,
        "final_url": str(final_url),
        "title": str(title),
        "text": str(text)[:max_chars],
        "truncated": len(str(text)) > max_chars,
        "fetch_provider": VENDORS[vendor_key].fetch_kind,
    }


def fetch(vendor_key: str, url: str, *, objective: str, max_chars: int = 12_000) -> VendorCall:
    spec = VENDORS[vendor_key]
    if spec.custom_fetch:
        started = time.perf_counter()
        try:
            page = fetch_page(url, max_chars=max_chars)
            return VendorCall(
                status="ok", latency_ms=round((time.perf_counter() - started) * 1000),
                raw_request={"url": url, "fetch_kind": "custom_http_with_playwright_fallback"},
                raw_response=page, error=None,
                attempts=[{"attempt": 1, "provider": page["fetch_provider"]}],
                cost_usd=0.0, page=page,
            )
        except Exception as exc:  # noqa: BLE001
            return VendorCall(
                status="error", latency_ms=round((time.perf_counter() - started) * 1000),
                raw_request={"url": url, "fetch_kind": "custom_http_with_playwright_fallback"},
                raw_response=None, error=f"{type(exc).__name__}: {exc}",
                attempts=[], cost_usd=0.0,
            )
    if not spec.native_fetch:
        return VendorCall(
            status="unavailable", latency_ms=0,
            raw_request={"url": url, "vendor": vendor_key}, raw_response=None,
            error="vendor-native web_fetch is unavailable; custom fetch is intentionally not implemented",
            attempts=[], cost_usd=None,
        )
    _assert_public_url(url)
    if vendor_key.startswith("parallel_"):
        call = _request(
            method="POST", url="https://api.parallel.ai/v1/extract",
            headers={"x-api-key": os.environ["PARALLEL_API_KEY"], "Content-Type": "application/json"},
            body={"urls": [url], "objective": objective, "max_chars_total": max_chars, "advanced_settings": {"full_content": True}},
            timeout=120,
        )
    elif vendor_key.startswith("exa_"):
        call = _request(
            method="POST", url="https://api.exa.ai/contents",
            headers={"x-api-key": os.environ["EXA_API_KEY"], "Content-Type": "application/json"},
            body={"urls": [url], "text": {"maxCharacters": max_chars, "includeHtmlTags": False}},
        )
    elif vendor_key == "firecrawl":
        call = _request(
            method="POST", url="https://api.firecrawl.dev/v2/scrape",
            headers={"Authorization": f"Bearer {os.environ['FIRECRAWL_API_KEY']}", "Content-Type": "application/json"},
            body={"url": url, "formats": ["markdown"], "onlyMainContent": True},
        )
    elif vendor_key == "tavily":
        call = _request(
            method="POST", url="https://api.tavily.com/extract",
            headers={"Authorization": f"Bearer {os.environ['TAVILY_API_KEY']}", "Content-Type": "application/json"},
            body={"urls": [url], "extract_depth": "advanced"},
        )
    elif vendor_key.startswith("linkup_"):
        call = _request(
            method="POST", url="https://api.linkup.so/v1/fetch",
            headers={"Authorization": f"Bearer {os.environ['LINKUP_API_KEY']}", "Content-Type": "application/json"},
            body={"url": url, "mode": "standard", "renderJs": False},
        )
    else:  # pragma: no cover
        raise KeyError(vendor_key)
    if call.status == "ok":
        try:
            call.page = _page(vendor_key, url, call.raw_response, max_chars)
        except Exception as exc:  # noqa: BLE001
            call.status = "error"
            call.error = f"{type(exc).__name__}: {exc}"
    reported_cost = _reported_dollar_cost(call.raw_response)
    call.cost_usd = (
        reported_cost if reported_cost is not None else spec.fetch_unit_cost_usd
    ) if call.status == "ok" else 0.0
    return call
