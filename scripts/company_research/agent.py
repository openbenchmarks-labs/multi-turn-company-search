"""GPT-5.6 company-discovery agent using vendor-backed function tools only."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from typing import Any
from urllib.parse import urldefrag

from .vendors import VENDORS, VendorCall, fetch, search


DEFAULT_MODEL = "gpt-5.6-sol"
DEFAULT_REASONING_EFFORT = "medium"
REASONING_EFFORTS = {"none", "low", "medium", "high", "xhigh", "max"}
PROTOCOL_VERSION = "company-research-agent-v2"
OUTPUT_SCHEMA_VERSION = "company-set-v1"
PARSER_VERSION = "structured-company-set-v1"

SYSTEM_PROMPT = """You are a company research agent. Identify companies that satisfy ALL
constraints in the user's question. Use the supplied vendor web_search tool to
form focused research queries. When web_fetch is available, use it only on an
exact URL returned by an earlier search when the snippet is insufficient.

Do not use prior knowledge as evidence. Do not include a company unless the
collected evidence supports every constraint. Prefer precision over padding.
Return each company once under its canonical public name, include its canonical
domain when known, cite the URLs supporting it, and briefly state the evidence.
An empty company list is valid when the evidence is insufficient."""

WEB_SEARCH_TOOL = {
    "type": "function",
    "name": "web_search",
    "description": "Search through the benchmark vendor. Use focused queries for missing evidence.",
    "parameters": {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
        "additionalProperties": False,
    },
    "strict": True,
}

WEB_FETCH_TOOL = {
    "type": "function",
    "name": "web_fetch",
    "description": "Fetch an exact URL returned by an earlier vendor web_search call.",
    "parameters": {
        "type": "object",
        "properties": {"url": {"type": "string"}},
        "required": ["url"],
        "additionalProperties": False,
    },
    "strict": True,
}

COMPANY_SET_SCHEMA = {
    "type": "object",
    "properties": {
        "companies": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "domain": {"type": ["string", "null"]},
                    "cited_urls": {"type": "array", "items": {"type": "string"}},
                    "evidence": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "url": {"type": "string"},
                                "claim": {"type": "string"},
                            },
                            "required": ["url", "claim"],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": ["name", "domain", "cited_urls", "evidence"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["companies"],
    "additionalProperties": False,
}


@dataclass(frozen=True)
class AgentConfig:
    model: str = DEFAULT_MODEL
    reasoning_effort: str = DEFAULT_REASONING_EFFORT
    max_turns: int = 8
    max_searches: int = 14
    max_results: int = 10
    max_searches_per_turn: int | None = 2
    max_fetches: int = 14
    max_fetches_per_turn: int | None = 2
    max_output_tokens: int = 6000
    model_input_usd_per_million: float | None = 5.0
    model_cached_input_usd_per_million: float | None = 0.5
    model_cache_write_usd_per_million: float | None = 6.25
    model_output_usd_per_million: float | None = 30.0

    def validate(self) -> None:
        if not self.model.strip():
            raise ValueError("model must not be empty")
        if self.reasoning_effort not in REASONING_EFFORTS:
            raise ValueError(
                f"reasoning_effort must be one of {sorted(REASONING_EFFORTS)}"
            )
        for name in ("max_turns", "max_results", "max_output_tokens"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.max_searches <= 0:
            raise ValueError("max_searches must be positive")
        if self.max_fetches < 0:
            raise ValueError("max_fetches cannot be negative")
        for name in ("max_searches_per_turn", "max_fetches_per_turn"):
            if getattr(self, name) is not None and getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive when supplied")
        for name in (
            "model_input_usd_per_million",
            "model_cached_input_usd_per_million",
            "model_cache_write_usd_per_million",
            "model_output_usd_per_million",
        ):
            if getattr(self, name) is not None and getattr(self, name) < 0:
                raise ValueError(f"{name} cannot be negative")


@dataclass
class SearchTrace:
    turn_index: int
    call_index: int
    tool_call_id: str
    query: str
    call: VendorCall


@dataclass
class FetchTrace:
    turn_index: int
    call_index: int
    tool_call_id: str
    url: str
    call: VendorCall


@dataclass
class AgentResult:
    question: str
    vendor_key: str
    final_response: str
    status: str
    error: str | None
    latency_ms: int
    searches: list[SearchTrace] = field(default_factory=list)
    fetches: list[FetchTrace] = field(default_factory=list)
    model_responses: list[dict[str, Any]] = field(default_factory=list)
    input_tokens: int = 0
    cached_input_tokens: int = 0
    cache_write_input_tokens: int = 0
    output_tokens: int = 0
    model_cost_usd: float | None = None

    def raw_model_payload(self) -> dict[str, Any]:
        return {
            "protocol_version": PROTOCOL_VERSION,
            "vendor_key": self.vendor_key,
            "responses": self.model_responses,
            "turn_count": len(self.model_responses),
            "usage": {
                "input_tokens": self.input_tokens,
                "cached_input_tokens": self.cached_input_tokens,
                "cache_write_input_tokens": self.cache_write_input_tokens,
                "output_tokens": self.output_tokens,
                "total_tokens": self.input_tokens + self.output_tokens,
            },
        }


def prompt_sha256() -> str:
    return hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest()


def _item_type(item: Any) -> str:
    return str(item.get("type") if isinstance(item, dict) else getattr(item, "type", "") or "")


def _function_calls(response: Any) -> list[Any]:
    return [item for item in (getattr(response, "output", None) or []) if _item_type(item) == "function_call"]


def _call_value(call: Any, key: str) -> str:
    value = call.get(key) if isinstance(call, dict) else getattr(call, key, "")
    return str(value or "")


def _arguments(call: Any) -> dict[str, Any]:
    raw = call.get("arguments") if isinstance(call, dict) else getattr(call, "arguments", "{}")
    if isinstance(raw, dict):
        return raw
    try:
        value = json.loads(raw or "{}")
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def _item_as_input(item: Any) -> dict[str, Any]:
    if isinstance(item, dict):
        return item
    if hasattr(item, "model_dump"):
        return item.model_dump(mode="json", exclude_none=True)
    payload = {
        "type": getattr(item, "type", None),
        "id": getattr(item, "id", None),
        "call_id": getattr(item, "call_id", None),
        "name": getattr(item, "name", None),
        "arguments": getattr(item, "arguments", None),
        "status": getattr(item, "status", None),
    }
    return {key: value for key, value in payload.items() if value is not None}


def _tool_output(call: Any, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "function_call_output",
        "call_id": _call_value(call, "call_id") or _call_value(call, "id"),
        "output": json.dumps(payload, ensure_ascii=False),
    }


def _response_dict(response: Any) -> dict[str, Any]:
    if hasattr(response, "model_dump"):
        return response.model_dump(mode="json", exclude_none=True)
    if isinstance(response, dict):
        return response
    return {"output_text": str(getattr(response, "output_text", ""))}


def _usage(response: Any) -> tuple[int, int, int, int]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return 0, 0, 0, 0
    if isinstance(usage, dict):
        details = usage.get("input_tokens_details") or {}
        def detail_value(key: str) -> Any:
            return details.get(key) if isinstance(details, dict) else getattr(details, key, 0)

        return (
            int(usage.get("input_tokens") or 0),
            int(detail_value("cached_tokens") or 0),
            int(detail_value("cache_write_tokens") or 0),
            int(usage.get("output_tokens") or 0),
        )
    details = getattr(usage, "input_tokens_details", None)
    return (
        int(getattr(usage, "input_tokens", 0) or 0),
        int(getattr(details, "cached_tokens", 0) or 0),
        int(getattr(details, "cache_write_tokens", 0) or 0),
        int(getattr(usage, "output_tokens", 0) or 0),
    )


def _model_cost(
    config: AgentConfig,
    input_tokens: int,
    cached_input_tokens: int,
    cache_write_input_tokens: int,
    output_tokens: int,
) -> float | None:
    rates = (
        config.model_input_usd_per_million,
        config.model_cached_input_usd_per_million,
        config.model_cache_write_usd_per_million,
        config.model_output_usd_per_million,
    )
    if any(rate is None for rate in rates):
        return None
    if cached_input_tokens + cache_write_input_tokens > input_tokens:
        raise ValueError("cached and cache-write input tokens exceed total input tokens")
    uncached_input_tokens = input_tokens - cached_input_tokens - cache_write_input_tokens
    return round(
        uncached_input_tokens * config.model_input_usd_per_million / 1_000_000
        + cached_input_tokens * config.model_cached_input_usd_per_million / 1_000_000
        + cache_write_input_tokens * config.model_cache_write_usd_per_million / 1_000_000
        + output_tokens * config.model_output_usd_per_million / 1_000_000,
        6,
    )


def run_agent(
    question: str,
    *,
    vendor_key: str,
    client: Any,
    config: AgentConfig,
) -> AgentResult:
    """Execute one paid agent trial. Callers own all authorization safeguards."""
    config.validate()
    spec = VENDORS[vendor_key]
    allow_fetch = (spec.native_fetch or spec.custom_fetch) and config.max_fetches > 0
    tools = [WEB_SEARCH_TOOL, *([WEB_FETCH_TOOL] if allow_fetch else [])]
    conversation: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    allowed_fetch_urls: set[str] = set()
    result = AgentResult(question, vendor_key, "", "error", None, 0)
    started = time.perf_counter()
    try:
        for turn_zero in range(config.max_turns):
            turn_index = turn_zero + 1
            searches_this_turn = 0
            fetches_this_turn = 0
            final_turn = turn_index == config.max_turns
            if final_turn:
                conversation.append({
                    "role": "system",
                    "content": "This is the final turn. Return the best supported company set now.",
                })
            request: dict[str, Any] = {
                "model": config.model,
                "reasoning": {"effort": config.reasoning_effort},
                "input": conversation,
                "max_output_tokens": config.max_output_tokens,
                "store": False,
                "text": {
                    "format": {
                        "type": "json_schema",
                        "name": "company_set",
                        "strict": True,
                        "schema": COMPANY_SET_SCHEMA,
                    }
                },
            }
            if not final_turn:
                request["tools"] = tools
            response = client.responses.create(**request)
            result.model_responses.append(_response_dict(response))
            input_tokens, cached_input_tokens, cache_write_input_tokens, output_tokens = _usage(response)
            result.input_tokens += input_tokens
            result.cached_input_tokens += cached_input_tokens
            result.cache_write_input_tokens += cache_write_input_tokens
            result.output_tokens += output_tokens
            calls = _function_calls(response)
            if not calls:
                result.final_response = str(getattr(response, "output_text", "") or "").strip()
                if not result.final_response:
                    raise RuntimeError(f"model returned no structured output on turn {turn_index}")
                if not any(trace.call.status == "ok" for trace in result.searches):
                    raise RuntimeError("model returned without a successful vendor web search")
                result.status = "ok"
                break

            for item in getattr(response, "output", None) or []:
                conversation.append(_item_as_input(item))
            for call in calls:
                name = _call_value(call, "name")
                call_id = _call_value(call, "call_id") or _call_value(call, "id")
                arguments = _arguments(call)
                if name == "web_search":
                    query = str(arguments.get("query") or "").strip()
                    if not query:
                        conversation.append(_tool_output(call, {"error": "query is required"}))
                        continue
                    if len(result.searches) >= config.max_searches:
                        conversation.append(_tool_output(call, {"error": "search budget exhausted; answer now"}))
                        continue
                    if config.max_searches_per_turn is not None and searches_this_turn >= config.max_searches_per_turn:
                        conversation.append(_tool_output(call, {"error": "per-turn search budget exhausted"}))
                        continue
                    searches_this_turn += 1
                    vendor_call = search(vendor_key, query, max_results=config.max_results)
                    trace = SearchTrace(turn_index, len(result.searches) + 1, call_id, query, vendor_call)
                    result.searches.append(trace)
                    for hit in vendor_call.hits or []:
                        if hit.get("url"):
                            allowed_fetch_urls.add(urldefrag(str(hit["url"]))[0])
                    conversation.append(_tool_output(call, {
                        "vendor": vendor_key,
                        "query": query,
                        "hits": vendor_call.hits or [],
                        "error": vendor_call.error,
                    }))
                elif name == "web_fetch":
                    url = str(arguments.get("url") or "").strip()
                    normalized = urldefrag(url)[0]
                    if not allow_fetch:
                        conversation.append(_tool_output(call, {"error": "vendor-native web_fetch is unavailable"}))
                        continue
                    if len(result.fetches) >= config.max_fetches:
                        conversation.append(_tool_output(call, {"error": "fetch budget exhausted; answer now"}))
                        continue
                    if config.max_fetches_per_turn is not None and fetches_this_turn >= config.max_fetches_per_turn:
                        conversation.append(_tool_output(call, {"error": "per-turn fetch budget exhausted"}))
                        continue
                    if not url or normalized not in allowed_fetch_urls:
                        conversation.append(_tool_output(call, {"error": "URL must exactly match a prior search result"}))
                        continue
                    fetches_this_turn += 1
                    vendor_call = fetch(vendor_key, url, objective=question)
                    result.fetches.append(FetchTrace(turn_index, len(result.fetches) + 1, call_id, url, vendor_call))
                    conversation.append(_tool_output(call, {"page": vendor_call.page, "error": vendor_call.error}))
                else:
                    conversation.append(_tool_output(call, {"error": f"unknown tool {name}"}))
        else:
            raise RuntimeError("agent exhausted max_turns without a final company set")
    except Exception as exc:  # noqa: BLE001
        result.status = "error"
        result.error = f"{type(exc).__name__}: {exc}"
    result.latency_ms = round((time.perf_counter() - started) * 1000)
    try:
        result.model_cost_usd = _model_cost(
            config,
            result.input_tokens,
            result.cached_input_tokens,
            result.cache_write_input_tokens,
            result.output_tokens,
        )
    except ValueError as exc:
        result.status = "error"
        result.error = f"{type(exc).__name__}: {exc}"
    return result


def result_manifest(result: AgentResult) -> dict[str, Any]:
    """Serializable local representation useful for tests and offline replay."""
    return {
        "question": result.question,
        "vendor_key": result.vendor_key,
        "final_response": result.final_response,
        "status": result.status,
        "error": result.error,
        "latency_ms": result.latency_ms,
        "model_input_tokens": result.input_tokens,
        "model_cached_input_tokens": result.cached_input_tokens,
        "model_cache_write_input_tokens": result.cache_write_input_tokens,
        "model_output_tokens": result.output_tokens,
        "model_total_tokens": result.input_tokens + result.output_tokens,
        "model_turn_count": len(result.model_responses),
        "searches": [
            {**asdict(trace), "call": asdict(trace.call)} for trace in result.searches
        ],
        "fetches": [
            {**asdict(trace), "call": asdict(trace.call)} for trace in result.fetches
        ],
        "raw_model_response": result.raw_model_payload(),
        "model_cost_usd": result.model_cost_usd,
    }
