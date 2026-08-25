"""Hardened local webpage fetch adapted from self-serve-backend scrapers.

The fast path is a bounded streaming HTTP request. JavaScript-heavy or blocked
pages fall back to a concurrency-limited Playwright Chromium session. Every
top-level URL and browser request is checked against private/local addresses.
"""

from __future__ import annotations

import ipaddress
import os
import re
import socket
import threading
import time
from html.parser import HTMLParser
from typing import Any, Callable
from urllib.parse import urljoin, urlparse

import requests


DEFAULT_MAX_CHARS = 12_000
DEFAULT_MAX_BYTES = 1_500_000
DEFAULT_MAX_REDIRECTS = 5
DEFAULT_NAVIGATION_TIMEOUT_MS = 12_000
DEFAULT_SETTLE_MS = 2_000
DEFAULT_MAX_BROWSERS = 4
MIN_USEFUL_TEXT_CHARS = 200
ALLOWED_CONTENT_TYPES = ("text/html", "application/xhtml+xml", "text/plain")
REDIRECT_STATUS = {301, 302, 303, 307, 308}


def _int_env(name: str, default: int, minimum: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return default
    return value if value >= minimum else default


MAX_CONCURRENT_BROWSERS = _int_env(
    "COMPANY_RESEARCH_FETCH_MAX_BROWSERS", DEFAULT_MAX_BROWSERS, 1
)
NAVIGATION_TIMEOUT_MS = _int_env(
    "COMPANY_RESEARCH_FETCH_NAV_TIMEOUT_MS", DEFAULT_NAVIGATION_TIMEOUT_MS, 1
)
SETTLE_MS = _int_env("COMPANY_RESEARCH_FETCH_SETTLE_MS", DEFAULT_SETTLE_MS, 0)
_browser_semaphore = threading.BoundedSemaphore(MAX_CONCURRENT_BROWSERS)


def validate_playwright_installation() -> None:
    """Launch Chromium once so paid execution cannot discover a broken fallback late."""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise RuntimeError(
            "custom web fetch requires Playwright; install requirements and run "
            "`playwright install chromium`"
        ) from exc
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            browser.close()
    except Exception as exc:
        raise RuntimeError(
            "custom web fetch Chromium preflight failed; run `playwright install chromium` "
            "and verify the VM browser dependencies"
        ) from exc


class ReadableTextParser(HTMLParser):
    ignored = {"script", "style", "noscript", "svg", "canvas", "template", "nav", "footer"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.ignored_depth = 0
        self.in_title = False
        self.title_parts: list[str] = []
        self.text_parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        lowered = tag.lower()
        if lowered in self.ignored:
            self.ignored_depth += 1
        if lowered == "title" and not self.ignored_depth:
            self.in_title = True

    def handle_endtag(self, tag: str) -> None:
        lowered = tag.lower()
        if lowered == "title":
            self.in_title = False
        if lowered in self.ignored and self.ignored_depth:
            self.ignored_depth -= 1

    def handle_data(self, data: str) -> None:
        if self.ignored_depth:
            return
        value = data.strip()
        if not value:
            return
        if self.in_title:
            self.title_parts.append(value)
        self.text_parts.append(value)


def html_to_text(raw: str) -> tuple[str, str]:
    parser = ReadableTextParser()
    parser.feed(raw)
    title = re.sub(r"\s+", " ", " ".join(parser.title_parts)).strip()
    text = re.sub(r"\s+", " ", " ".join(parser.text_parts)).strip()
    return title, text


def assert_public_http_url(
    url: str,
    *,
    resolver: Callable[..., list[tuple[Any, ...]]] = socket.getaddrinfo,
) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("web_fetch only supports absolute http(s) URLs")
    if parsed.username or parsed.password:
        raise ValueError("web_fetch does not allow credentials in URLs")
    if parsed.hostname.lower() == "localhost":
        raise ValueError("web_fetch blocks localhost and private-network URLs")
    try:
        addresses = resolver(parsed.hostname, parsed.port or (443 if parsed.scheme == "https" else 80))
    except socket.gaierror as exc:
        raise ValueError(f"could not resolve fetch hostname: {parsed.hostname}") from exc
    if not addresses:
        raise ValueError(f"could not resolve fetch hostname: {parsed.hostname}")
    for address in addresses:
        ip = ipaddress.ip_address(address[4][0])
        if not ip.is_global:
            raise ValueError("web_fetch blocks localhost and private-network URLs")


def _page_result(
    *,
    requested_url: str,
    final_url: str,
    title: str,
    text: str,
    status_code: int | None,
    content_type: str,
    bytes_read: int,
    max_chars: int,
    max_bytes: int,
    started: float,
    provider: str,
) -> dict[str, Any]:
    clean = re.sub(r"\s+", " ", text).strip()
    return {
        "requested_url": requested_url,
        "final_url": final_url,
        "title": title.strip(),
        "text": clean[:max_chars],
        "status_code": status_code,
        "content_type": content_type,
        "bytes_read": bytes_read,
        "truncated": len(clean) > max_chars or bytes_read >= max_bytes,
        "latency_ms": round((time.perf_counter() - started) * 1000),
        "fetch_provider": provider,
    }


def fetch_http(
    url: str,
    *,
    max_chars: int = DEFAULT_MAX_CHARS,
    max_bytes: int = DEFAULT_MAX_BYTES,
    timeout: tuple[int, int] = (10, 25),
    session_factory: Callable[[], Any] = requests.Session,
) -> dict[str, Any]:
    """Stream a page with bounded bytes and validate each redirect target."""
    started = time.perf_counter()
    current_url = url
    session = session_factory()
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,text/plain;q=0.9,*/*;q=0.1",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate",
    }
    try:
        for redirect_count in range(DEFAULT_MAX_REDIRECTS + 1):
            assert_public_http_url(current_url)
            response = session.get(
                current_url, headers=headers, timeout=timeout, allow_redirects=False, stream=True
            )
            if response.status_code in REDIRECT_STATUS:
                location = response.headers.get("Location")
                response.close()
                if not location:
                    raise RuntimeError(f"redirect without Location from {current_url}")
                if redirect_count >= DEFAULT_MAX_REDIRECTS:
                    raise RuntimeError("web_fetch redirect limit exceeded")
                current_url = urljoin(current_url, location)
                continue
            response.raise_for_status()
            content_type = response.headers.get("Content-Type", "").split(";", 1)[0].strip().lower()
            if content_type and not any(content_type.startswith(kind) for kind in ALLOWED_CONTENT_TYPES):
                response.close()
                raise RuntimeError(f"unsupported web_fetch content type: {content_type}")
            chunks: list[bytes] = []
            bytes_read = 0
            for chunk in response.iter_content(chunk_size=16_384):
                if not chunk:
                    continue
                remaining = max_bytes - bytes_read
                if remaining <= 0:
                    break
                chunks.append(chunk[:remaining])
                bytes_read += min(len(chunk), remaining)
                if bytes_read >= max_bytes:
                    break
            encoding = response.encoding or "utf-8"
            raw = b"".join(chunks).decode(encoding, errors="replace")
            status_code = response.status_code
            response.close()
            if content_type == "text/plain":
                title, text = "", raw
            else:
                title, text = html_to_text(raw)
            return _page_result(
                requested_url=url, final_url=current_url, title=title, text=text,
                status_code=status_code, content_type=content_type, bytes_read=bytes_read,
                max_chars=max_chars, max_bytes=max_bytes, started=started, provider="custom_http",
            )
        raise RuntimeError("web_fetch redirect limit exceeded")  # pragma: no cover
    finally:
        close = getattr(session, "close", None)
        if callable(close):
            close()


def _browser_route(route: Any) -> None:
    request = route.request
    if request.resource_type in {"image", "media", "font"}:
        route.abort()
        return
    try:
        assert_public_http_url(request.url)
    except (ValueError, OSError):
        route.abort()
        return
    route.continue_()


def fetch_playwright(
    url: str,
    *,
    max_chars: int = DEFAULT_MAX_CHARS,
    max_bytes: int = DEFAULT_MAX_BYTES,
) -> dict[str, Any]:
    """Render a JavaScript page with bounded concurrent Chromium instances."""
    assert_public_http_url(url)
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:  # pragma: no cover - exercised in deployed runner
        raise RuntimeError(
            "Playwright fallback requires the playwright package and `playwright install chromium`"
        ) from exc
    started = time.perf_counter()
    with _browser_semaphore, sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        try:
            context = browser.new_context(
                accept_downloads=False,
                service_workers="block",
                user_agent=(
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
                ),
            )
            try:
                page = context.new_page()
                page.route("**/*", _browser_route)
                page.set_default_navigation_timeout(NAVIGATION_TIMEOUT_MS)
                response = page.goto(url, wait_until="domcontentloaded", timeout=NAVIGATION_TIMEOUT_MS)
                page.wait_for_timeout(SETTLE_MS)
                final_url = page.url
                assert_public_http_url(final_url)
                content_type = ""
                status_code = None
                if response is not None:
                    status_code = response.status
                    content_type = response.headers.get("content-type", "").split(";", 1)[0].lower()
                    if status_code >= 400:
                        raise RuntimeError(f"browser fetch returned HTTP {status_code}")
                    if content_type and not any(content_type.startswith(kind) for kind in ALLOWED_CONTENT_TYPES):
                        raise RuntimeError(f"unsupported browser fetch content type: {content_type}")
                raw = page.content()
                encoded = raw.encode("utf-8")
                bounded = encoded[:max_bytes].decode("utf-8", errors="replace")
                title, text = html_to_text(bounded)
                return _page_result(
                    requested_url=url, final_url=final_url, title=title, text=text,
                    status_code=status_code, content_type=content_type or "text/html",
                    bytes_read=min(len(encoded), max_bytes), max_chars=max_chars,
                    max_bytes=max_bytes, started=started, provider="custom_playwright",
                )
            finally:
                context.close()
        finally:
            browser.close()


def fetch_page(
    url: str,
    *,
    max_chars: int = DEFAULT_MAX_CHARS,
    max_bytes: int = DEFAULT_MAX_BYTES,
    browser_fallback: bool = True,
) -> dict[str, Any]:
    """Use bounded HTTP first, then Playwright for failures or thin pages."""
    http_error: Exception | None = None
    try:
        page = fetch_http(url, max_chars=max_chars, max_bytes=max_bytes)
        if len(str(page.get("text") or "")) >= MIN_USEFUL_TEXT_CHARS or not browser_fallback:
            return page
    except Exception as exc:  # noqa: BLE001
        http_error = exc
        if not browser_fallback:
            raise
    try:
        return fetch_playwright(url, max_chars=max_chars, max_bytes=max_bytes)
    except Exception as browser_error:
        if http_error is not None:
            raise RuntimeError(
                f"HTTP fetch failed ({http_error}); Playwright fallback failed ({browser_error})"
            ) from browser_error
        raise RuntimeError(f"HTTP fetch returned thin text; Playwright fallback failed ({browser_error})") from browser_error
