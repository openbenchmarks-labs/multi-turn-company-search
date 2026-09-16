"""Offline search and native fetch contracts for You highlights/core."""
import os
import unittest
from unittest.mock import patch
from company_research import vendors


class YouContracts(unittest.TestCase):
    def test_search_and_separate_contents_fetch(self):
        payload = {"results": {"web": [{"url": "https://example.com", "snippets": ["fallback"],
                    "contents": {"highlights": ["Relevant evidence"]}}]}}
        for arm in ("you_highlights", "you_highlights_core"):
            with self.subTest(arm=arm), patch.dict(os.environ, {"YOU_API_KEY": "test-key"}):
                body = {"query": "Unchanged question?", "count": 8,
                        "extraction": {"extraction_mode": "highlights"}}
                if arm.endswith("_core"):
                    body["knowledge"] = "core"
                call_result = vendors.VendorCall(status="ok", latency_ms=1, raw_request={}, raw_response=payload,
                                                 error=None, attempts=[], cost_usd=None)
                with patch.object(vendors, "_request", return_value=call_result) as call:
                    result = vendors.search(arm, "Unchanged question?", max_results=8)
                    self.assertEqual(result.hits[0]["snippet"], "Relevant evidence")
                    self.assertEqual(call.call_args.kwargs["body"], body)
                    self.assertEqual(call.call_args.kwargs["url"], "https://ydc-index.io/v1/search")
                call_result = vendors.VendorCall(status="ok", latency_ms=1, raw_request={},
                    raw_response=[{"url": "https://example.com", "markdown": "Page"}],
                    error=None, attempts=[], cost_usd=None)
                with patch.object(vendors, "_request", return_value=call_result) as call:
                    result = vendors.fetch(arm, "https://example.com", objective="question")
                    self.assertEqual(result.page["text"], "Page")
                    self.assertEqual(call.call_args.kwargs["url"], "https://ydc-index.io/v1/contents")
                    self.assertEqual(call.call_args.kwargs["body"], {
                        "urls": ["https://example.com"], "formats": ["markdown", "metadata"]})
                self.assertIn(arm, vendors.DEFAULT_VENDOR_KEYS)
                self.assertTrue(vendors.VENDORS[arm].native_fetch)

    def test_retired_plain_search_not_in_roster(self):
        self.assertNotIn("you", vendors.VENDORS)


if __name__ == "__main__":
    unittest.main()
