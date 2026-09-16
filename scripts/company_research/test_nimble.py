import os
import unittest
from unittest.mock import patch
import nimble
from company_research import vendors

class NimbleTests(unittest.TestCase):
    def test_search_and_extract(self):
        for depth in ("lite", "standard"):
            key = f"nimble_{depth}"
            raw = vendors.VendorCall("ok", 1, {}, {"results": [{"url": "https://a.test", "description": "evidence", "content": "FULL PAGE"}, {"url": "https://a.test"}, {}]}, None, [], None)
            with patch.dict(os.environ, {"NIMBLE_API_KEY": "test", "NIMBLE_EXTRACT_USD_PER_REQUEST": ""}), patch.object(vendors, "_request", return_value=raw) as call, patch.object(vendors, "_assert_public_url"):
                result = vendors.search(key, "query", max_results=1)
                self.assertEqual(call.call_args.kwargs["body"], {"query": "query", "search_depth": depth, "full_content": False, "focus": "general", "max_results": 1})
                self.assertEqual(result.hits[0]["snippet"], "evidence")
                self.assertEqual(result.cost_usd, nimble.SEARCH_PRICES[depth])
                raw.raw_response = {"status": "success", "data": {"markdown": "abcdef"}}
                result = vendors.fetch(key, "https://a.test", objective="research", max_chars=3)
                self.assertEqual(call.call_args.kwargs["url"], nimble.EXTRACT_URL)
                self.assertEqual(call.call_args.kwargs["body"], {"url": "https://a.test", "formats": ["markdown"]})
                self.assertEqual(result.page["text"], "abc")
                self.assertTrue(result.page["truncated"])
                self.assertIsNone(result.cost_usd)
                with patch.dict(os.environ, {"NIMBLE_EXTRACT_USD_PER_REQUEST": "0.002"}):
                    self.assertEqual(vendors.fetch(key, "https://a.test", objective="research").cost_usd, .002)
