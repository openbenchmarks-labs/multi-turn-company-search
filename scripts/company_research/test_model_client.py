from __future__ import annotations

import unittest
from unittest.mock import patch

from company_research.model_client import build_openai_client


class ModelClientTest(unittest.TestCase):
    def test_builds_standard_openai_client_from_api_key(self) -> None:
        calls = []

        def client_class(**kwargs):
            calls.append(kwargs)
            return object()

        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}, clear=True):
            build_openai_client(client_class, timeout=12, max_retries=3)
        self.assertEqual(
            calls,
            [{"api_key": "test-key", "timeout": 12, "max_retries": 3}],
        )

    def test_api_key_is_required(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "OPENAI_API_KEY"):
                build_openai_client(lambda **kwargs: object())


if __name__ == "__main__":
    unittest.main()
