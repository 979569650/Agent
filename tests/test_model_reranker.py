import os
import unittest

from infra.retrieval import model_reranker


class TestModelReranker(unittest.TestCase):
    def test_disabled_reranker_keeps_order(self):
        os.environ["RERANKER_ENABLED"] = "false"
        model_reranker._get_reranker.cache_clear()

        items = ["a", "b", "c"]
        out = model_reranker.rerank_with_model("test", items)
        self.assertEqual(out, items)


if __name__ == "__main__":
    unittest.main()
