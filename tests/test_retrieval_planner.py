import unittest

from core.domain.retrieval_planner import rewrite_query, infer_retrieval_filters, rerank_hits


class TestRetrievalPlanner(unittest.TestCase):
    def test_rewrite_query_adds_synonyms(self):
        q = rewrite_query("看一下身份证图片")
        self.assertIn("idcard", q.lower())

    def test_infer_filters(self):
        f = infer_retrieval_filters("看图像内容", vlm_only=True)
        self.assertTrue(f["prefer_image"])
        self.assertTrue(f["vlm_only"])

    def test_rerank_prefers_image(self):
        hits = [
            ("text", "[文本来源]\n这是普通笔记"),
            ("image_note", "[图片文本来源]\n这是身份证图片语义"),
        ]
        ranked = rerank_hits("身份证", hits, prefer_image=True)
        self.assertTrue(ranked[0].startswith("[图片文本来源]"))


if __name__ == "__main__":
    unittest.main()
