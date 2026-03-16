import unittest

from infra.retrieval.search_pipeline import build_ranked_context


class TestSearchPipeline(unittest.TestCase):
    def test_build_ranked_context_prefers_image(self):
        context = build_ranked_context(
            query="看身份证图片",
            vlm_only=False,
            text_hits=["[文本来源]\n普通内容"],
            image_note_hits=["[图片文本来源]\n身份证语义"],
            image_hits=["[图片来源]\nidcard-back.jpg"],
            k=3,
        )
        self.assertIn("[图片文本来源]", context)

    def test_build_ranked_context_vlm_only_hint(self):
        context = build_ranked_context(
            query="不要ocr 看图",
            vlm_only=True,
            text_hits=["[文本来源]\n普通内容"],
            image_note_hits=[],
            image_hits=["[图片来源]\nidcard-back.jpg"],
            k=3,
        )
        self.assertIn("仅用图片语义理解", context)


if __name__ == "__main__":
    unittest.main()
