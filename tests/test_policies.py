import unittest

from core.domain.policies import should_use_retrieval, should_force_vlm_only


class TestPolicies(unittest.TestCase):
    def test_smalltalk_should_skip_retrieval(self):
        self.assertFalse(should_use_retrieval("你好"))
        self.assertFalse(should_use_retrieval("hello"))

    def test_retrieval_intent_should_enable_retrieval(self):
        self.assertTrue(should_use_retrieval("帮我检索本地笔记里的 LangGraph 总结"))
        self.assertTrue(should_use_retrieval("看一下身份证图片内容"))

    def test_vlm_only_markers(self):
        self.assertTrue(should_force_vlm_only("不要ocr，看图"))
        self.assertFalse(should_force_vlm_only("请总结这篇笔记"))


if __name__ == "__main__":
    unittest.main()
