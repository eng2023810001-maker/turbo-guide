import sys
import types
import unittest


# Lightweight stubs so helper-unit-tests can run without heavy runtime deps.
if "faiss" not in sys.modules:
    faiss_stub = types.ModuleType("faiss")
    faiss_stub.read_index = lambda *args, **kwargs: None
    sys.modules["faiss"] = faiss_stub

if "sentence_transformers" not in sys.modules:
    st_mod = types.ModuleType("sentence_transformers")

    class _SentenceTransformer:
        def __init__(self, *args, **kwargs):
            pass

        def encode(self, texts):
            return [[0.0] * 4 for _ in texts]

    st_mod.SentenceTransformer = _SentenceTransformer
    sys.modules["sentence_transformers"] = st_mod

if "langchain_community.chat_models" not in sys.modules:
    lc_mod = types.ModuleType("langchain_community.chat_models")

    class _ChatOllama:
        def __init__(self, *args, **kwargs):
            pass

        def invoke(self, *_args, **_kwargs):
            class _Resp:
                content = "ok"

            return _Resp()

    lc_mod.ChatOllama = _ChatOllama
    sys.modules["langchain_community.chat_models"] = lc_mod


if "pandas" not in sys.modules:
    pd_mod = types.ModuleType("pandas")

    class _Series(list):
        def replace(self, *_args, **_kwargs):
            return self

        def dropna(self):
            return self

        def value_counts(self):
            class _Counts(dict):
                def head(self, _n):
                    return self

                def to_dict(self):
                    return dict(self)

            counts = _Counts()
            for item in self:
                if item:
                    counts[item] = counts.get(item, 0) + 1
            return counts

    pd_mod.Series = _Series
    pd_mod.NA = None
    sys.modules["pandas"] = pd_mod

if "langchain_core.messages" not in sys.modules:
    msg_mod = types.ModuleType("langchain_core.messages")

    class _Msg:
        def __init__(self, content):
            self.content = content

    msg_mod.HumanMessage = _Msg
    msg_mod.SystemMessage = _Msg
    sys.modules["langchain_core.messages"] = msg_mod

from drilling_chatbot import AdvancedDrillingAssistant, SafeCalculator
from chat_logic import DrillingAssistant


class TestSafeCalculator(unittest.TestCase):
    def test_basic_math(self):
        calc = SafeCalculator()
        self.assertEqual(calc.evaluate("2+3*4"), 14.0)

    def test_invalid_math(self):
        calc = SafeCalculator()
        self.assertIsNone(calc.evaluate("import os"))


class TestParsingHelpers(unittest.TestCase):
    def test_extract_well_name_arabic(self):
        self.assertEqual(AdvancedDrillingAssistant._extract_well_name("ما حالة بئر 3"), "WELL-03")

    def test_extract_well_name_english(self):
        self.assertEqual(AdvancedDrillingAssistant._extract_well_name("well 12 status"), "WELL-12")

    def test_normalize_query_synonyms(self):
        q = AdvancedDrillingAssistant._normalize_query("What is depth and الوقت الضائع?")
        self.assertIn("md", q)
        self.assertIn("npt_net_time", q)


class TestChatLogicAdapter(unittest.TestCase):
    def test_chat_streamlit_delegates(self):
        assistant = DrillingAssistant()
        out = assistant.chat_streamlit("hello")
        self.assertIsInstance(out, str)
        self.assertEqual(assistant.ask("hello"), out)


if __name__ == "__main__":
    unittest.main()
