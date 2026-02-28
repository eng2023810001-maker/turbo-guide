from dataclasses import dataclass
from typing import Optional


@dataclass
class ChatLogicConfig:
    base_dir: str = r"D:\Database\DataPipeline_Renamed"
    llm_model_name: str = "qwen3:8b"
    embedding_model_name: str = "all-MiniLM-L6-v2"
    top_k: int = 8
    min_similarity: float = 0.35
    temperature: float = 0.05


class DrillingAssistant:
    """
    Lightweight adapter imported by Streamlit UI.
    Heavy dependencies are loaded lazily to avoid import-time crashes.
    """

    def __init__(self, config: Optional[ChatLogicConfig] = None):
        cfg = config or ChatLogicConfig()
        self._backend = None
        self._init_error = None

        try:
            from drilling_chatbot import AdvancedDrillingAssistant, AssistantConfig

            backend_cfg = AssistantConfig(
                base_dir=cfg.base_dir,
                llm_model_name=cfg.llm_model_name,
                embedding_model_name=cfg.embedding_model_name,
                top_k=cfg.top_k,
                min_similarity=cfg.min_similarity,
                temperature=cfg.temperature,
            )
            self._backend = AdvancedDrillingAssistant(backend_cfg)
        except Exception as exc:
            self._init_error = exc

    def chat_streamlit(self, user_query: str) -> str:
        if self._backend is None:
            return (
                "❌ تعذر تهيئة مساعد الحفر. تأكد من تثبيت المتطلبات (faiss / sentence-transformers / langchain) "
                f"ووجود ملفات VectorDB.\nDetails: {self._init_error}"
            )
        return self._backend.chat_response(user_query)

    def ask(self, user_query: str) -> str:
        return self.chat_streamlit(user_query)


__all__ = ["DrillingAssistant", "ChatLogicConfig"]
