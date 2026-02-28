import ast
import operator
import os
import pickle
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import faiss
import pandas as pd
from langchain_community.chat_models import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from sentence_transformers import SentenceTransformer


@dataclass
class AssistantConfig:
    base_dir: str = r"D:\Database\DataPipeline_Renamed"
    llm_model_name: str = "qwen3:8b"
    embedding_model_name: str = "all-MiniLM-L6-v2"
    top_k: int = 8
    min_similarity: float = 0.35
    temperature: float = 0.05

    @property
    def vector_db_dir(self) -> str:
        return os.path.join(self.base_dir, "VectorDB")


class SafeCalculator:
    """Safe arithmetic evaluator for numeric expressions."""

    _ops = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
        ast.Mod: operator.mod,
    }

    def evaluate(self, expression: str) -> Optional[float]:
        expression = expression.strip()
        if not expression:
            return None

        try:
            node = ast.parse(expression, mode="eval").body
            return float(self._eval(node))
        except Exception:
            return None

    def _eval(self, node: ast.AST) -> float:
        if isinstance(node, ast.Num):
            return float(node.n)
        if isinstance(node, ast.UnaryOp) and type(node.op) in self._ops:
            return self._ops[type(node.op)](self._eval(node.operand))
        if isinstance(node, ast.BinOp) and type(node.op) in self._ops:
            left = self._eval(node.left)
            right = self._eval(node.right)
            return self._ops[type(node.op)](left, right)
        raise ValueError("Unsupported expression")


class AdvancedDrillingAssistant:
    """
    Back-end assistant logic only.
    Can be plugged into any existing Streamlit design without changing UI components.
    """

    def __init__(self, config: Optional[AssistantConfig] = None):
        self.config = config or AssistantConfig()
        self.calculator = SafeCalculator()

        self.embedding_model = SentenceTransformer(self.config.embedding_model_name)
        self.llm = ChatOllama(model=self.config.llm_model_name, temperature=self.config.temperature)

        self.vector_index = None
        self.metadata: List[Dict[str, Any]] = []
        self._load_vector_db()

    # ------------------------
    # Data and parsing helpers
    # ------------------------
    def _load_vector_db(self) -> None:
        index_path = os.path.join(self.config.vector_db_dir, "drilling_data.index")
        meta_path = os.path.join(self.config.vector_db_dir, "drilling_metadata.pkl")

        if not (os.path.exists(index_path) and os.path.exists(meta_path)):
            return

        self.vector_index = faiss.read_index(index_path)
        with open(meta_path, "rb") as f:
            self.metadata = pickle.load(f)

    @staticmethod
    def _contains_arabic(text: str) -> bool:
        return bool(re.search(r"[\u0600-\u06FF]", text))

    @staticmethod
    def _normalize_digits(text: str) -> str:
        return text.translate(str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789"))

    @classmethod
    def _normalize_query(cls, query: str) -> str:
        q = cls._normalize_digits(query.lower())
        synonyms = {
            "measured depth": "md",
            "depth": "md",
            "العمق": "md",
            "الوقت الضائع": "npt_net_time",
            "npt": "npt_net_time",
            "نوع الفشل": "failure",
            "فشل": "failure",
        }
        for old, new in synonyms.items():
            q = q.replace(old, new)
        return q

    @classmethod
    def _extract_well_name(cls, query: str) -> Optional[str]:
        q = cls._normalize_digits(query.lower())
        patterns = [
            r"(?:well|بئر|البئر)[\s\-_]*(?:رقم\s*)?(\d+)",
            r"\b(?:w|well)[\s\-_]?(\d+)\b",
        ]
        for pattern in patterns:
            match = re.search(pattern, q)
            if match:
                return f"WELL-{int(match.group(1)):02d}"
        return None

    @classmethod
    def _extract_math_expression(cls, query: str) -> Optional[str]:
        q = cls._normalize_digits(query)
        match = re.search(r"([\d\s\(\)\.\+\-\*\/\%]+)", q)
        if not match:
            return None
        expr = match.group(1).strip()
        if expr and re.search(r"[\+\-\*\/\%]", expr):
            return expr
        return None

    # ------------------------
    # Retrieval and summaries
    # ------------------------
    def _similarity_search(self, query: str, top_k: Optional[int] = None) -> List[Tuple[float, Dict[str, Any]]]:
        if self.vector_index is None:
            return []

        k = top_k or self.config.top_k
        query_vec = self.embedding_model.encode([query]).astype("float32")
        distances, indices = self.vector_index.search(query_vec, k)

        results: List[Tuple[float, Dict[str, Any]]] = []
        for distance, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            similarity = 1 / (1 + float(distance))
            if similarity < self.config.min_similarity:
                continue
            results.append((similarity, self.metadata[idx]))
        return results

    @staticmethod
    def _to_float(value: Any) -> float:
        try:
            return float(str(value).replace(",", "").strip())
        except Exception:
            return 0.0

    def _compute_well_summary(self, well_name: str) -> Dict[str, Any]:
        rows = [m for m in self.metadata if str(m.get("well_name", "")).upper() == well_name.upper()]
        if not rows:
            return {}

        max_depth = max((self._to_float(r.get("md_to")) for r in rows), default=0.0)
        total_npt = sum(self._to_float(r.get("npt_net_time")) for r in rows)
        top_failures = (
            pd.Series([str(r.get("equip_fail_type", "")).strip() for r in rows])
            .replace("", pd.NA)
            .dropna()
            .value_counts()
            .head(5)
            .to_dict()
        )

        return {
            "well_name": well_name,
            "records": len(rows),
            "max_depth_md": max_depth,
            "total_npt_hours": total_npt,
            "top_failures": top_failures,
        }

    def _is_summary_request(self, query: str) -> bool:
        q = self._normalize_query(query)
        keywords = ["summary", "ملخص", "overview", "حالة", "report"]
        return any(k in q for k in keywords)

    def _render_context(self, docs: List[Tuple[float, Dict[str, Any]]]) -> str:
        lines = []
        for score, meta in docs:
            lines.append(
                (
                    f"Score:{score:.3f} | Well:{meta.get('well_name', '')} | Date:{meta.get('date_report', '')} | "
                    f"MD:{meta.get('md_to', '')} | NPT:{meta.get('npt_net_time', '')} | "
                    f"Failure:{meta.get('equip_fail_type', '')} - {meta.get('failure_title', '')} | "
                    f"Activity:{meta.get('comment_summary', '')} {meta.get('activity_memo', '')}"
                )
            )
        return "\n".join(lines)

    # ------------------------
    # Public interface for Streamlit UI integration
    # ------------------------
    def chat_response(self, user_query: str) -> str:
        """Compatibility method: keep your UI design unchanged; call this method from your existing components."""
        return self.answer(user_query)

    def answer(self, user_query: str) -> str:
        if not user_query or not user_query.strip():
            return "يرجى إدخال سؤال واضح." if self._contains_arabic(user_query or "") else "Please provide a valid question."

        is_ar = self._contains_arabic(user_query)
        lang_instruction = "Arabic" if is_ar else "English"

        expr = self._extract_math_expression(user_query)
        if expr:
            result = self.calculator.evaluate(expr)
            if result is not None:
                return f"نتيجة العملية هي: {result:g}" if is_ar else f"Calculation result: {result:g}"

        well_name = self._extract_well_name(user_query)

        if well_name and self._is_summary_request(user_query):
            summary = self._compute_well_summary(well_name)
            if not summary:
                return f"لا توجد بيانات للبئر {well_name}." if is_ar else f"No records found for {well_name}."

            summary_prompt = (
                "Create a concise engineering well summary using ONLY this data and mention key numbers explicitly. "
                f"Language must be {lang_instruction}.\n"
                f"DATA: {summary}"
            )
            return self.llm.invoke([HumanMessage(content=summary_prompt)]).content

        normalized_query = self._normalize_query(user_query)
        search_query = f"{normalized_query} {well_name or ''}".strip()
        docs = self._similarity_search(search_query)

        if not docs:
            return (
                "لم أجد سياقاً كافياً في قاعدة البيانات للإجابة بدقة."
                if is_ar
                else "I could not find sufficient database context for an accurate answer."
            )

        context = self._render_context(docs)
        system = SystemMessage(
            content=(
                "You are a senior drilling engineer assistant. Use only the provided context. "
                "Do not guess. If evidence is insufficient, say so clearly. "
                "Always answer in the same language as the question. "
                "For numeric requests, output value + unit and keep precision."
            )
        )
        user = HumanMessage(
            content=(
                f"Question language: {lang_instruction}\n"
                f"Question: {user_query}\n\n"
                f"Retrieved context:\n{context}\n\n"
                "Return a direct answer with brief evidence bullets from context."
            )
        )
        return self.llm.invoke([system, user]).content


if __name__ == "__main__":
    assistant = AdvancedDrillingAssistant()
    print(assistant.chat_response("ما هو ملخص بئر 1؟"))
