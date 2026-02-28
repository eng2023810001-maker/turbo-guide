## Drilling AI Chatbot (Streamlit)

### Run
```bash
pip install streamlit faiss-cpu sentence-transformers langchain langchain-community pandas matplotlib seaborn numpy
streamlit run streamlit_app.py
```

### Important
- `streamlit_app.py` now matches your provided UI design.
- No design changes are required to use the chatbot backend.
- Chat integration uses `from chat_logic import DrillingAssistant` and calls `assistant.chat_streamlit(user_text)`.

### Backend files
- `drilling_chatbot.py`: retrieval, Arabic/English behavior, numeric reasoning, well summaries.
- `chat_logic.py`: adapter class expected by your UI (`DrillingAssistant.chat_streamlit`).

### Configuration defaults
- Base data directory: `D:\Database\DataPipeline_Renamed`
- LLM: `qwen3:8b`
- Embeddings: `all-MiniLM-L6-v2`


### Troubleshooting
- If you see `cannot import name DrillingAssistant from chat_logic`, ensure the file is named `chat_logic.py` and contains `class DrillingAssistant`.
- If chatbot initialization fails, install dependencies and verify VectorDB files:
  - `pip install faiss-cpu sentence-transformers langchain langchain-community pandas`
