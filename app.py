import json
from pathlib import Path

import streamlit as st

from ask_chatbot_openrouter import ChatBot


PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = PROJECT_ROOT / "ask_config_openrouter.json"

if "chatbot" not in st.session_state:
    st.session_state.chatbot = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "model_selected" not in st.session_state:
    st.session_state.model_selected = False
if "selected_model_id" not in st.session_state:
    st.session_state.selected_model_id = None
if "selected_model_name" not in st.session_state:
    st.session_state.selected_model_name = None
if "pending_query" not in st.session_state:
    st.session_state.pending_query = None

st.set_page_config(page_title="STEP-RAG", page_icon="⚙️", layout="wide")
st.title("⚙️ STEP-RAG")
st.caption("Ask engineering questions grounded in the geometry of STEP CAD files.")
if st.session_state.model_selected:
    st.markdown(f"**🧠 Model in Use:** `{st.session_state.selected_model_name}`")

# Load LLM models from config
with CONFIG_PATH.open("r", encoding="utf-8") as f:
    config_data = json.load(f)
model_options = config_data["llm_model"]["models"]
model_names = list(model_options.keys())

# Step 1: Model Selection (One-time)
if not st.session_state.model_selected:
    selected_model_name = st.selectbox("Choose an LLM model", model_names)
    if st.button("✅ Confirm Model and Start Chat"):
        with st.spinner("Loading model and initializing..."):
            st.session_state.selected_model_id = model_options[selected_model_name]
            try:
                chatbot = ChatBot(config_path=str(CONFIG_PATH))
                chatbot.model_id = st.session_state.selected_model_id
            except (OSError, ValueError) as exc:
                st.error(str(exc))
                st.stop()
            st.session_state.chatbot = chatbot
            st.session_state.model_selected = True
            st.session_state.selected_model_name = selected_model_name
        st.success(f"✅ Model '{selected_model_name}' selected. You may now chat.")
        st.rerun()
    else:
        st.warning("Please confirm your model selection to begin.")
        st.stop()

if query := st.chat_input("Ask a question..."):
    st.session_state.pending_query = query
    st.session_state.chat_history.append({
        "question": query,
        "answer": "",
        "metrics": "",
        "sources": ""
    })
    st.rerun()

for i, entry in enumerate(st.session_state.chat_history):
    with st.chat_message("user"):
        st.markdown(entry["question"])
    if entry["answer"]:
        with st.chat_message("assistant"):
            st.markdown(entry["answer"])
            if entry.get("metrics"):
                st.markdown(entry["metrics"])
            if entry.get("sources"):
                st.markdown(entry["sources"])

if st.session_state.pending_query:
    with st.spinner("Generating answer..."):
        try:
            result = st.session_state.chatbot.ask(st.session_state.pending_query, return_score=True)
        except Exception as exc:
            st.session_state.chat_history[-1]["answer"] = f"Unable to generate an answer: {exc}"
            st.session_state.pending_query = None
            st.rerun()

    if isinstance(result, tuple):
        if len(result) == 5:
            answer, avg_score, k, cosine_sim, sources = result
        elif len(result) == 4:
            answer, avg_score, k, cosine_sim = result
            sources = []
        else:
            answer, avg_score, k = result
            cosine_sim = None
            sources = []
    else:
        answer = result
        avg_score = k = cosine_sim = None
        sources = []

    score_display = ""
    if avg_score is not None and k is not None:
        score_display = f"`Average Vector relevance scores: {avg_score:.4f} (k={k})`"
        if cosine_sim is not None:
            score_display += f" · `Answer-context cosine similarity: {cosine_sim:.4f}`"

    sources_md = ""
    if sources:
        sources_md = "**References used:**\n\n" + "\n".join(
            f"- {source}" for source in sources
        )

    st.session_state.chat_history[-1]["answer"] = answer
    st.session_state.chat_history[-1]["metrics"] = score_display
    st.session_state.chat_history[-1]["sources"] = sources_md

    st.session_state.pending_query = None

    st.rerun()
