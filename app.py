#streamlit run app.py --server.headless false

import streamlit as st
import json
from ask_chatbot_openrouter import ChatBot  # changed import

# Init session state
# Safe Session State Initialization
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

st.set_page_config(page_title="LLM Chat with Context", layout="wide")
st.title("🤖 Chat with your PDF (RAG)")
if st.session_state.model_selected:
    st.markdown(f"**🧠 Model in Use:** `{st.session_state.selected_model_name}`")

# Load LLM models from config
with open("ask_config_openrouter.json", "r") as f:
    config_data = json.load(f)
model_options = config_data["llm_model"]["models"]
model_names = list(model_options.keys())

# Step 1: Model Selection (One-time)
if not st.session_state.model_selected:
    selected_model_name = st.selectbox("Choose an LLM model", model_names)
    if st.button("✅ Confirm Model and Start Chat"):
        with st.spinner("Loading model and initializing..."):
            st.session_state.selected_model_id = model_options[selected_model_name]
            st.session_state.chatbot = ChatBot(config_path="ask_config_openrouter.json")
            st.session_state.chatbot.model_id = st.session_state.selected_model_id
            st.session_state.model_selected = True
            st.session_state.selected_model_name = selected_model_name
        st.success(f"✅ Model '{selected_model_name}' selected. You may now chat.")
        st.rerun()  # Ensure fresh UI
    else:
        st.warning("Please confirm your model selection to begin.")
        st.stop()

# Step 2: Chat Interface
# Update chat_input() logic
if query := st.chat_input("Ask a question..."):
    st.session_state.pending_query = query
    # Show immediately in history (no waiting message)
    st.session_state.chat_history.append({
        "question": query,
        "answer": "",
        "metrics": "",
        "sources": ""
    })
    st.rerun()

# Show chat history (user + assistant)
for i, entry in enumerate(st.session_state.chat_history):
    with st.chat_message("user"):
        st.markdown(entry["question"])
    if entry["answer"]:
        with st.chat_message("assistant"):
            st.markdown(entry["answer"])
            if entry.get("metrics"):
                st.markdown(entry["metrics"], unsafe_allow_html=True)
            if entry.get("sources"):
                st.markdown(entry["sources"], unsafe_allow_html=True)

# If there's a pending query, now process it and update the last message
if st.session_state.pending_query:
    with st.spinner("Generating answer..."):
        result = st.session_state.chatbot.ask(st.session_state.pending_query, return_score=True)

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

    # Format result metrics
    score_display = ""
    if avg_score is not None and k is not None:
        score_display = f"`Average Vector relevance scores: {avg_score:.4f} (k={k})`"
        if cosine_sim is not None:
            score_display += f" &nbsp; `Answer to Chunk Cosine Similarity: {cosine_sim:.4f}`"

    # Format sources
    sources_md = ""
    if sources:
        sources_md = (
            "<div style='color: grey; font-size: 0.85em; margin-top: 0.5em;'>"
            "<b>References used:</b><br>"
            + "<br>".join([f"{src}" for src in sources]) +
            "</div>"
        )

    # Replace the last message with actual answer
    st.session_state.chat_history[-1]["answer"] = answer
    st.session_state.chat_history[-1]["metrics"] = score_display
    st.session_state.chat_history[-1]["sources"] = sources_md

    # Clear the pending flag
    st.session_state.pending_query = None

    st.rerun()
