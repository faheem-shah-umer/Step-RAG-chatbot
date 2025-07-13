# STEP RAG Chatbot

This project provides a Streamlit-based chatbot interface to interact with STEP (CAD) files using Retrieval-Augmented Generation (RAG) techniques. It integrates LangGraph, OpenRouter LLMs, and a local Qdrant vector store to support technical Q&A in the context of tolerance management and mechanical design.

## 🚀 Features

- Parses and analyzes STEP files (3D CAD) for shape, features, holes, bounding boxes, volumes, roles, and relationships.
- Converts extracted text data into embeddings using Sentence Transformers.
- Supports multiple OpenRouter LLMs (including multimodal models).
- Web-based interactive chatbot interface with model selection.
- Retrieval via Similarity Search or Max Marginal Relevance (MMR).
- Cosine similarity evaluation for answer-context relevance.
- Tracks chat history and provides source attribution.

## 📦 Components

- `app.py`: Streamlit app with model selection and chat interface.
- `Step2vstore.py`: Processes and analyzes STEP files and populates the Qdrant vector database.
- `ask_chatbot_openrouter.py`: Core chatbot logic, context retrieval, query generation, and LLM interaction.
- `ask_config_openrouter.json`: Settings for LLM models, vector store path, search method, and answer format.
- `config.json`: Contains STEP file input directory and vector DB path.
- `openrouter.env`: Stores the OpenRouter API key (not tracked in Git).

## ⚙️ Configuration

### `ask_config_openrouter.json`
- `llm_model.models`: Supported OpenRouter model IDs.
- `llm_model.selected`: Default LLM to load on startup.
- `search_method.selected`: Choose between similarity and MMR strategies.
- `answer_mode`: Prompting style guiding technical tone and behavior.

### `config.json`
- `data_sources.step.directory`: Folder containing STEP files.
- `vector_store.path`: Location of Qdrant vector DB for storing embeddings.

## 🧪 Quickstart

1. **Install requirements**

   ```bash
   pip install -r requirements.txt

2. **Set OpenRouter API key**

   Create `.env` file or use `openrouter.env`:
   ```env
   OPENROUTER_API_KEY=your-api-key
   ```

3. **Run the PDF uploader**
   ```bash
   streamlit run upload.py
   ```

4. **Run the chatbot interface**
   ```bash
   streamlit run app.py
   ```

## Notes

- Requires Python 3.9+
- Ensure vector DB path exists (or gets created)
- Avoid re-processing same PDFs via deduplication hash logic

## License

This project is licensed under the MIT License.