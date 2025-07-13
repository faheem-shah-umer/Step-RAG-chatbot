# as default ask_config.json. For Custom config file run - python indexing_vstore.py --config "file_Name.json"

import os
import json
import uuid
import argparse
from dotenv import load_dotenv
from collections import deque
from langchain_community.vectorstores import Qdrant
from langchain_huggingface import HuggingFaceEmbeddings
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage
import requests
from sentence_transformers import SentenceTransformer, util

class ChatBot:
    def __init__(self, config_path="ask_config_InterferenceClient.json"):
        self.load_config(config_path)
        self.init_llm()
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
        self.conversation_history = deque(maxlen=5)
        self.initialize_vector_store()
        self.setup_workflow()

    def load_config(self, config_path):
        with open(config_path, 'r') as file:
            self.config = json.load(file)
        self.vector_store_path = self.config['vector_store']['path']
        self.answer_mode = self.config.get('answer_mode')
        load_dotenv("openrouter.env")
        self.api_key = os.getenv("OPENROUTER_API_KEY")

    def init_llm(self):
        self.model_id = self.config["llm_model"]["selected"] 
        self.model_locked = False
        self.api_url = "https://openrouter.ai/api/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    def initialize_vector_store(self):
        if not os.path.exists(self.vector_store_path):
            print("Warning: Vector store path does not exist. Context retrieval may not work.")
            self.vstore = None
            return
        try:
            self.vstore = Qdrant.from_existing_collection(
                embedding=self.embeddings,
                path=self.vector_store_path,
                collection_name="all"
            )
        except Exception as e:
            print(f"Warning: Failed to load vector store: {e}\nContext retrieval may be limited.")
            self.vstore = None

    def setup_workflow(self):
        self.memory = MemorySaver()
        self.thread_id = uuid.uuid4()
        self.config["configurable"] = {"thread_id": self.thread_id}
        self.workflow = StateGraph(state_schema=MessagesState)
        self.workflow.add_edge(START, "model")
        self.workflow.add_node("model", self._call_model)
        self.app = self.workflow.compile(checkpointer=self.memory)

    
    def retrieve_context(self, query):
        if self.vstore is None:
            return "No context available (vector DB missing)."
        try:
            search_method = self.config.get("search_method", {}).get("selected", "similarity_search")
            print(f"🔍 Using search method: {search_method}")
            
            if search_method == "similarity_search":
                relevant_chunks = self.vstore.similarity_search_with_score(query, k=5)
            elif search_method == "max_marginal_relevance_search":
                relevant_chunks = self.vstore.max_marginal_relevance_search(query, k=5)
            elif search_method == "max_marginal_relevance_search_with_score_by_vector":
                query_embedding = self.embeddings.embed_query(query)
                relevant_chunks = self.vstore.max_marginal_relevance_search_with_score_by_vector(embedding=query_embedding,k=5)
            else:
                relevant_chunks = self.vstore.similarity_search_with_score(query, k=5)
            return relevant_chunks
        except Exception as e:
            return f"Error retrieving context: {str(e)}"

    def construct_query(self, user_query, context):
        history_lines = [f"Q{i+1}: {qa['question']}\n{qa['answer']}" for i, qa in enumerate(list(self.conversation_history)[-5:])]
        history = "\n---------------------\n".join(history_lines)
        history_section = f"Previous Conversations:\n{history}\n\n" if history else ""
        full_prompt = f"{history_section}Context:\n{context}\n\n{self.answer_mode}\n\nQuestion: {user_query}"
        print(f"\n======  PROMPT SENT TO LLM ======\n{full_prompt}\n===================================\n")
        return full_prompt

    def generate_response(self, input_text, print_model=True):
        payload = {
            "model": self.model_id,
            "messages": [
                {"role": "system", "content": "detailed thinking on"},
                {"role": "user", "content": input_text}
            ],
            "temperature": 0.7,
            "max_tokens": 1024,
            "provider": {
                "require_parameters": True,
                "allow_fallbacks": False
            }
        }
        response = requests.post(
            url=self.api_url,
            headers=self.headers,
            data=json.dumps(payload)
        )
        try:
            raw = response.json()
        except Exception:
            return "Error parsing response"
        if "error" in raw:
            return f"⚠️ Error: {raw['error']['message']}, \nplease try with another LLM Model again"
        used_model = raw.get("model", "unknown")
        if print_model:
            print(f"\n⚡ Model used by OpenRouter: {used_model}\n")
        return raw["choices"][0]["message"]["content"]

    def evaluate_metrics(self, question, answer, contexts):
        # Cosine Similarity, no LLM API call
        model = SentenceTransformer("all-MiniLM-L6-v2")
        answer_emb = model.encode(answer, convert_to_tensor=True)
        context_embs = model.encode(contexts, convert_to_tensor=True)
        similarity_score = util.cos_sim(answer_emb, context_embs).mean().item()

        # Print evaluation scores at the very bottom
        print("Evaluation Scores:")
        print(f"Cosine Similarity: {similarity_score:.4f}")
        return similarity_score

    def ask(self, query, return_score=False):
        context_chunks = self.retrieve_context(query)

        # Guard for error string from retrieve_context
        if isinstance(context_chunks, str) and context_chunks.startswith("Error retrieving context"):
            print(context_chunks)
            context_text = context_chunks
            modified_query = self.construct_query(query, context_text)
            response = self.generate_response(modified_query, print_model=True)
            self.conversation_history.append({"question": query, "answer": response})
            print(f"Q: {query}\nA: {response}\n")
            return response

        # Improved handling for various context chunk types
        if isinstance(context_chunks, list) and len(context_chunks) > 0:
            if isinstance(context_chunks[0], tuple):
                docs = [c[0] for c in context_chunks]
                contexts = [doc.page_content for doc in docs]
                scores = [c[1] for c in context_chunks]
            elif hasattr(context_chunks[0], "page_content"):
                docs = context_chunks
                contexts = [doc.page_content for doc in docs]
                scores = None
            elif isinstance(context_chunks[0], str):
                docs = context_chunks
                contexts = context_chunks
                scores = None
            else:
                docs = context_chunks
                contexts = [str(c) for c in context_chunks]
                scores = None
        else:
            docs = context_chunks
            contexts = [str(context_chunks)]
            scores = None

        # Extract and display source info
        sources = []
        if isinstance(docs, list):
            for doc in docs:
                metadata = getattr(doc, "metadata", {})
                source_info = f"{metadata.get('filename', 'unknown')} (page {metadata.get('page', '?')})"
                sources.append(source_info)
        else:
            metadata = getattr(docs, "metadata", {})
            source_info = f"{metadata.get('filename', 'unknown')} (page {metadata.get('page', '?')})"
            sources.append(source_info)

        print("\nSources of retrieved chunks:")
        for s in sources:
            print(f" - {s}")

        context_text = "\n".join(contexts)
        modified_query = self.construct_query(query, context_text)
        response = self.generate_response(modified_query, print_model=True)
        self.conversation_history.append({"question": query, "answer": response})
        print(f"Q: {query}\nA: {response}\n")

        avg_score = None
        k = None
        if scores:
            avg_score = sum(scores) / len(scores) if scores else 0
            k = len(scores)
            print(f"\nAverage Vector relevance scores:  {avg_score:.4f} (k={k})\n")
        cosine_sim = self.evaluate_metrics(query, response, contexts)

        if return_score:
            return response, avg_score, k, cosine_sim, sources
        return response


    def _call_model(self, state: MessagesState):
        user_query = state["messages"][-1].content
        context = self.retrieve_context(user_query)
        modified_query = self.construct_query(user_query, context)
        response = self.generate_response(modified_query)
        return {"messages": [HumanMessage(content=response)]}

    def reset_memory(self):
        self.thread_id = uuid.uuid4()
        self.config = {"configurable": {"thread_id": self.thread_id}}
        self.conversation_history.clear()
        print("Memory reset.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Chatbot with Configurable Settings")
    parser.add_argument('-c', '--config', default='ask_config_openrouter.json', help='Path to the configuration file.')
    args = parser.parse_args()
    chatbot = ChatBot(config_path=args.config)
    while True:
        query = input("Enter your question (type 'exit' to quit, 'reset' to reset memory): ")
        if query.lower() == 'exit':
            print("Exiting...")
            break
        elif query.lower() == 'reset':
            chatbot.reset_memory()
        else:
            chatbot.ask(query)
