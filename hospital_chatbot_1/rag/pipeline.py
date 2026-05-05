"""
rag/pipeline.py - RAG (Retrieval-Augmented Generation) Pipeline
===============================================================
This is the brain of the chatbot. The flow is:

  User Question
       ↓
  Embed the question using Google Embedding API
       ↓
  Search FAISS vector store for top-K relevant chunks
       ↓
  Build a prompt: [System Prompt] + [Retrieved Context] + [Chat History] + [Question]
       ↓
  Send to Google Gemini for a grounded answer
       ↓
  Return answer + source references
"""

import os
from typing import Optional
from rag.embedder import Embedder
from rag.vector_store import VectorStore
from rag.gemini_client import GeminiClient
from rag.history import ConversationHistory


class RAGPipeline:
    def __init__(self):
        self.embedder = Embedder()
        self.vector_store = VectorStore()
        self.llm = GeminiClient()
        self.history = ConversationHistory()
        self._ready = False

        # Build the vector store from hospital knowledge base
        self._initialize_knowledge_base()

    def _initialize_knowledge_base(self):
        """Load hospital documents and build FAISS index."""
        from data.hospital_data import get_all_documents
        documents = get_all_documents()
        print(f"📚 Indexing {len(documents)} knowledge chunks...")
        self.vector_store.build_index(documents, self.embedder)
        self._ready = True
        print("✅ Knowledge base indexed!")

    def query(self, user_question: str, session_id: str = "default") -> dict:
        """
        Full RAG pipeline:
        1. Embed question
        2. Retrieve relevant chunks
        3. Build augmented prompt
        4. Generate answer with Gemini
        5. Store in history
        """
        # Step 1: Retrieve relevant context chunks
        relevant_chunks = self.vector_store.search(
            query=user_question,
            embedder=self.embedder,
            top_k=4
        )

        # Step 2: Build context string from retrieved chunks
        context_text = "\n\n".join([
            f"[Source: {chunk['source']}]\n{chunk['text']}"
            for chunk in relevant_chunks
        ])

        # Step 3: Get recent conversation history
        chat_history = self.history.get_history(session_id, last_n=6)

        # Step 4: Build the full prompt
        prompt = self._build_prompt(user_question, context_text, chat_history)

        # Step 5: Generate answer via Gemini
        answer = self.llm.generate(prompt)

        # Step 6: Store this turn in history
        self.history.add(session_id, user_question, answer)

        # Step 7: Collect source names
        sources = list(set([chunk["source"] for chunk in relevant_chunks]))

        return {
            "answer": answer,
            "sources": sources,
            "confidence": "high" if relevant_chunks else "low"
        }

    def _build_prompt(self, question: str, context: str, history: list) -> str:
        """
        Assemble the final prompt sent to Gemini.
        Structure:
          - System instructions (who the bot is, how to behave)
          - Retrieved hospital knowledge context
          - Conversation history (last N turns)
          - Current user question
        """
        system_prompt = """You are a helpful and friendly medical assistant chatbot for City General Hospital.
You help patients and visitors with:
- Hospital timings and department schedules
- Doctor fees and consultation charges
- Appointment booking information
- Disease/diagnosis information and general medical guidance
- Emergency contact numbers
- Pharmacy and lab services

IMPORTANT RULES:
1. Only answer based on the provided hospital context below.
2. If information is not in the context, say: "I don't have that specific information. Please call us at +1-800-HOSPITAL or visit the reception."
3. Never diagnose or prescribe medication. Always recommend seeing a doctor for medical decisions.
4. Be empathetic and professional.
5. Keep answers concise but complete.
6. Format lists clearly with bullet points when appropriate.
"""

        history_text = ""
        if history:
            history_text = "\n--- Conversation History ---\n"
            for turn in history:
                history_text += f"Patient: {turn['user']}\nAssistant: {turn['bot']}\n"

        prompt = f"""{system_prompt}

--- Hospital Knowledge Base ---
{context}

{history_text}
--- Current Question ---
Patient: {question}
Assistant:"""

        return prompt

    def is_ready(self) -> bool:
        return self._ready

    def clear_history(self, session_id: str):
        self.history.clear(session_id)