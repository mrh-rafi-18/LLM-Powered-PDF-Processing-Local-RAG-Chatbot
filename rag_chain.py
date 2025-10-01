# =========================
# rag_chain.py 
# =========================
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from collections import deque
from typing import List
from langchain.schema import Document

load_dotenv()


class RAGChain:
    def __init__(self, retriever, max_turns: int = 4):
        """
        Chat-style RAG system with multi-turn memory.
        Only last 'max_turns' of human/AI messages are sent to the LLM.
        """
        self.model = ChatOllama(
                model="llama3.2",
                temperature=0.3,
                max_tokens=200
            )

        

        self.retriever = retriever
        self.max_turns = max_turns  # number of past messages to keep for each user/AI
        self.history = deque(maxlen=2*max_turns)  # store alternating Human/AI messages

        self.system="You are a helpful assistant for  PDF documents Q/A from partial document contexts."
        # Prompt template
        self.prompt_template = PromptTemplate(
            template="""
            Use the following context to answer the question concisely and accurately.
            
            Context: {context}
            

            User Question: {question}
            """,
            
            input_variables=["context", "question"]
        )

    def query(self, user_query: str) -> str:
        # Retrieve relevant documents
        retrieved_docs = self.retriever.retrieve(user_query, k=5)
        retrieved_context = "\n\n".join([doc.page_content for doc in retrieved_docs])

        # Build final text prompt from template
        prompt= self.prompt_template.invoke({
            "context":retrieved_context,
            "question":user_query
         }).to_string()


        # Add to history
        self.history.append(prompt)

        # Build message list: system + last N turns
        messages_to_send = [self.system] + list(self.history)

        # Generate response
        response = self.model.invoke(messages_to_send)

        # Wrap LLM response and add to history
        ai_msg = "Ai response: "+ response.content
        self.history.append(ai_msg)

        return response.content  
