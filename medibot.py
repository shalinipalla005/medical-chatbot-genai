import os
import streamlit as st

from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv, find_dotenv

# Load API keys
load_dotenv(find_dotenv())
# st.write("Loaded key:", os.getenv("GROQ_API_KEY"))

DB_FAISS_PATH = "vectorstore/db_faiss"

# Cache vectorstore for performance
@st.cache_resource
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db


def set_custom_prompt():
    """Enhanced prompt with structured output formatting."""
    custom_prompt_template = """
    You are a professional and concise medical assistant chatbot.
    Use ONLY the context provided below to answer the user's question clearly and factually.
    If you don't know the answer, say so politely.

    Format your response in **markdown** with:
    - A short **summary** of the answer (2–3 lines max)
    - Then a **detailed explanation** using bullet points or numbered lists where appropriate
    - Avoid repetition and small talk
    - End with: "_Source: Internal medical knowledge base_"

    ---
    Context:
    {context}

    Question:
    {question}

    ---
    Answer:
    """
    return PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])


def main():
    st.title("Medical Assistant Chatbot")

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])

    prompt = st.chat_input("Type your medical question here...")

    if prompt:
        st.chat_message('user').markdown(prompt)
        st.session_state.messages.append({'role': 'user', 'content': prompt})

        try:
            vectorstore = get_vectorstore()
            if vectorstore is None:
                st.error("Failed to load the vector store")
                return

            qa_chain = RetrievalQA.from_chain_type(
                llm=ChatGroq(
                    model_name="meta-llama/llama-4-maverick-17b-128e-instruct",
                    temperature=0.2,
                    groq_api_key=os.environ["GROQ_API_KEY"],
                ),
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
                return_source_documents=True,
                chain_type_kwargs={'prompt': set_custom_prompt()}
            )

            response = qa_chain.invoke({'query': prompt})

            result = response["result"]
            source_docs = response["source_documents"]

            # Format the output neatly
            formatted_sources = "\n".join(
                [f"- {doc.metadata.get('source', 'Unknown source')}" for doc in source_docs]
            )
            result_to_show = f"{result}\n\n**Referenced Documents:**\n{formatted_sources}"

            st.chat_message('assistant').markdown(result_to_show)
            st.session_state.messages.append({'role': 'assistant', 'content': result_to_show})

        except Exception as e:
            st.error(f"⚠️ Error: {str(e)}")


if __name__ == "__main__":
    main()
