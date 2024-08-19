import streamlit as st
import chromadb
from anthropic import Anthropic
from dotenv import load_dotenv
import os
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

CHROMA_PATH = r"chroma_db"


# Load models and initialize clients
@st.cache_resource
def load_models_and_clients():
    embed = hub.load(
        "https://tfhub.dev/google/universal-sentence-encoder-multilingual/3"
    )
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = chroma_client.get_collection(name="azure_ml_docs")
    anthropic = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    return embed, collection, anthropic


embed, collection, anthropic = load_models_and_clients()


def get_relevant_context(query, n_results=10):
    query_embedding = embed([query]).numpy().tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=n_results)
    ranked_results = rank_results(query, results["documents"][0])
    return "\n\n".join(ranked_results)  # Return all ranked results


def rank_results(query, documents):
    # Use TF-IDF and cosine similarity for more robust ranking
    vectorizer = TfidfVectorizer().fit([query] + documents)
    query_vec = vectorizer.transform([query])
    doc_vecs = vectorizer.transform(documents)
    similarities = cosine_similarity(query_vec, doc_vecs)[0]

    # Sort documents by similarity score
    sorted_docs = [doc for _, doc in sorted(zip(similarities, documents), reverse=True)]
    return sorted_docs


system_prompt = """
You are an AI assistant specializing in Azure Machine Learning. Your responses must be based EXCLUSIVELY on the 
information provided to you in the 'Relevant information' section. Do not use any external knowledge or information 
that is not explicitly given.

Before answering, carefully review ALL provided information to ensure a comprehensive and consistent response.

If the provided information is not sufficient to answer a question:
1. Clearly state that you don't have enough information to provide a complete answer.
2. Explain what specific information is missing or unclear.
3. Ask for clarification or additional details if necessary.

Do not make assumptions or use any knowledge beyond what is explicitly provided. If you're unsure, it's better 
to admit lack of information than to guess or use external knowledge.

Maintain a friendly and helpful tone throughout the conversation, but prioritize accuracy based solely on the 
given information over completeness of the answer.
"""

st.title("Azure ML Chatbot")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("What would you like to know about Azure ML?"):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    relevant_context = get_relevant_context(prompt)

    with st.chat_message("assistant"):
        try:
            response = anthropic.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=1000,
                system=f"{system_prompt}\n\nRelevant information: {relevant_context}",
                messages=[*st.session_state.messages],
            )
            full_response = response.content[0].text

            # Display the full response
            st.markdown(full_response)
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            full_response = (
                "I'm sorry, I encountered an error while processing your request."
            )
            st.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})

    # Limit conversation history to last 10 messages
    if len(st.session_state.messages) > 10:
        st.session_state.messages = st.session_state.messages[-10:]

st.sidebar.title("About")
st.sidebar.info(
    "This is an AI-powered chatbot specializing in Azure Machine Learning. It uses RAG (Retrieval-Augmented Generation) to provide accurate and up-to-date information."
)
