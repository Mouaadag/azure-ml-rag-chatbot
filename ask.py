import chromadb
from anthropic import Anthropic
from dotenv import load_dotenv
import os
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text

load_dotenv()

CHROMA_PATH = r"chroma_db"

# Check for GPU
print("Num GPUs Available: ", len(tf.config.list_physical_devices("GPU")))

# Load the model
embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-multilingual/3")

chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = chroma_client.get_collection(name="azure_ml_docs")

anthropic = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))


def get_relevant_context(query):
    query_embedding = embed([query]).numpy().tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=3)
    return "\n".join(results["documents"][0])


system_prompt = """
You are an AI assistant specializing in Azure Machine Learning. Your responses should be based solely on the 
information provided to you about Azure ML. If you don't have enough information to answer a question, 
say so and ask for clarification. Maintain a friendly and helpful tone throughout the conversation.
"""

conversation_history = []

print(
    "Azure ML Chatbot: Hello! I'm here to help you with Azure Machine Learning. What would you like to know?"
)

while True:
    user_input = input("You: ")

    if user_input.lower() in ["exit", "quit", "bye"]:
        print(
            "Azure ML Chatbot: Goodbye! If you have more questions about Azure ML in the future, feel free to ask."
        )
        break

    relevant_context = get_relevant_context(user_input)

    conversation_history.append({"role": "user", "content": user_input})

    response = anthropic.messages.create(
        model="claude-3-sonnet-20240229",
        max_tokens=300,
        system=system_prompt,
        messages=[
            {"role": "system", "content": f"Relevant information: {relevant_context}"},
            *conversation_history,
        ],
    )

    assistant_response = response.content[0].text
    print("Azure ML Chatbot:", assistant_response)

    conversation_history.append({"role": "assistant", "content": assistant_response})

    # Limit conversation history to last 10 messages to manage token usage
    if len(conversation_history) > 10:
        conversation_history = conversation_history[-10:]

print("Thank you for using the Azure ML Chatbot!")
