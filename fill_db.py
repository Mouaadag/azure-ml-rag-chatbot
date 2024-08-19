import os
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
from tqdm import tqdm

DATA_PATH = r"data"
CHROMA_PATH = r"chroma_db"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
BATCH_SIZE = 64  # Adjust based on your GPU memory


def main():
    # Check for GPU
    print("Num GPUs Available: ", len(tf.config.list_physical_devices("GPU")))

    # Load the model
    embed = hub.load(
        "https://tfhub.dev/google/universal-sentence-encoder-multilingual/3"
    )

    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = chroma_client.get_or_create_collection(name="azure_ml_docs")

    print("Loading documents...")
    loader = PyPDFDirectoryLoader(DATA_PATH)
    raw_documents = loader.load()

    print("Splitting documents...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        is_separator_regex=False,
    )

    chunks = text_splitter.split_documents(raw_documents)
    total_chunks = len(chunks)
    print(f"Total chunks: {total_chunks}")

    for i in tqdm(range(0, total_chunks, BATCH_SIZE), desc="Processing batches"):
        batch = chunks[i : i + BATCH_SIZE]
        texts = [chunk.page_content for chunk in batch]

        # Generate embeddings
        embeddings = embed(texts).numpy().tolist()

        # Prepare data for upsert
        ids = [f"ID{j}" for j in range(i, min(i + BATCH_SIZE, total_chunks))]
        metadatas = [chunk.metadata for chunk in batch]

        # Upsert to Chroma
        collection.upsert(
            ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas
        )

    print(f"Added {total_chunks} chunks to the database.")


if __name__ == "__main__":
    main()
