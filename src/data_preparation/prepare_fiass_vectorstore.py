import os
import json
import pickle
import pandas as pd
import faiss
from datasets import load_dataset
from sentence_transformers import SentenceTransformer


def build_knowledge_base():
    # 1. Setup directories
    os.makedirs("models/faiss", exist_ok=True)
    
    # 2. Load your FAQ dataset
    # This code loads the dataset and converts it into a list of dictionaries:
    # faq_data = [{"question": ..., "answer": ...}, ...]
    raw_dataset = load_dataset("MakTek/Customer_support_faqs_dataset")
    train_dataset = raw_dataset["train"] if "train" in raw_dataset else raw_dataset

    faq_data = [
        {"question": question, "answer": answer}
        for question, answer in zip(train_dataset["question"], train_dataset["answer"])
    ]

    print(f"Loaded {len(faq_data)} FAQ entries.")
    print("Sample faq_data:", faq_data[:3])

    # Extract questions and answers into clean lists
    questions = [item["question"] for item in faq_data]
    answers = [item["answer"] for item in faq_data]
    
    # 3. Load the embedding model
    # sentence-transformers/all-MiniLM-L6-v2 is ultra-lightweight (22M parameters) 
    # and runs instantly on standard CPUs while keeping high text semantic understanding.
    print("📥 Loading sentence embedding model...")
    embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    
    # # 4. Generate Vector Embeddings
    print("⚡ Generating vector embeddings for FAQ questions...")
    question_embeddings = embedding_model.encode(questions, show_progress_bar=True)
    
    # # Get the embedding vector dimension size (MiniLM maps text to a 384-dimensional space)
    embedding_dimension = question_embeddings.shape[1]
    
    # 5. Create and Populate the FAISS Index
    # IndexFlatL2 measures similarities using Euclidean Distance (L2 distance)
    print(f"🏗️ Building FAISS Index (Dimension size: {embedding_dimension})...")
    index = faiss.IndexFlatL2(embedding_dimension)
    index.add(question_embeddings) # Add the vectors into the index database
    
    # 6. Save the Index and Text Metadata to Disk
    # FAISS only stores vectors, not text. We must save the answers list separately 
    # to map the numerical search result back to the corresponding answer string.
    index_path = "models/faiss/faq_index.faiss"
    metadata_path = "models/faiss/faq_metadata.pkl"
    
    faiss.write_index(index, index_path)
    with open(metadata_path, "wb") as f:
        pickle.dump(answers, f)
        
    print("\n🎉 FAISS Retrieval system constructed successfully!")
    print(f"💾 Saved Vector Index ➡️ {index_path}")
    print(f"💾 Saved Text Answers Metadata ➡️ {metadata_path}")

    # ==========================================
    # VERIFICATION TESTING (Sanity Check)
    # ==========================================
    print("\n🔍 Running verification search testing...")
    test_query = "Can I pay using PayPal?"
    
    # Step A: Embed user query
    query_vector = embedding_model.encode([test_query])
    
    # Step B: Query the vector index (k=1 means return the single closest matching FAQ match)
    distances, indices = index.search(query_vector, k=1)
    
    # Step C: Retrieve the corresponding answer text string
    matched_index = indices[0][0]
    retrieved_answer = answers[matched_index]
    
    print(f"User Query: '{test_query}'")
    print(f"Retrieved Match Answer: '{retrieved_answer}' (Distance Score: {distances[0][0]:.4f})")


if __name__ == "__main__":
    build_knowledge_base()