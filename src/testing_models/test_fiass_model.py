import pickle
import faiss
from sentence_transformers import SentenceTransformer

INDEX_PATH = "models/faiss/faq_index.faiss"
METADATA_PATH = "models/faiss/faq_metadata.pkl"

def load_faiss():
    index = faiss.read_index(INDEX_PATH)
    with open(METADATA_PATH, "rb") as f:
        answers = pickle.load(f)
    return index, answers

def search_questions(queries, k=3):
    index, answers = load_faiss()
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    embeddings = model.encode(queries, show_progress_bar=True)
    distances, indices = index.search(embeddings, k)

    for query_idx, query in enumerate(queries):
        print(f"\nQuery: {query}")
        for rank in range(k):
            idx = int(indices[query_idx][rank])
            dist = float(distances[query_idx][rank])
            print(f"  Match {rank + 1}: idx={idx}, dist={dist:.4f}")
            print(f"    Answer: {answers[idx]}")

if __name__ == "__main__":
    queries = [
        "How can I track my order?",
        "What payment methods do you accept?",
        "Can I return an item after 30 days?",
        "How long does shipping take?",
        "Do you ship internationally?"
    ]
    search_questions(queries, k=3)