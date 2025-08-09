
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import argparse

def search(query, model_name='all-MiniLM-L6-v2', index_path='faiss_index.bin', data_path='documents.npy', k=3):
    """Searches the FAISS index for the most similar documents to a query."""
    model = SentenceTransformer(model_name)
    index = faiss.read_index(index_path)
    documents = np.load(data_path, allow_pickle=True)
    
    query_embedding = model.encode([query], convert_to_tensor=False)
    distances, indices = index.search(np.array(query_embedding, dtype=np.float32), k)
    
    results = []
    for i, idx in enumerate(indices[0]):
        if idx != -1:
            results.append({
                'distance': distances[0][i],
                'document': documents[idx]
            })
    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Search for documents similar to a query.')
    parser.add_argument('query', type=str, help='The search query.')
    parser.add_argument('--k', type=int, default=3, help='The number of results to return.')
    args = parser.parse_args()
    
    results = search(args.query, k=args.k)
    
    print(f"Top {len(results)} results for query: '{args.query}'")
    for result in results:
        print(f"- Distance: {result['distance']:.4f}")
        print(f"  Document: {result['document']}")
        print("-" * 20)
