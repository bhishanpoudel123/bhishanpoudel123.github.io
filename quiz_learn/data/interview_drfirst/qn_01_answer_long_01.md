
**Q: One of the things you did to improve the latency was cache the frequently asked questions. Could you explain how you did this, given there might be different variations?**

Yes — latency reduction was critical for ensuring a smooth user experience with the AI chatbot I developed at Cencora. Since users often asked semantically similar questions (e.g., “What are this quarter's KPIs?” vs. “Can I see the Q2 business performance?”), I implemented a **semantic caching mechanism** to accelerate response times:

### 🧠 Semantic Caching with Embedding-Based Matching

1. **Vector Embedding of Questions**
   I used OpenAI's embedding model (or `sentence-transformers` for local inference) to convert both incoming user queries and stored FAQs into dense vector representations.

2. **Similarity Matching**
   On each query, I compared the incoming vector with existing cached vectors using **cosine similarity**. If a match exceeded a defined similarity threshold (e.g., 0.92), I served the **cached response** instantly without re-querying the LLM or backend systems.

3. **Approximate Nearest Neighbor (ANN) Search**
   To scale this efficiently, especially with hundreds of FAQs, I leveraged `FAISS` (Facebook AI Similarity Search) for high-speed similarity searches within vector space.

4. **Dynamic Cache Invalidation**
   Since business data evolves, I tagged cache entries with TTLs (time-to-live) or attached metadata (e.g., timestamp, data version) to invalidate stale answers and ensure real-time relevance.

5. **Cold Start Optimization**
   For newly asked questions that didn’t have a match, the chatbot would fetch fresh data, generate a response using the LLM, and **store the new Q\&A pair** in the cache for future reuse.

6. **Hybrid Strategy with RAG**
   In some cases, I combined this caching approach with Retrieval-Augmented Generation (RAG) using **LlamaIndex** — allowing the system to first check cache, then fallback to a RAG pipeline if needed.

# FAISS embedding Code

```python
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Load an embedding model (e.g., sentence-transformers)
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Sample cached questions and their responses
cached_questions = [
    "What are this quarter's KPIs?",
    "Show me the latest business performance.",
    "Can I see the Q2 financial report?",
    "Give me details about revenue growth."
]

cached_answers = [
    "The KPIs for this quarter are revenue growth, customer retention, and operating margin.",
    "Here is the latest business performance report.",
    "The Q2 financial report shows a 10% increase in revenue and stable profit margins.",
    "Revenue growth is projected at 8% for this quarter."
]

# Convert questions into embeddings
question_embeddings = embedding_model.encode(cached_questions)
question_embeddings = np.array(question_embeddings).astype('float32')

# Initialize FAISS index for fast similarity search
dimension = question_embeddings.shape[1]  # Embedding dimension
index = faiss.IndexFlatL2(dimension)
index.add(question_embeddings)

def get_cached_answer(user_query, threshold=0.92):
    """Checks the cache for a semantically similar question and returns the cached answer."""
    query_embedding = embedding_model.encode([user_query]).astype('float32')

    # Search for nearest neighbors
    _, indices = index.search(query_embedding, k=1)  # Find top-1 match
    matched_index = indices[0][0]

    # Compute cosine similarity
    query_vector = query_embedding[0]
    matched_vector = question_embeddings[matched_index]
    similarity = np.dot(query_vector, matched_vector) / (np.linalg.norm(query_vector) * np.linalg.norm(matched_vector))

    if similarity >= threshold:
        return cached_answers[matched_index]  # Return cached answer

    return None  # No match found

# Example user input
user_input = "How did the business perform in Q2?"
cached_response = get_cached_answer(user_input)

if cached_response:
    print(f"💾 Cached Answer: {cached_response}")
else:
    print("🤖 No match found. Querying AI chatbot...")
    # Call your AI chatbot function here and store the new answer in the cache

```
