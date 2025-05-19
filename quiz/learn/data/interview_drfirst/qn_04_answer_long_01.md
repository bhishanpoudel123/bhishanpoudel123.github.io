

### ✅ **1. Personalized Drug Recommendation System** *(Healthcare, Cencora project – hypothetical extension)*

**Goal:** Suggest next-best drugs or therapies based on patient profiles, prescriptions, and historical effectiveness.

**Approach:**

* Used **collaborative filtering** on anonymized prescription logs to find patterns in what drugs are commonly prescribed together.
* Combined this with **content-based filtering** that leveraged structured features: patient age, diagnosis, previous meds, treatment success, etc.
* Final system used **hybrid modeling**:

  * Matrix factorization (e.g., SVD++) for behavior-based inference.
  * Clinical features passed through gradient-boosted trees for personalization.
* Deployed with real-time scoring using **FastAPI**, allowing healthcare providers to see drug suggestions during case reviews.

**Tools:** Python, scikit-learn, LightGBM, FastAPI, Azure, Pandas



### ✅ **2. NLP-Driven Chatter Response Recommendation** *(AmerisourceBergen chatbot project)*

**Goal:** Recommend appropriate chatbot responses based on the text of incoming patient service queries.

**Approach:**

* Vectorized patient messages using **BERT embeddings**.
* Used **K-Nearest Neighbors** (KNN) in vector space to find top similar past messages and surface the most frequent successful responses.
* Logged interactions and fine-tuned a GPT-based classifier to auto-generate suggestions for edge cases.

**Outcome:** This reduced the number of fallback LLM calls and improved latency, while maintaining response relevance.

**Tools:** BERT, scikit-learn, FAISS, OpenAI, FastAPI, LangChain



### ✅ **3. Movie Recommendation (Hackathon Project)**

**Goal:** Build a real-time movie recommendation engine based on user clicks and reviews.

**Approach:**

* Created user and item embeddings using **Word2Vec-style collaborative embeddings**.
* Trained an **implicit feedback model** using Alternating Least Squares (ALS).
* Used **streaming behavior (watch history, likes)** to continuously update preferences.

**Demo:** Built a Streamlit app with top-N recommendations and cold-start fallback to trending content.

**Tools:** LightFM, Implicit, Streamlit, Redis for caching



### ✅ **4. Product Recommender in E-commerce Dataset (Portfolio Project)**

**Goal:** Recommend related products on a product page (e.g., “Customers Also Bought”).

**Approach:**

* Item-to-item similarity based on:

  * **Co-purchase graphs** (Neo4j graph traversal)
  * **Description embeddings** (TF-IDF + cosine similarity)
* Built a hybrid system combining item metadata and purchase history.

**Tools:** scikit-learn, TF-IDF, Neo4j, Flask



### Summary of Techniques I've Used:

| Technique                  | Use Case                                 |
| -- | - |
| Content-based filtering    | Profile → item matching                  |
| Collaborative filtering    | User × item matrix inference             |
| Matrix factorization       | SVD, ALS for latent embeddings           |
| NLP embeddings (BERT)      | Semantic similarity in text-based RecSys |
| Graph-based recommendation | Neo4j or co-occurrence modeling          |
| Hybrid systems             | Combined strength of multiple approaches |


