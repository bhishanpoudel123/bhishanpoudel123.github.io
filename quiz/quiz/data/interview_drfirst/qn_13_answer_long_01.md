# Q: TF-IDF

**Interviewer**: You also mentioned other models like TF-IDF. What are these models and how do they compare to BERT?

**Me**: Yes, I've worked with several text representation models before the deep learning era and alongside more advanced approaches. Let me explain these:

**TF-IDF (Term Frequency-Inverse Document Frequency)**:
- A statistical measure that evaluates word importance in a document relative to a corpus
- **Term Frequency**: Counts how frequently a word appears in a document
- **Inverse Document Frequency**: Downweights words that appear in many documents
- Advantages: Computationally efficient, interpretable, works well for search and document classification
- Limitations: Ignores word order and semantics, no contextual understanding

**Bag of Words**:
- Represents text as an unordered set of words with their frequency counts
- Creates sparse vector representations where each dimension corresponds to a word in the vocabulary
- Simple but loses all grammatical and word order information
- Often serves as a baseline for text classification tasks

**Word2Vec**:
- Neural network-based embedding technique that maps words to dense vector spaces
- Words with similar meanings cluster together in the vector space
- Captures some semantic relationships (e.g., "king" - "man" + "woman" ≈ "queen")
- Limited by single vector per word regardless of context (no polysemy handling)

**GloVe (Global Vectors)**:
- Combines global matrix factorization with local context window methods
- Creates word embeddings that capture linear substructures of word relationships
- Trained on aggregate global word-word co-occurrence statistics

**Comparison with BERT**:
- Earlier models (TF-IDF, BoW) treat words as discrete symbols without inherent relationships
- Word2Vec and GloVe capture static word semantics but miss contextual variations
- BERT generates dynamic contextualized embeddings where the same word has different representations based on surrounding context
- BERT incorporates bidirectional context and can handle polysemy, idioms, and complex language structures

In my work at Cencora for clinical text analytics, I evaluated all these approaches. While TF-IDF and simpler embeddings worked for basic classification tasks, BERT significantly outperformed them for understanding nuanced patient communications that required contextual comprehension.
                                                                        **Interviewer**: Could you also provide the code for implementing TF-IDF in Python?

**Me**: Certainly. Here's how I would implement TF-IDF in Python using scikit-learn, which is the approach I've used in my NLP projects:

```python
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Sample data - in a real project, this would be loaded from a file
documents = [
    "Patient reports severe headache and nausea",
    "Customer complaint about drug delivery delay",
    "Patient experiencing side effects from medication",
    "Question about insurance coverage for prescription",
    "Request for refill of existing prescription"
]

# Labels for classification
labels = [0, 1, 0, 1, 1]  # Binary labels for demonstration

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    documents, labels, test_size=0.2, random_state=42
)

# Initialize and fit the TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer(
    min_df=2,           # Minimum document frequency
    max_df=0.8,         # Maximum document frequency
    ngram_range=(1, 2), # Consider both unigrams and bigrams
    stop_words='english'# Remove English stop words
)

# Transform the documents to TF-IDF features
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Train a classifier using the TF-IDF features
classifier = LogisticRegression(random_state=42)
classifier.fit(X_train_tfidf, y_train)

# Make predictions
y_pred = classifier.predict(X_test_tfidf)

# Evaluate the model
print(classification_report(y_test, y_pred))

# For new documents
new_text = ["Patient requesting information about drug interactions"]
new_text_tfidf = tfidf_vectorizer.transform(new_text)
prediction = classifier.predict(new_text_tfidf)
print(f"Prediction for new text: {prediction}")

# To examine the TF-IDF values
feature_names = tfidf_vectorizer.get_feature_names_out()
# Get TF-IDF values for the first document
first_document_vector = X_train_tfidf[0]
# Create a dictionary of terms and their TF-IDF values
tfidf_results = {feature_names[i]: first_document_vector[0, i] 
                for i in first_document_vector.indices}
# Sort by TF-IDF score in descending order
sorted_tfidf = {k: v for k, v in sorted(tfidf_results.items(), key=lambda item: item[1], reverse=True)}
print("Top TF-IDF terms in first document:", list(sorted_tfidf.items())[:5])
```

In my clinical text analytics project at Cencora, I used a similar approach but with more sophisticated preprocessing:

1. Applied medical-specific text cleaning (handling abbreviations, medical terms)
2. Adjusted the vectorizer parameters based on our healthcare vocabulary
3. Combined TF-IDF features with domain-specific features 
4. Conducted thorough hyperparameter tuning

While this approach worked well for initial classification tasks, we later transitioned to BERT-based models for more nuanced understanding of patient communications, as they better captured context and medical terminology relationships.                                                                                                                          

