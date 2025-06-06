# Recommender System for GoodBooks-10k Dataset

## Project Overview

This project implements various recommender system techniques on the GoodBooks-10k dataset, which contains book ratings from users. The goal is to explore different recommendation approaches and compare their effectiveness in suggesting relevant books to users.

## Dataset Description

The GoodBooks-10k dataset consists of:
- **Books data**: Contains metadata about 10,000 books including:
  - `book_id`: Unique identifier
  - `title`: Book title
  - `authors`: Author names
  - `original_publication_year`: Year of publication
  - `average_rating`: Average rating from all users
  - And other metadata like ISBN, language, etc.

- **Ratings data**: Contains user-book interactions:
  - `user_id`: Unique user identifier
  - `book_id`: Book identifier
  - `rating`: Rating score (1-5)

For this project, we're working with a subset (1,000 books and 5,000 ratings) to make computation more manageable.

## Implemented Recommendation Approaches

### 1. Collaborative Filtering

#### a. Item-Based (Cosine Similarity)
- **Approach**: Finds similar books based on user rating patterns
- **Implementation**:
  - Creates user-item matrix
  - Computes cosine similarity between items
  - Recommends items similar to those the user has liked
- **Pros**:
  - Simple to implement
  - Works well when item features are hard to define
  - Can capture subtle relationships between items
- **Cons**:
  - Cold start problem for new items
  - Sparsity can be an issue with limited user-item interactions
  - Doesn't incorporate item metadata

#### b. User-Based (PyTorch Neural Network)
- **Approach**: Learns user and book embeddings to predict ratings
- **Implementation**:
  - Uses PyTorch to create embedding layers for users and books
  - Trains a neural network to predict ratings
  - Recommends books with highest predicted ratings
- **Pros**:
  - Can capture complex patterns in user preferences
  - Embeddings can learn latent features
  - Handles large datasets efficiently
- **Cons**:
  - Requires more computational resources
  - Needs sufficient training data
  - Harder to interpret than simpler methods

### 2. Model-Based Approaches

#### a. SVD (SciPy)
- **Approach**: Matrix factorization using Singular Value Decomposition
- **Implementation**:
  - Creates normalized user-item matrix
  - Applies SVD to decompose into user and item factors
  - Reconstructs matrix to predict missing ratings
- **Pros**:
  - Handles sparsity better than memory-based methods
  - Captures latent factors in the data
  - Efficient for medium-sized datasets
- **Cons**:
  - Cold start problem
  - Hard to incorporate additional features
  - Computationally intensive for very large matrices

#### b. SVD (Surprise Library)
- **Approach**: Optimized SVD implementation from Surprise library
- **Implementation**:
  - Uses built-in Dataset and SVD classes
  - Includes hyperparameter tuning capabilities
  - Provides evaluation metrics
- **Pros**:
  - Easy to use API
  - Built-in cross-validation
  - Optimized implementation
- **Cons**:
  - Less flexible than custom implementations
  - Still suffers from standard SVD limitations

### 3. Knowledge-Based Recommender
- **Approach**: Uses explicit rules based on book metadata
- **Implementation**:
  - Extracts user preferences (favorite authors, publication years)
  - Filters books matching these criteria
  - Ranks by popularity/rating
- **Pros**:
  - No cold start problem for new users
  - Transparent and explainable
  - Can incorporate domain knowledge
- **Cons**:
  - Requires manual rule creation
  - Doesn't learn from user behavior
  - Limited personalization

### 4. Content-Based Filtering (TF-IDF)
- **Approach**: Recommends similar books based on content features
- **Implementation**:
  - Creates TF-IDF vectors from book titles and authors
  - Computes cosine similarity between books
  - Recommends books similar to those the user liked
- **Pros**:
  - Works without user rating data
  - No cold start for new items
  - Explainable recommendations
- **Cons**:
  - Limited to observable features
  - Doesn't capture user behavior patterns
  - Quality depends on feature engineering

## Comparative Analysis

| Method               | Personalization | Cold Start Handling | Explainability | Scalability |
|----------------------|-----------------|---------------------|----------------|-------------|
| Item-Based CF        | High            | Poor (items)        | Medium         | Medium      |
| User-Based NN        | Very High       | Poor (both)         | Low            | High        |
| SVD                  | High            | Poor (both)         | Medium         | Medium      |
| Knowledge-Based      | Low             | Excellent           | High           | High        |
| Content-Based        | Medium          | Good (users)        | High           | High        |

## Potential Improvements

1. **Hybrid Approaches**:
   - Combine collaborative and content-based filtering
   - Use knowledge-based rules to handle cold start
   - Ensemble methods to leverage strengths of different approaches

2. **Advanced Techniques**:
   - Deep learning models (Neural Collaborative Filtering)
   - Graph-based recommendations
   - Context-aware recommendations (time, location)

3. **Feature Engineering**:
   - Incorporate more book metadata (genres, descriptions)
   - Use NLP techniques on book descriptions
   - Add temporal features for user preferences

4. **Evaluation Framework**:
   - Implement proper train-test splits
   - Add evaluation metrics (precision, recall, NDCG)
   - User studies for qualitative assessment

5. **Scalability Improvements**:
   - Approximate nearest neighbors for similarity
   - Distributed computing for large datasets
   - Incremental learning for new data

## How to Choose an Approach

1. **For new systems with little data**:
   - Start with content-based or knowledge-based
   - Gradually incorporate collaborative filtering as data accumulates

2. **For mature systems with abundant data**:
   - Use collaborative filtering or matrix factorization
   - Consider deep learning approaches for maximum personalization

3. **When explainability is important**:
   - Prefer content-based or knowledge-based
   - Use hybrid approaches that can provide explanations

4. **For cold start problems**:
   - Implement robust content-based fallbacks
   - Use demographic or contextual information

## Conclusion

This project demonstrates a comprehensive exploration of recommender system techniques on book rating data. Each approach has its strengths and weaknesses, and the best solution often depends on the specific requirements of the application, the available data, and the stage of the product lifecycle. Future work could focus on building hybrid systems that combine the strengths of these different approaches while mitigating their individual weaknesses.