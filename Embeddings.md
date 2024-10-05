# Embedding

- A technique in ML that represents high-dimensional data in a lower-dimensional embedding space while preserving semantic relationships.
- The embedding vectors in the embedding space capture the semantic relationships and meaningful context.
- e.g. "Penguins are running behind the seal", "Eagles are flying very low today", "Dog is not feeling well".
- While reading the data model will understand the context and put everything in the category of Birds and Animals.
- Modern embedding computes embeddings for each token rather than the entire word to understand the context.

### **Why Use Embedding?**
- **Dimensionality Reduction:** Reduces computational complexity, making models faster and more efficient.
- **Semantic Preservation:** Ensures that similar items are close together in the embedding space.
- **Feature Learning:** Automatically extracts meaningful features from raw data.

### **Types of Embeddings**
1. Word Embeddings: 
- Represent words as dense vectors in a continuous space, capturing semantic relationships between words.
- Popular techniques include Word2Vec, GloVe, and FastText.

2. Document Embeddings: 
- Represent entire documents as a single vector, capturing the overall topic or theme of the document.

3. Image Embeddings: 
- Represent images as numerical vectors, capturing visual features like colour, texture, and shape.

4. Audio Embeddings: 
- Represent audio signals as numerical vectors, capturing acoustic features like pitch, timbre, and rhythm.
