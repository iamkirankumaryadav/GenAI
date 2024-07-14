# **Vector Databases**

### **Data as Vectors:**  
- Instead of rows and columns, vector databases store data points as vectors.
- These vectors are like coordinates in high-dimensional space, with each coordinate representing a specific data feature.

### **Similarity Searches:**  
- The power of vector databases lies in finding similar data points.
- They use special indexing techniques to search for vectors closest to a query vector, even if they aren't exact matches. 

### **Unstructured Data:**  
- Traditional databases struggle with unstructured data like text, images, and sensor readings.
- Vector databases excel here, as these data types can be converted into vectors that capture their essential meaning.
- A traditional database might store pictures based on filename or information.
- A vector database could find similar images based on what's actually in the picture, like colour, objects, or scenes.

### Applications of vector databases:
* **Recommendation Systems:** Recommending products, articles, or music based on similar user preferences.
* **Image and Video Search:** Finding similar images or videos based on visual content. Example: Google Image Search.
* **Natural Language Processing (NLP):**  Understanding the meaning of text and enabling tasks like chatbots and machine translation.
* **Fraud Detection:** Identifying fraudulent transactions based on patterns in historical data.

### Embeddings in Vector Databases: Storage, Optimization, and Retrieval

- Vector databases are built to handle the unique needs of embeddings, those high-dimensional vectors capturing the essence of data points.

### **Storage:**

1. **Compression:**
- Since vectors can be large, techniques like product quantization reduce storage size by representing the vector with smaller, lower-dimensional codes.

2. **Product quantization (PQ)**
- A technique for compressing high-dimensional vectors used in vector databases.
- It significantly reduces the memory footprint of these vectors, allowing you to store and process larger datasets more efficiently.

3. **Inverted Indexing:**
- Similar to text databases, vectors are associated with metadata (tags, labels) for easier retrieval based on these attributes.

### **Optimization:**

**Indexing Strategies:** Unlike traditional databases, vector databases use specialized indexing methods like:
* **Hierarchical Navigable Small World (HNSW):** Creates a graph and index-based network structure for efficient navigation towards similar vectors.
* **Inverted File (IVF):** Partitions data into smaller clusters, allowing faster searches within relevant groups.
* **Locality-Sensitive Hashing (LSH):** Hashes similar vectors to the same bucket, enabling faster approximate nearest neighbour searches.

### **Retrieval:**

1. **Similarity Search:**
- The core function! Vector databases perform efficient similarity searches.
- You provide a query vector, and the database retrieves the closest matching vectors based on a distance metric (like cosine similarity). 

2. **Approximate Nearest Neighbors (ANN):**
- Due to the high dimensionality of vectors, exact matches can be computationally expensive.
- Vector databases often focus on finding very close matches (nearest neighbours) to achieve a balance between speed and accuracy.

### **Trade-off:**
1. **Accuracy vs. Speed:**
- More accurate searches take longer. Vector databases allow you to tune this based on your specific needs.

2. **Storage vs. Performance:**
- Compression techniques reduce storage but might impact retrieval speed slightly.

