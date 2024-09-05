# **Vector Databases**

- Vector databases are specialized databases designed to efficiently store and retrieve high-dimensional numerical vectors.
- Vectors are representations of complex data (text, images, audio) that have been encoded into a numerical format using embeddings.
- A powerful tool for working with high-dimensional data and enabling applications that rely on semantic understanding and similarity search.

### **Key Features and Benefits:**

1. **Similarity Search:** 
- Vector databases excel at finding the most similar vectors to a given query vector.
- This is crucial for applications like semantic search, recommendation systems, and anomaly detection.

2. **High-Dimensional Data:** 
- They can handle data with thousands or even millions of dimensions, which is common in many ML applications.

3. **Scalability:** 
- Vector databases are designed to scale efficiently, allowing them to handle large datasets and heavy workloads.

4. Performance Optimization: 
- They employ specialized indexing techniques and query optimization algorithms to deliver fast and accurate results.

### **Data as Vectors:**  
- Instead of rows and columns, vector databases store data (text, image, audio, video, etc) as vectors.
- These vectors are like coordinates in high-dimensional space, with each coordinate representing a specific data feature.

### **Similarity Searches:**  
- The power of vector databases lies in finding similar data points.
- They use special indexing techniques to search for vectors closest to a query vector, even if they aren't exact matches. 

### **Unstructured Data:**  
- Traditional databases struggle with unstructured data like text, images, audio, video, and sensor readings.
- Vector databases excel here, as these data types can be converted into vectors that capture their essential meaning.
- A traditional database might store pictures based on filename, index, memory location or other metadata information.
- A vector database could find similar images based on what's actually in the picture, like colour, objects, or scenes.

### **Applications of vector databases:**
* **Recommendation Systems:** Recommending products, articles, or music based on similar user preferences.
* **Image and Video Search:** Finding similar images or videos based on visual content. Example: Google Image Search.
* **Natural Language Processing (NLP):**  Understanding the meaning of text and enabling tasks like chatbots and machine translation.
* **Fraud Detection:** Identifying fraudulent transactions based on patterns in historical data.

### Embeddings in Vector Databases: Storage, Optimization, and Retrieval

### **Storage:**

1. **Compression:**
- Techniques like product quantization reduce storage size by representing the vector with smaller, lower-dimensional codes.

2. **Product quantization (PQ)**
- It significantly reduces the memory footprint of these vectors, allowing you to store and process larger datasets more efficiently.

3. **Inverted Indexing:**
- Similar to text databases, vectors are associated with metadata (tags, labels) for easier retrieval based on these attributes.

### **Optimization:**
**Indexing Strategies:** Unlike traditional databases, vector databases use specialized indexing methods like:
* **Hierarchical Navigable Small World (HNSW):** Creates a graph and index network for efficient navigation towards similar vectors.
* **Inverted File (IVF):** Partitions data into smaller clusters, allowing faster searches within relevant groups.
* **Locality-Sensitive Hashing (LSH):** Hashes similar vectors to the same bucket, enabling faster approximate nearest neighbour searches.

### **Hashing**  
- A process of transforming a large input (like a text string or a file) into a fixed-size output (called a hash value).
- This output is typically much smaller than the original input, the same input will always produce the same hash value.

### **Why is hashing used?**

1. **Data integrity:** 
- Hashing is used to verify the integrity of data. If the hash of a file changes, it means the file has been modified.

2. **Password storage:** 
- Passwords are often stored as hashes to protect their security.
- If a database containing hashed passwords is compromised, the actual passwords cannot be recovered.

3. **Indexing:** 
- Hashing can be used to create indexes for data structures like hash tables, which provide efficient data retrieval.

4. **Cryptography:** 
- Hashing is a fundamental component of many cryptographic algorithms (Encrypt and decrypt data)
- Digital signatures, secret key, personal access token, public key and message authentication codes.

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

