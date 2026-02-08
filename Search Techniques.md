# Search Techniques

## Keyword Search (Lexical Search) 
- **Definition:** Matches exact characters or tokens like a textbook index or "Ctrl + F."
- **Strengths:** Provides a simple, fast, and accurate way to find specific strings.
- **Weaknesses:** Lacks contextual understanding and cannot process word meanings or synonyms.
- **Step 1 (Index):** Scans all documents to create a comprehensive list of every unique word.
- **Step 2 (Match):** Searches the index for an exact character-for-character string match.
- **Step 3 (Result):** Displays every document or page where that specific keyword is found.
- **Ranking (BM25):** Uses the BM25 algorithm to rank results based on keyword frequency and document length. 

## Vector Search (Semantic Similarity)
- **Definition:** A mathematical approach using ML (vectorization) to handle text, images, and audio.
- **Embeddings:** Converts data into numerical vectors (lists of numbers) where each position represents a specific feature.
- **Storage:** Saves these numerical representations within a specialized vector database.
- **Process:** Converts queries into vectors and calculates the mathematical distance between them and the stored data.
- **Distance Metric:** Uses "Cosine Similarity" to find results by measuring the angle between query and data vectors.
- **Cosine Scale:** An angle of 0° indicates identical content, small angles show high similarity, and 90° indicates no relation.
- **Applications:** Ideal for recommendation systems, image retrieval, and multimodal searches involving unstructured data.
- **Trade-offs:** Effectively finds conceptual similarities but is computationally expensive and requires new embeddings for every update.

## Semantic Search
- **Definition:** A retrieval technique focused on intent and contextual meaning rather than exact word matching.
- **Methodology:** Vector search with advanced layers like knowledge graphs and intent analysis.
- **Applications:** Powers chatbots, customer support tools, and Retrieval-Augmented Generation (RAG) systems.
- **Strengths:** Excels at understanding natural language, nuances of context, and specific user intent.
- **Weaknesses:** More complex to implement and prone to "false positives" that are related but irrelevant.
- **Core Function:** Transforms data into meaningful embeddings to find results based on conceptual similarity.

## Hybrid Search (Modern Standard) 
- **Definition:** A combination used by Google and Elasticsearch that merges keyword precision with vector intelligence.
- **Simultaneous Processing:** Runs an exact-match lexical search and a conceptual vector search at the same time.
- **Keyword Role:** Identifies specific words, phrases, and technical terms for high precision.
- **Vector Role:** Interprets query meaning, handles synonyms, and processes natural language.
- **Fusion and Ranking:** Merges results from both methods into a single list to prioritize the most relevant content.
- **Primary Benefit:** Captures both specific jargon and broad intent to provide the most comprehensive results.

## Search Technology Comparison
Feature |	Keyword Search | Vector Search | Semantic Search | Embeddings | Hybrid Search
:--- | :--- | :--- | :---- | :--- | :---
What |	A search for exact text matches. | A search for mathematical proximity. | A search for human intent. | The numerical representation of data. | A combination of Keyword + Vector.
How | Matches characters/tokens (ABC = ABC). | Calculates distance between points in space. | Interprets context and relationships. | Converts text/images into a list of numbers. | Runs two searches and merges the results.
Strength | Perfect for names, IDs, and specific codes. | Finds "similar" things without exact words. | Understands "What did the user actually mean?" | Essential "raw material" for Vector search. | The most accurate; covers all bases.
Weakness | Fails on synonyms (e.g., "fast" vs "quick"). | Can return "vaguely related" but wrong results. | Hardest and most expensive to build. | Useless on its own without a search engine. | Requires more computing power/setup.
Analogy | Looking up a word in a Dictionary index. | Finding a house by its GPS coordinates. | Asking a librarian for a recommendation. | The unique DNA of a piece of data. | Using a Map AND a local guide together.

## How does Vector Search works exactly?
- Vector search works by transforming raw data into high-dimensional numerical coordinates called embeddings.
- Instead of matching words, it measures the mathematical distance between these points to find conceptual similarity.
- **Embedding Generation:** An ML model (BERT for text or ResNet for images) converts the input into a vector, a list of numbers where each position represents a specific hidden feature.
- **Indexing:** To search billions of items in milliseconds, a Vector Database uses Approximate Nearest Neighbor (ANN) algorithms.
- **HNSW (Hierarchical Navigable Small World):** A graph-based structure that creates "shortcuts" to navigate large datasets quickly.
- **IVF (Inverted File Index):** Clusters similar vectors together to narrow the search area.
- **Querying:** Your search term is converted into a vector using the same model. The system then calculates which stored vectors are mathematically "closest" to your query vector.
- **Similarity Measurement:** The "closeness" is determined using a Distance Metric.
- **Cosine Similarity:** Measures the angle between vectors, ideal for text where content direction matters more than document length.
- **Euclidean Distance:** Measures the straight-line distance, best for image analysis or cases where absolute values (like pixel intensity) are key.
- **Dot Product:** Considers both direction and magnitude; often used when vectors are normalized to unit length.
- **Semantic Matching:** It understands that "canine" is similar to "dog" even though they share no common characters.
- **Multimodal:** It can compare different data types (e.g., searching for images using a text description) because they all live in the same mathematical space.
- **Scalability:** Modern systems use libraries like the FAISS Library to handle massive datasets with sub-second latency. 


