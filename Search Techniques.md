# Search Techniques

## Keyword Search (Lexical Search) 
- Traditional Method: Matches exact characters or tokens.
- Exactly like a physical index in textbook or "Ctrl + F" on computer.
- Simple, fast, and accurate way to find information.

Works in 3 Steps:
1. The Index: Scans all the documents and makes a giant list of every word it finds.
2. The Match: It looks for exact string of characters in its list.
3. The Result: It shows every page where the specific word appears.

Elastic Search BM25 Algorithm
- Modern versions use algorithms like BM25 to rank results based on term frequency and document length.
- Rank pages higher if the keyword appears multiple times. 

## Vector Search
- A mathematical approach that can handle multiple data types (text, images, audio)
- It converts data into numerical "embeddings" (list of numbers: `[0.12, -0.59, 0.88, ...]`) using ML.
- Each position in the list represents a dimension (a specific feature)
- These vectors are stored in a vector database.
- When you search, it converts the query into a vector, calculates the distance between the query vector and all stored vectors.
- The system finds results whose vectors are closest to query vector using distance metric cosine similarity.
- Measures the angle between two arrows (vectors) starting from zero.
- If the angle is 0, the vectors are identical.
- If the angle is small, they are very similar.
- If the angle is 90, they have nothing in common.
- Best for Image retrieval, recommendation system, and multimodal search. Handles unstructured data and find similarity.
- Computationally expensive, requires embeddings if data updates.

## Semantic Search
- A data retrieval technique that focuses on the intent and contextual meaning behind a query rather than just matching individual words.
- While it often uses vector search as a tool, it includes additional layers like knowledge graphs and intent analysis.
- It interprets the meaning of a query and type of search rather than just the words.
- For example, it knows that "how to fix a flat" is related to "tire repair".
- Best for Chatbots, customer support, and question-answering systems (RAG).
- Understands natural language, context, and user intent.
- Complex to implement and can occasionally produce "false positives" (results that are semantically related but not what the user wanted). 

## Hybrid Search (Modern Standard) 
- Google Search
- Elastic Search
- Precision of Keyword Search and Contextual Intelligence of Vector Search
- When you enter a query, a hybrid search system runs two searches at the same time:
- The Keyword Search looks for exact words and phrases.
- The Vector Search looks for the meaning of your query (handling synonyms and natural language)
- System takes the results form both, fuses them together into a single list, and ranks the most relevant ones at the top.

## Search Technology Comparison
Feature |	Keyword Search | Vector Search | Semantic Search | Embeddings | Hybrid Search
:--- | :--- | :--- | :---- | :--- | :---
What |	A search for exact text matches. | A search for mathematical proximity. | A search for human intent. | The numerical representation of data. | A combination of Keyword + Vector.
How | Matches characters/tokens (ABC = ABC). | Calculates distance between points in space. | Interprets context and relationships. | Converts text/images into a list of numbers. | Runs two searches and merges the results.
Strength | Perfect for names, IDs, and specific codes. | Finds "similar" things without exact words. | Understands "What did the user actually mean?" | Essential "raw material" for Vector search. | The most accurate; covers all bases.
Weakness | Fails on synonyms (e.g., "fast" vs "quick"). | Can return "vaguely related" but wrong results. | Hardest and most expensive to build. | Useless on its own without a search engine. | Requires more computing power/setup.
Analogy | Looking up a word in a Dictionary index. | Finding a house by its GPS coordinates. | Asking a librarian for a recommendation. | The unique DNA of a piece of data. | Using a Map AND a local guide together.
