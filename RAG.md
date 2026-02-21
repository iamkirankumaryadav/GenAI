# Retrieval Augmented Generation (RAG)
- An AI framework that combines the strengths of traditional information retrieval systems (such as search and databases) with the capabilities of generative large language models (LLMs).
- By combining your data and world knowledge with LLM, generated outputs are more accurate, up-to-date, and relevant to your specific needs.

### Why use RAG?
- LLMs are limited to their pre-trained data, this leads to outdated responses.
- RAG overcomes this by providing up-to-date information to LLMs.
- LLMs are powerful tools for generating creative and engaging text, but they can sometimes struggle with factual accuracy.
- This is because LLMs are trained on massive amounts of text data, which may contain inaccuracies or biases.
- RAGs usually retrieve facts via search, and modern search engines now leverage vector databases to efficiently retrieve relevant documents.
- Vector databases store documents as embeddings in a high-dimensional space, allowing for fast and accurate retrieval based on semantic similarity.
- Multi-modal embeddings can be used for images, audio and video, and more and these media embeddings can be retrieved alongside text embeddings or multi-language embeddings.
- Advanced search engines like Vertex AI Search use semantic search and keyword search together (called hybrid search)
- A re-ranker which scores search results to ensure the top returned results are the most relevant.
- Additionally searches perform better with a clear, focused query without misspellings, so prior to lookup, sophisticated search engines will transform a query and fix spelling mistakes.
- RAG helps to minimize contradictions and inconsistencies in the generated text.
- sThis significantly improves the quality of the generated text, and improves the user experience.
