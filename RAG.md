# Retrieval Augmented Generation (RAG)
- An AI framework that combines the strengths of traditional information retrieval systems (such as search and databases) with the capabilities of generative large language models (LLMs).
- By combining your data and world knowledge with LLM, generated outputs are more accurate, up-to-date, and relevant to your specific needs.

## Knowledge Base
- Knowledge Bases uses Retrieval-Augmented Generation (RAG). Here's how it works:

1. Document Processing:
- Upload your documents to Amazon S3
- The system creates embeddings (numerical representations of text) using models like Amazon Titan Text Embeddings
- Documents are chunked into manageable pieces

2. Vector Storage
- Those embeddings get stored in a vector database (like Amazon OpenSearch Serverless)
- This enables semantic searching - finding content based on meaning, not just keywords

3. Query Processing
- When someone asks a question:
- Their question gets converted to an embedding
- The system finds similar vectors in the database based on semantic search
- A foundation model generates a response using the retrieved content
- Responses include citations back to source documents

## Setting Up Your Knowledge Base
1. Initial Setup
- When creating a new knowledge base, you'll need to provide a name and handle the creation of necessary IAM roles to enable interactions between services like Amazon S3 and Amazon OpenSearch Service.

2. Configure Data Source
- Your knowledge base needs data to work with. e.g. Amazon S3 as the primary data source for storing runbooks, user guides, etc.
- You'll need to specify the location of your documents and give your data source a name.

3. Choose Parsing Strategy
- The parsing strategy determines how LLM extracts content from your documents.
- The default parser handles common file types like text and PDFs,
- Advanced options like Data Automation can handle multimodal content including charts and images.
- Choose based on your document types and extraction needs.

4. Select Chunking Strategy
- Chunking is crucial for effective retrieval.
- The default strategy breaks content into 300-token chunks while preserving sentence boundaries.
- Whether you choose default, fixed size, hierarchical, semantic, or no chunking depends on your document structure and how you want information to be retrieved.

5. Configure Models and Storage
- Your knowledge base needs two key components: an embedding model to convert text into vectors (like Amazon Titan Text Embeddings v2) and a vector database to store these embeddings (like Amazon OpenSearch Serverless).
- These work together to enable semantic search capabilities.

6. Final Steps
- After reviewing and creating your knowledge base, you'll need to sync your data source or upload/ingest to generate embeddings.
- Once syncing is complete, you can start using the RetrieveAndGenerate API to query your knowledge base and get responses based on your private content.

## Real-World Applications
Knowledge Bases can power various use cases:
- DevOps documentation search
- Customer support systems
- Sales enablement
- Product training
- Security operations
- HR assistance

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
