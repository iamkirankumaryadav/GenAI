## What is an LLM?
- A Large Language Model trained on huge text data.
- A type of AI model designed to understand and generate human-like text based on the input it receives.

### Key Features of an LLM
- **Large**: Trained on massive datasets (text from books, articles, websites, etc.).
- **Language**: Focused on understanding and producing natural language.
- **Model**: Uses DL architectures, typically based on transformers.

### How It Works?
- LLMs learn patterns in language by predicting the next word in a sentence during training.
- They use billions (or even trillions) of parameters to capture context, grammar, facts, and reasoning.

### Examples
- GPT-4, GPT-5 (OpenAI)
- Claude (Anthropic)
- LLaMA (Meta)
- Gemini (Google)

### What Can They Do?
- Answer questions
- Summarize text
- Translate languages
- Generate code
- Assist in creative writing

### GPT (Generative Pre-trained Transformer)
- A series of LLM developed by OpenAI, based on the Transformer architecture, designed for various natural language processing tasks.

### Hugging Face
- A popular platform that provides pre-trained language models and tools to use and fine-tune LLM.
 
## How is GPT different from BERT?

GPT (Generative Pre-trained Transformer) | BERT (Bidirectional Encoder Representations from Transformers)
:--- | :---   
Decoder-only | Encoder-only 
Processes text left-to-right (unidirectional) | Reads text bidirectionally (both left and right full context).
Trained to predict the next word (autoregressive) | Trained with **Masked Language Modeling (MLM)** and **Next Sentence Prediction (NSP)**.
Good for text generation | Good for understanding context and classification tasks
Chatbots, creative writing, code generation, summarization | Sentiment analysis, question answering, named entity recognition (NER).

## What are Embeddings? 

- Embeddings are numerical representations of data (like words, sentences, or even images) in a continuous vector space.
- They capture the semantic meaning of the input so that similar items are close together in this space.

### Why Do We Need Embeddings?
- Computers can’t understand raw text, they need numbers.
- Embeddings allow models to work with text in a way that preserves **meaning** and **relationships**.

### Key Properties
- Each word/sentence is represented as a vector (e.g., 300 dimensions).
- Similar meanings → vectors are close together (low distance).
- Different meanings → vectors are far apart.

### Example
- Word2Vec, GloVe, and modern transformer embeddings.

### Where Are They Used?
- Search engines (semantic search)
- Recommendation systems
- Chatbots
- Clustering and classification

## What is Tokenization?
- **Tokenization** is the process of breaking down text into smaller units called **tokens**,
- which are the basic building blocks that a language model understands.

### What Are Tokens?
- A token can be:
- A **word** (e.g., "apple")
- A **subword** (e.g., "ap", "ple")
- A **character** (rare in modern LLMs)
- Example:  `"I love AI"` → `["I", "love", "AI"]`

### Why Tokenization Matters?
- LLMs don’t process raw text, they work with **token IDs** (numbers).
- Tokenization affects:
- **Model size** (vocabulary)
- **Efficiency**
- **Context length** (how many tokens fit in a prompt)

### Types of Tokenization
- **Word-level**: Splits by words (simple, but large vocab).
- **Subword-level**: Splits into smaller chunks (used in GPT, BERT).
- **Byte Pair Encoding (BPE)**: Common in GPT models.
- **SentencePiece**: Used in models like T5.

### Example with GPT
Text: `"ChatGPT is amazing!"`  
Tokens: `["Chat", "GPT", " is", " amazing", "!"]`  
Each token → mapped to an integer ID.

### Fine-Tuning
- The process of further training a pre-trained LLM on specific tasks or datasets to adapt it for domain-specific applications.

### Knowledge Distillation
- A technique in which a LLM transfers its knowledge to a smaller model to achieve similar performance with reduced computational resources.

### Language Modeling
- Language modeling is a task in which LLM learn to predict the probability of a word or token given its context within a sequence of text.

### Memory Mechanism
- Memory mechanisms in LLM enable them to store and retrieve information efficiently, enhancing their ability to handle long-range dependencies.

### Natural Language Understanding (NLU)
- NLU is a subfield of NLP that focuses on enabling machines to understand human language.

### ULMFiT (Universal Language Model Fine-Tuning)
- A transfer learning approach that fine-tunes a pre-trained language model for various NLP tasks.
- Transfer learning is the application of knowledge gained from pre-training LLM to improve performance with limited labeled data.
