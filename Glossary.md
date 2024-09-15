# **Glossary of GenAI**

### **Agents (AI Assistant)**
- AI systems that can interact with their environment, learn from experience and perform tasks autonomously.
- GenAI agents are designed to understand the instructions and complete tasks using their knowledge and abilities.
- **Autonomy:** Agents can act independently, making decisions and taking actions without constant human intervention.
- **Adaptability:** They can learn from their experiences and adjust their behaviour accordingly.
- **Interaction:** Agents can interact with their environment, including other agents and humans.
- They are designed to mimic human-like behaviour and can be used for a variety of purposes.
- GenAI can access various resources, databases and APIs to gather information and complete tasks.
- Example: GenAI agents can plan a trip, find hotels, and flights, research activities, and make reservations.
- GenAI agents can be a powerful tool for boosting productivity by automating tasks and streamlining workflows.

**Types of agents**
1. **Rule-based agents:** Follow a set of predefined rules to make decisions.
2. **Reactive agents:** Respond directly to the environment without any consideration.
3. **Goal-based agents:** Have specific goals and plan their actions accordingly.
4. **Learning agents:** Can learn from experience and improve their performance over time.

### **Annotation**
- The process of labelling or tagging data. Used to train and fine-tune AI models.
- Text: POS Tagging, Sentiment Analysis | Image: Caption or labels | Audio: Transcripts or Lyrics.
- Annotation helps models to learn and understand the pattern, context, and meaning of data.

### **Artificial General Intelligence (AGI)**
- A hypothetical type of AI that would be much more powerful than we have today.
- **Human-like intelligence:** Learning, complex problem-solving, reasoning, and adapting to new situations.
- **Going beyond training:** It would be able to learn and perform new tasks without explicit programming.
- **Self-teaching/learning:** It would be able to learn and improve on its own without needing constant human intervention.
- **Scientific breakthroughs:** It could accelerate scientific discovery by analyzing vast amounts of data.
- **Boosting human productivity:** It can handle any intellectual task, freeing you up to focus on creative endeavours.

### **Artificial Super Intelligence (ASI)**
- ASI refers to a stage of AI that surpasses human intelligence across all fields.
- Including creativity, general wisdom, and problem-solving capabilities.
- **Unmatched cognitive abilities:** Process information at lightning speed. Analyze complex patterns instantaneously.
- **Self-improvement:** It could learn and potentially even rewrite/update/enhance its code to become even more intelligent.
- **Current state of ASI:** Purely hypothetical, actively debated, ethical considerations.
- **Potential of ASI:** Solving global problems (Climate, disease, poverty), scientific breakthroughs, boosting human potential.
- **Risks of ASI:** Existential threat, unforeseen consequences (actions we can't predict), job displacement.

### **Attention Mechanism | Self Attention:**
- The attention mechanism is a powerful technique used in various AI models to focus on important parts of the input data.
- How the attention mechanism works:
  1. Scanning the input data: See a variety of words in a sentence, or object in an image.
  2. Focusing on the query vector: Focus on the words or subject/topic which is most important in the query/prompt.
  3. Checking each vector: Compare with each word and see if it has a relevant relationship or information.
  4. Prioritizing the ones (Assigning weights): Relevant information gets a higher weight (more attention)
  5. Weighted sum: You collect the relevant information as compared to irrelevant ones.
- Example:
  - Machine Translation: You are translating "The red car is parked in the driveway". The model attends more to "red" and "car"
  - Text summarization: Summarizing a news article, the model attends to keywords like "earthquake" or "fire" to create a concise summary.
  - Question answering: Answer "What is the colour of the car?" The model attends to the colours to find the answer.
- By focusing on relevant parts of the data, the attention mechanism helps models make better predictions/decisions in various tasks.

### Autoregressive Model:
- Autoregressive (AR) models are statistical models that predict a variable's future value based on its past values.
- They assume that the current value is the linear combination of the past values with some random error term.
- The model is trained to maximize the likelihood of the next word given the preceding context.
- Application: Stock Price Prediction, collect data, determine the order, estimate coefficients and make predictions.

**Limitations of AR Model**
- **Stationarity:** AR models assume that the time series is stationary (statistical properties do not change over time)
- **Linearity:** The AR model assume a linear relationship between the current value and past values.
- **Limited forecasting horizon:** AR models may not be accurate for long-term predictions.

### **Bots (Automated Agents)** 
- Bots are automated programs that perform specific tasks or interact with users, often simulating human behaviour.
- They can be very simple or complex, but their core function is to automate repetitive actions.
- **Applications:** Chatbots, social media bots, search engine bots, recommendation bots, personal assistant bots.
- **Benefits:** Efficiency, 24/7 availability, consistency, scalability, save time and effort.

### **Chatbots**
- Chatbots are a specific type of bot designed for simple, rule-based system conversation with pre-defined responses.
- They use NLP and ML to understand and respond to user input/queries in a chat-like interface.
- Chatbots can be used for customer service, answering FAQs, scheduling appointments, or even providing entertainment.
- Examples: WhatsApp Meta AI, Google Message Chat Gemini Assistant, Apple Siri, and Career Website Chatbots.

### **ChatGPT (Generative Pre-trained Transformers)**
- ChatGPT leverages large language models (LLMs) to allow users to have human-like interactions.
- ChatGPT is trained to hold conversations that flow and adapt based on user input/queries/prompts.
- It can consider previous prompts (history) and replies to maintain context within a conversation.
- You can specify the length, format, style, and level of detail you prefer for the responses.
- Generates human-quality text, translates languages, writes different kinds of creative content, and answers questions.
- Have engaging conversations, brainstorm ideas, and compose emails, poems, articles, letters, codes, scripts, music, etc.
- ChatGPT is a powerful tool for conversation, creative exploration, and information gathering.
- GPT models are based on the transformer architecture, which processes and generates human-like text by learning from vast amounts of data.
- The pre-trained aspect refers to the initial extensive training these models undergo on large text corpora.
- Allowing them to understand and predict language patterns, context, and aspects of world knowledge.

### **Completions**

- Completions are the output produced by AI in response to a given input or prompt.
- When a user inputs a prompt, the AI model processes it and generates text that logically follows or completes the given input.
- Completions are based on the patterns, structures, and information the model has learned during its training phase on vast datasets.

### **Conversational AI**
- AI that enables computers to communicate and interact with humans in a natural language, such as text or voice.
- It's designed to simulate human conversation, making it feel like you're talking to another person.
- Enables computers to understand and respond to spoken or written language in a way that feels natural and engaging for users.
- NLP allows computers to interpret the meaning behind human language, including factors like context, slang, and sarcasm.
- Customer service chatbots, language learning apps, virtual assistance (Alexa, Siri, Google), social media bots, etc.
- Benefits: Improved customer experience, increased efficiency, personalized experiences, and accessibility.

### **Embeddings**
- Embeddings are a powerful tool for representing text data in a way that machines can understand and manipulate.
- Raw text data is difficult for machines to process directly. In an embedding, each word is a point in a vast space.
- Embeddings transform the text into a numerical representation. Condense text into a series of numbers/codes that capture meanings.
- They are carefully crafted to capture the semantic meaning of the text. Words with similar meanings will have similar embeddings.
- Allowing the model to understand the relationships, patterns, and contexts between the words.
- Embeddings represent text in a lower-dimensional embedding space compared to the original data.
- Each word is assigned a unique code, like a secret combination to find its location in the embedding space.
- This code isn't just random alphanumeric characters, it captures the meaning of the words.
- Words with similar meanings end up closer together in the space, the embeddings know they're related somehow.
- This makes it more efficient for AI models to process and analyze large amounts of text data.
- Embeddings help in search and retrieval, searching for information within a vast collection of documents.
- Embeddings can find documents with similar meaning to your query, even if they don't use the same words.
- Embeddings can identify the most important parts of a text document and generate concise summaries that capture the key points.
- Embeddings help to understand the intent behind a user's question and respond in a relevant and informative way.
- By converting text to a more manageable format, embeddings allow GenAI models to work faster and analyze larger datasets.
- Lower-dimensional embeddings require less storage space compared to raw text data, hence fast processing and less memory usage.
- Example: Find synonyms, and recommend similar products. Embeddings can be used for images too.

### **Encoder-Decoder**
- Encoder-decoders are like translators that can handle more than just words.
- **The Encoder:** Understanding the language, hears individual words, grasps the overall meaning, important ideas, and relationship between things. It gathers clues and pieces them together to form a condensed summary that captures the essence of the message.
- **The Decoder:** It receives the summarized message from the encoder (the context vector). The decoder uses its understanding of human language to craft a message that makes sense to humans. It considers the flow and order of words to create a natural-sounding explanation in our language.

**Understanding the concept:**
1. **Feeding the Encoder:** The encoder gets the message as input. This could be a sentence or an image they showed you.
2. **Making Sense of the Message:** The encoder analyzes the input piece by piece. It might use advanced techniques like recurrent neural networks (RNNs) to understand the sequence and how elements relate to each other.
3. **Encoding the Message:** As it processes the message, the encoder builds a context vector. This vector is like a cheat sheet with the key points and overall meaning extracted from the message.
4. **The Decoder:** Receives the context vector, like a summary briefing.
5. **Decoding into Human Terms:** Based on the context vector and its knowledge of the human language, the decoder starts generating our response. It might translate words, describe what it sees in an image, or even create a story based on the message.
6. **Building the Output:** The decoder might predict one word at a time, using the context vector and previous outputs to refine its understanding and create a coherent response.

**Applications:**
1. **Machine Translation:** Breaking down language barriers by translating between human languages.
2. **Text Summarization:** Condensing lengthy articles or documents into key points.
3. **Image Captioning:** Describing the content of an image in words.
4. **Text Generation:** Creating different creative text formats like poems, code, scripts, or musical pieces based on a given idea.

### **Few Shot Learning**
- A concept where the model is designed to learn and make accurate predictions or decisions based on a very limited amount of training data.
- Traditional machine learning models typically require large datasets to learn effectively.
- Few-shot learning techniques enable AI models to generalize from a small number of examples, often just a handful or even a single instance.
- This approach is especially valuable in situations where collecting large datasets is impractical or impossible, such as specialized academic fields or rare languages.

### **Fine Tuning**
- The process of taking a pre-trained AI model and further training it on a specific, often smaller, dataset to adapt it to particular tasks.
- Relevant in scenarios where a GenAI model, trained on varied datasets, needs to be specialized or optimized for specific applications.

### **Generative AI** 
- AI systems that can generate new content—such as texts, images, audio, and video—in response to prompts by a user.
- After being trained on an earlier set of data.
- **Text:** ChatGPT, Gemini, Meta AI | **Image:** DALL E, Ideogram, Midjourney | **Video:** Sora

### **Hallucinations** 
- Incorrect or misleading results that AI models generate. These errors can be caused by a variety of factors
- Factors: Insufficient training data, incorrect assumptions made by the model, or biases in the data used to train the model.
- The concept of AI hallucinations underscores the need for critical evaluation and verification of AI-generated information.

### **Inference** 
- The process where a trained AI model applies its learned knowledge to new, unseen data to make predictions, decisions, or generate content.
- It is essentially the phase where the AI model, after being trained on a large dataset, is now being used in real-world applications.
- During inference, the model utilises its learned patterns to perform the specific tasks it was designed for.
- Example: A language model that has been trained on a vast corpus of text can perform inference by generating a new essay, answering a student’s query, or summarizing a research article.

### **Large Language Models (LLMs)** 
- AI systems specifically designed to understand, generate, and interact with human language on a large scale.
- These models have been trained on enormous datasets since long time comprising a wide range of text sources.
- Enabling them to grasp the complexities, relationships, meanings and varied contexts of natural language.
- LLMs like GPT use particularly transformer architectures, to process and predict text sequences.

### **Model**
- Models are the algorithms that enable GenAI to process data, learn patterns, and perform tasks such as generating text, and images.
- Essentially, it is the core framework that embodies an AI’s learned knowledge and capabilities.
- A model is created through a process called training, where it is fed large amounts of data and learns to recognize patterns, make predictions, or generate outputs based on that data.
- Each model has its specific architecture and parameters, which define its abilities and limitations.

### Multimodal AI:
- Multimodal AI is like having different senses working together to understand the world, just like humans do.
- An AI that can process information from multiple sources, of multiple data types (Text, image, video, and audio)
- Real-world examples:
  1. Self-driving cars: Cameras (images), radar (signals from objects), LiDAR (3D laser scanning)
  2. Social media content moderation: AI can analyze text, images, audio, and videos to identify appropriate content.
  3. Virtual assistant: Image search, text prompts, voice commands (audio) and Google camera search with live translation.
  4. Medical diagnosis: AI can analyze medical images (X-rays, MRIs) and patient history (text) for more accurate diagnosis.
- By combining different types of information, multimodal AI gets a richer understanding of the world.
- AI can perform tasks that would be difficult for traditional AI / unimodal AI models.

### **Multi-Agent Framework:**

- A computational model that involves multiple autonomous agents interacting with each other and their environment to achieve a common goal.
- These agents are intelligent entities that can make decisions, act independently, and communicate with each other.

### Key Components of a Multi-Agent Framework
1. **Agents:** 
- The fundamental units of the framework. Agents have their own goals and actions.
- They can be software programs, hardware devices, or even human beings.
 
2. **Environment:** 
- The environment is the context in which the agents operate.
- It provides information to the agents and can be affected by their actions.

3. **Interaction:** 
- Agents interact with each other and the environment through communication and actions. 

4. **Architecture:** 
- The structure and organization of the agents and their interactions. It can be centralized, decentralized, or hybrid.

### Benefits of Multi-Agent Frameworks
* **Flexibility:** Multi-agent frameworks are highly flexible and can adapt to changing environments.
* **Scalability:** They can handle large-scale problems by distributing tasks among multiple agents.
* **Decentralization:** Agents can operate independently, reducing the risk of a single point of failure.
* **Emergent Behavior:** Complex behaviours can emerge from the interactions of simple agents.

### Applications of Multi-Agent Frameworks
* **Robotics:** Autonomous robots can coordinate their actions to accomplish tasks.
* **Artificial Intelligence:** Multi-agent systems can be used to model complex social and economic systems.
* **Game Theory:** Agents can be used to study strategic interactions and decision-making.
* **Distributed Systems:** Multi-agent frameworks can be used to design and manage distributed systems.

### **Natural Language Programming (NLP)**
- NLP is a field at the intersection of computer science, AI, and linguistics.
- Focused on enabling computers to understand, interpret, and generate human language in a way that is both meaningful and useful.
- The development of algorithms that can analyze, comprehend, and respond to text or voice data like how humans do.

### **Parameters**
- During training, a GenAI model is exposed to vast amounts of data.
- By adjusting the values of its parameters, the model learns to identify patterns and relationships within the data.
- It's like fine-tuning the connections between the nodes to achieve the desired outcome.
- Parameters are the core components that define the behaviour of the model and determine how it processes input data to produce output.
- GenAI has two main important parameters: Weights and biases associated with the neurons.
- Weights: A higher weight signifies a stronger connection and a greater impact on the final output.
- Biases: Act like thresholds, shifting the activation level, allowing the neuron to adjust its output independently of its input.
- The model adjusts these parameters to minimize the difference between its output and the actual data.
- The better these parameters are tuned, the more accurately the model can perform its intended task
- Increased Training Time: Training a model with billions of parameters can take days or even weeks on powerful hardware.
- Computational Demands: Running such models also requires significant computational resources.
- Overfitting: With too many parameters, it can become overly focused on the training data and struggle to perform well on unseen data.

### **Positional Encoding**  
- A technique used in ML models, particularly transformers, to address the order of elements within a sequence.
- Unlike RNNs, which process information sequentially, transformers handle all elements in a sequence simultaneously.

**The Problem: Missing Order Information**
- Imagine you have a sentence like "The quick brown fox jumps over the lazy dog."
- A transformer model sees all the words at once, like looking at a jumbled mess of letters.
- Without understanding the order, it can't grasp the relationships between words or the overall meaning of the sentence. 

**The Solution: Adding Positional Information**
- Positional encoding assigns a unique code to each element (word) in the sequence.
- This code doesn't just contain random numbers, it captures the element's position in a way the model can understand.
- Think of it like seat numbers on an aeroplane. Each word gets a unique "seat number" that tells the model its position relative to other words.

**Creating the Codes: Sine and Cosine Functions**
- The specific code for each position is often calculated using sine and cosine functions.
- These functions create a wave-like pattern that repeats over the sequence length.
- The advantage of using sine and cosine functions is that they can represent a wide range of positions effectively.

**Benefits of Positional Encoding**
- **Improved Understanding of Order:** The model can learn how the order of elements affects the overall meaning.
- **Better Performance in Sequence Tasks:** Positional encoding has significantly improved the performance of transformers in tasks like machine translation, text summarization, and question answering.
- **Flexibility:** The core concept of adding positional information remains crucial for transformers to function effectively with sequences.

### **Prompt**
- A prompt is the input given to an AI model to initiate or guide its generation process.
- This input acts as a set of instructions that the AI uses to produce its output.
- Prompts are crucial in defining the nature, scope, and specificity of the output generated by the AI system.
- For instance, in a text-based GenAI model like GPT, a prompt could be a sentence or a question that the model then completes or answers in a coherent and contextually appropriate manner.

### **Prompt Engineering**
- Prompt engineering refers to the crafting of input prompts to effectively guide AI models in producing specific and desired outputs.
- This practice involves formulating and structuring prompts to leverage the AI’s understanding and capabilities.
- Optimizing the relevance, accuracy, and quality of the generated content.

### **Reinforcement Learning**
- A type of ML where an agent learns to make decisions by performing actions in an environment to achieve a certain goal.
- The learning process is guided by feedback in the form of rewards or punishments.
- Positive reinforcement for desired actions and negative reinforcement for undesired actions.
- The agent learns to maximize its cumulative reward through trial and error, gradually improving its strategy over time.

### **Retrieval Augmented Generation (RAG)**
- A technique that combines the strengths of LLMs with external knowledge sources beyond its training data.
- It enhances the accuracy, relevance, and factuality of AI-generated text by providing the model with up-to-date information.
- User Input: When a user asks a question, query or provide a prompt.
- LLM Query Generation: The LLM analyzes your input and generates a query to search the information retrieval system.
- The system retrieves relevant information from an external knowledge base.
- This could be a database, document repository, access to cloud storage (Drive) or any other structured or unstructured data source.
- Information Retrieval: The system searches its database and retrieves relevant documents or passages that match the LLM's query.
- Augmented Input: The retrieved information is then incorporated/combined with the original LLM prompt.
- Response Generation: The LLM, now armed with both your original prompt and the retrieved information, generates its final response.
- Benefits: Factual consistency, improved and relevant response, domain adaptibility.

### Semantic Model
- A way to represent information in a structured way that captures the meaning or semantics of the data.
- It's like creating a map of concepts and how they relate to each other.
- This model is often used in fields like AI, NLP, and knowledge management.
- Think of it like a family tree:
  -  **Concepts (Entities):** The people in the family (e.g., 'person', 'parent', 'child')
  -  **Relationships:** The connections between people (e.g., 'is a parent of', 'is a child of')
  -  **Attributes:** The characteristics of each person (e.g., 'name', 'age', 'occupation')
      
### **Semantic Network** 
- A knowledge representation method used in AI to model relationships between concepts.
- It's like a mind map, where concepts are like bubbles and the connections between them show how they're related. 
- Nodes: These represent the concepts, entities, objects (car, house) or abstract ideas (love, freedom).
- Links: These connect the nodes and represent the relationships between the concepts by labelling.

### **Softmax Output**
- In the context of neural networks, is the result of applying the softmax function to the final layer of a network with multiple output classes.
- This function takes a vector of real numbers as input and transforms them into a probability distribution across those classes.
- **Multiple Classes:** Softmax is typically used in tasks where the network needs to classify an input into one of several discrete categories.     - Example: An image recognition model might have multiple classes for different objects (cat, dog, car, etc.).
- **Probabilities:** Unlike the raw output values from the network, which might be simple scores, the softmax output gives you a probability for each class. These probabilities range between 0 and 1, and they all sum up to 1.
- **Interpreting the Output:** The highest value in the softmax output represents the class that the network is most confident the input belongs to. The closer a value is to 1, the higher the confidence. Conversely, values closer to 0 indicate lower confidence.
- For instance, imagine a softmax output vector: [0.1, 0.8, 0.1]. Here, the second element (0.8) is the highest, suggesting the network is 80% confident the input belongs to class 2. The other classes have a much lower probability (10% each).

**Why Softmax?**
- **Clear Interpretation:** Softmax probabilities are easy to understand and interpret. You can see at a glance which class the network favours and how certain it is about its prediction.
- **Comparison of Classes:** Softmax allows you to compare the likelihood of the input belonging to different classes. This is crucial for tasks where you need to know not just the most likely class, but also the confidence level in that prediction.
- **In essence, softmax output provides a probabilistic interpretation of the network's final prediction, making it a valuable tool for tasks involving multi-class classification.**

### **Temperature**
- A parameter that controls the randomness or creativity in a model‘s responses.
- When generating text, a higher temperature value leads to more varied and unpredictable outputs.
- While a lower temperature results in more conservative and expected responses.

### **Tokens**
- Tokens are the smallest units of data that an AI model processes.
- In NLP, tokens typically represent words, punctuations, or even individual characters, depending on the tokenization method used.
- Tokenization is the process of converting text into smaller, manageable units for the AI to analyze and understand.

### **Training**
- Training is the process by which an ML model, such as a neural network, learns to perform a specific task.
- This is achieved by exposing the model to a large set of data, known as the training dataset.
- Allowing it to iteratively adjust its internal parameters to minimize errors in its output.
- During training, the model makes predictions or generates outputs based on its current state.
- These outputs are then compared to the desired results, and the difference (or error) is used to adjust the model’s parameters.
- This process is repeated numerous times, with the model gradually improving its accuracy and ability to perform the task.
- For example, a language model is trained on vast amounts of text so that it learns to understand and generate human-like language.

### **Transformers** 
- A specific type of neural network architecture that has revolutionized how machines process and generate text.
- Unlike traditional neural networks that process information one piece at a time.
- Transformers excel at handling sequences, like sentences or code.
- They can analyze the entire sequence simultaneously, allowing them to capture relationships between words or elements within the sequence.
- There are two main components to a transformer:
- **Encoder:** Takes the input sequence and processes it, capturing the meaning and relationships between the elements. It essentially creates a condensed representation of the sequence.
- **Decoder:** Takes the encoded representation from the encoder and uses it to generate an output sequence, like a translated sentence or a continuation (completion) of a story.
- A key feature of transformers is the concept of attention.
- The attention mechanism allows the model to focus on the most relevant parts of the input sequence for a specific task.
- Transformers can capture relationships between words even if they are far apart in a sequence.
- The ability to process the entire sequence simultaneously allows for parallel processing, making transformers faster and more efficient than older architectures.
- Applications: Chatbots, machine translation, text summarization

### **Tuning**
- The process of adjusting a pre-trained model to better suit a specific task or set of data.
- This involves modifying the model’s parameters so that it can more effectively process, understand, and generate information relevant to a particular application.
- Tuning is different from the initial training phase, where a model learns from a large, diverse dataset.
- Instead, it focuses on refining the model’s capabilities based on a more targeted dataset or specific performance objectives.

### **Zero-Shot Learning**
- A concept where an AI model learns to perform tasks that it has not explicitly been trained to do.
- Unlike traditional ML methods that require examples from each class or category they’re expected to handle,
- Zero-shot learning enables the model to generalize from its training and make inferences about new, unseen categories.
- This is achieved by training the model to understand and relate abstract concepts or attributes that can be applied broadly.
- For instance, a model trained in zero-shot learning could categorize animals it has never seen before. e.g. Unicorn
- It infers knowledge about these new categories by relying on its understanding of the relationships and similarities between different concepts.
