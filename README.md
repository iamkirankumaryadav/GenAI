# GenAI
Generative Artificial Intelligence

### **Generative AI**
- An AI technology that can create new content in text, image, music, and video forms.
- Businesses are exceptionally excited by the possibilities of specialized generative AI models,
- because they can be trained in an organization's corporate data and they can collaborate with research and advanced analytics teams to boost productivity.
- The best thing is their ability to write human-like text and extract complex ideas synthesized from the vast volume of information on which they are trained.
- The powerhouse behind creating entirely new data, like text formats, images, music, or even code.
- Innovations in GenAI are accelerating at a breathtakingly fast speed as businesses all over the world experiment with it.
- GenAI has launched a new era in technology, that promises both greater productivity and new ways to solve business problems.

### **Machine Learning**
- The training method for generative AI models.
- We feed the model a massive amount of existing data, and the model learns the patterns and relationships within that data.

### **Large Language Models (LLMs)** 
- Text-based generative AI. 
- LLMs are trained on massive amounts of text data and can generate new text formats, translate languages, write different creative content, and answer your questions in an informative way.

### **Generative Adversarial Networks (GANs)**  
- A deep learning architecture, that trains two neural networks to complete against each other to improve and generate more authentic new data.
- One network (generator) creates new data, like images of faces, while the other (discriminator) tries to identify if it's real or generated.
- This competition with the discriminator helps the generator create increasingly realistic and authentic outputs.

**How do GANs work?**

1. **Generator:** The generator's job is to create new data (like images of faces) that resemble the real data from the training dataset.
2. **Discriminator:** The discriminator's job is to analyze the data (both real and generated) and determine if it's realistic or fake.

**The Competitive Loop:**

1. **The Generator Makes a Guess:** 
- The generator starts creating new data samples based on its given training dataset.

2. **The Discriminator Judges:** 
- The discriminator receives both real data (from the training dataset) and the generated data from the generator.
- It analyzes both and tries to classify them as real or fake.

3. **Learning from Each Other:** 
- Based on the discriminator's feedback, the generator refines its approach to create more realistic data in the next round.

**Usecases of GANs**
- Generate realistic images through text based prompts or by modifying existing images.
- GAN can also edit images: Converting low resolution images to a high resolution image. Turning a black & white image to color.
- GAN can be used for data augmentation to create synthetic data with all the attributes of real world data.
- GANs can also create realistic faces, characters, and animals for animation and video.
- In ML, data augmentation artificially increases the training set by creating modified copies of a dataset using existing data.
- We can use GenAI model to accurately guess and complete some missing information in a dataset.
- GANs can also generate the remaining or incomplete portion/surface of an image.
- GANs can generate 3D models from 2D photo or scanned images.
- In healthcare it combines X-rays and other body scans to create realistic images of organs for surgucal planning and simulation.

### **Foundation Models**
- The versatile tools of generative AI. A toolbox with many different AI techniques.
- The building blocks for various generative AI applications.
- A versatile and powerful base that can be customized for specific tasks across different data types (Text, Image, Video, Audio, etc)
- They can be trained on various data types (text, code, images) and then adapted for specific tasks, like writing different creative text formats or generating images based on a description.

**How does the foundation model work?**

1. **Training on Diverse Data:**
- Foundation models are trained on a massive and varied dataset. This data can include text documents, code, and even images.

2. **Learning Underlying Patterns:**  
- The foundation model learns the underlying patterns and relationships within the data, regardless of the type.

3. **Adaptability is Key:** 
- Once trained, they can be applied to various tasks.
- The foundation model can be fine-tuned for specific generative AI applications.

**Examples:**

1. **Creative Text Generation:** 
- Imagine a company wanting an AI assistant that can write different creative text formats like poems, scripts, or marketing copy.
- They can take a foundation model trained on massive amounts of text data and fine-tune it for these specific tasks.

2. **Image Enhancement:**
- A photo editing app that enhances or restores old photos could leverage a foundation model.
- The developers could use a foundation model trained on various images and then adapt it to recognize specific features in photos (like faces or landscapes) and improve their quality.

3. **Advanced Machine Translation:** 
- A foundation model trained on a massive dataset of text and code in multiple languages could be fine-tuned to become a super-accurate machine translation tool.

### **Latent Space** 
- A mysterious universe where Neural Networks learn and store information.
- A compressed version of the data learned by a generative model.
- A multi-dimensional unintuitive and unimaginable space (Only computable and possible technically)
- A more manageable representation that the model can use to efficiently generate new content.

**Example:**
- Imagine you asked your image generation tool to create an image of Tom Cruise.
- Between my typing (describing the image) and the image created on the screen whatever happens in that involuntary, inaccessible, intermediate, and invisible space is called a latent space.
- When you train a generative AI model on a massive dataset (like images), the data itself can be very complex and high-dimensional.
- Imagine millions of images with all their details about colours, shapes, and textures.
- Latent space comes in to simplify things. The model essentially learns to represent this complex data in a lower-dimensional space.
- Think of it like compressing all those image details into a more manageable form, like a map with key features.
- It acts like a compressed version of the original data, but it still holds the important information needed for generating new content.

### **Unsupervised Learning** 
- A machine learning technique where the model learns patterns from unlabeled data (data without predefined categories).
- Many generative AI models use unsupervised learning to discover hidden structures within the data they are trained on.

### **Autoencoder**
- A type of artificial neural network used in generative AI.
- Learning efficient representations of data for tasks like dimensionality reduction, anomaly detection, and even, as a stepping stone for more advanced generative models.
- An autoencoder studies a piece of art (data) and then tries to recreate it (reconstruction) using just the key features they've captured (compressed representation).
- Autoencoders themselves don't directly generate entirely new data, they play a crucial role in generative AI by helping models understand and represent complex data efficiently.
- This understanding becomes the foundation for creating new and interesting content. 

**How do autoencoders work?**

1. **Encoder:** 
- It takes the input data and compresses it into a smaller representation, capturing the essential features.
- This compressed version is called the latent space. 

2. **Decoder:** 
- It takes the compressed representation from the encoder (the sketch) and expands it back into a format similar to the original data.
- Tries to recreate the content very close to the original one.

3. **Training Process:** 
- During training, the autoencoder is shown a lot of data examples.
- It tries to encode the data into a latent space and then decode it back into a reconstructed version as close to the original as possible.
- The autoencoder keeps adjusting its encoder and decoder based on how well the reconstruction matches the original data.

**Applications:** 

1. **Dimensionality Reduction:** 
- Autoencoders can be used to reduce the number of features needed to represent the data.
- This can help speed up other machine learning algorithms.

2. **Anomaly Detection:** 
- Autoencoders can identify unusual data points that don't fit the patterns learned during training.

3. **Pre-training for Generative Models:** 
- The latent space learned by autoencoders can be a good starting point for training more advanced generative models. 

[**Important Terms**](https://www.analyticsvidhya.com/blog/2024/01/generative-ai-terms/)

### **RLHF Reinforcement Learning with Human Feedback**
- A technique that combines reinforcement learning (RL) with human input to train AI models.
- Imagine training a robot dog with treats (rewards) and verbal commands (feedback).

**Core Idea:**

#### 1. **Traditional Reinforcement Learning:** 
- In standard reinforcement learning, an AI model interacts with an environment and learns through trial and error.
- It receives rewards for good actions and penalties for bad ones.
- The goal is for the model to learn a policy that maximizes rewards.

#### 2. **Adding the Human Touch:**  
- In RLHF, humans provide additional feedback beyond just rewards. This feedback can be in various forms, like:
- **Explicit ratings:** Humans might rate the model's outputs (like text generation) on a scale of good to bad.
- **Choosing preferred options:** Humans might choose which of two outputs from the model they prefer. 

**The Training Process:**

1. **The Model Takes Action:** The RLHF model acts, like generating text, translating a language, or creating an image.
2. **Human Feedback is Provided:** Humans evaluate the model's output and provide feedback.
3. **The Model Learns:** Based on the feedback (rewards, ratings, or preferences), the model adjusts its internal parameters to improve its performance in the next round.

**Benefits of RLHF:**

1. **Handling Complex Tasks:** 
- RLHF is particularly useful for tasks where defining a reward function (a system for assigning rewards) is difficult.
- For example, it's hard to define what makes a joke "funny" in mathematical terms, but humans can easily rate jokes. 

2. **Human Values and Preferences:**  
- By incorporating human feedback, RLHF ensures that the model learns behaviours and outputs that align with human values and preferences.
