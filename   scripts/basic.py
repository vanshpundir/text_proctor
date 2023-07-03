from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity

# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


# Sentences we want sentence
sentences = ["""Supervised learning is a foundational machine learning concept where algorithms learn from labeled data to make predictions based on input features. It's widely used in image recognition, natural language processing, and more. The process starts with a labeled dataset, training the model to predict outputs for new data. The goal is to generalize well to unseen instances, capturing patterns from labeled examples to make accurate predictions.""", "A machine learning technique called unsupervised learning uses algorithms to learn from unlabeled data and make predictions based on input properties. for example, Kmean Clustering, heirarichal Clustering, etc. To create a mapping function, it entails gathering and preparing data, choosing and training models. The objective is to accurately anticipate unknown data using the training data's right responses.","acitivation is worng and spelling too"]

# Load model from HuggingFace Hub
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
model.to(torch.device("cuda"))

# Tokenize sentences
encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
encoded_input = encoded_input.to(torch.device("cuda"))

# Compute token embeddings for sentences
with torch.no_grad():
    model_output = model(**encoded_input)

# Perform pooling for sentences
sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

# Normalize sentence embeddings
sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

# Source sentence
source_sentence = """Supervised learning is a prominent and foundational concept in the field of machine learning. It involves training algorithms to learn from labeled data and make predictions based on input features. This technique has found extensive applications across various domains, including image recognition, natural language processing, speech recognition, medical diagnosis, and financial forecasting, among others.
At its core, supervised learning follows the notion of learning from examples. The process begins with a labeled dataset that consists of input-output pairs, where each input is a set of features, and each corresponding output is the desired target or label. The algorithm uses this labeled data to build a model that can accurately predict the correct output for unseen inputs.
The main objective of supervised learning is to generalize well to new, unseen data. The model is trained to capture patterns and relationships between the input features and the corresponding output labels. By learning from the labeled examples, the model can infer the underlying patterns and make predictions on new, previously unseen instances."""

# Tokenize the source sentence
encoded_source = tokenizer(source_sentence, padding=True, truncation=True, return_tensors='pt')
encoded_source = encoded_source.to(torch.device("cuda"))

# Compute token embeddings for the source sentence
with torch.no_grad():
    source_output = model(**encoded_source)

# Perform pooling for the source sentence
source_embedding = mean_pooling(source_output, encoded_source['attention_mask'])

# Normalize the source embedding
source_embedding = F.normalize(source_embedding, p=2, dim=1)

# Move tensors to CPU and convert to NumPy arrays
source_embedding = source_embedding.cpu().numpy()
sentence_embeddings = sentence_embeddings.cpu().numpy()

# Compute cosine similarity between the source embedding and sentence embeddings
similarity_scores = cosine_similarity(source_embedding, sentence_embeddings)

print("Similarity scores:")
print(similarity_scores)
