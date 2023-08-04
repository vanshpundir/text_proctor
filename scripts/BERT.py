from transformers import BertTokenizer, BertModel
import torch

# Load BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained("distilbert-base-uncased")
model = BertModel.from_pretrained("distilbert-base-uncased")

# Two sample sentences
sentence1 = "I am from India."
sentence2 = "I am not from India."

# Tokenize the sentences
tokens = tokenizer([sentence1, sentence2], padding=True, truncation=True, return_tensors="pt")

# Get the BERT embeddings
with torch.no_grad():
    outputs = model(**tokens)

# Get the embeddings of each sentence
embeddings = outputs.last_hidden_state

# Calculate the cosine similarity between the embeddings
similarity_score = torch.cosine_similarity(embeddings[0].mean(dim=0), embeddings[1].mean(dim=0), dim=0)

# Print the cosine similarity score
print("Cosine Similarity:", similarity_score.item())
