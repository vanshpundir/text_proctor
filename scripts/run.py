from sentence_transformers import SentenceTransformer, InputExample, losses
import torch

# Load the fine-tuned model
model = SentenceTransformer('./third_party/model/fine_tuned_model')

# Example sentences
example1 = InputExample(texts=['I am from India.', 'I am from India.'])
example2 = InputExample(texts=['I love this movie.', 'This movie is terrible.'])

# Get the embeddings of the sentences using the fine-tuned model
embeddings1 = model.encode(example1.texts)
embeddings2 = model.encode(example2.texts)

# Convert NumPy arrays to PyTorch tensors
embeddings1 = torch.tensor(embeddings1)
embeddings2 = torch.tensor(embeddings2)

# Calculate the cosine similarity between the embeddings
cosine_similarity_score = torch.cosine_similarity(embeddings1[0], embeddings1[1], dim=0)
print("Cosine Similarity Example 1:", cosine_similarity_score.item())

cosine_similarity_score = torch.cosine_similarity(embeddings2[0], embeddings2[1], dim=0)
print("Cosine Similarity Example 2:", cosine_similarity_score.item())


# Evaluation
# we require contrastive loss