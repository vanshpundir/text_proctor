from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
import torch
# Define the model. Either from scratch or by loading a pre-trained model
model = SentenceTransformer('distilbert-base-nli-mean-tokens')

# Define your train examples with more diverse sentences and labels (cosine similarity scores)
train_examples = [
    InputExample(texts=['I am from India.', 'I am not from India.'], label=0.1),  # Example with negation
    InputExample(texts=['I love this movie.', 'This movie is terrible.'], label=0.05),  # Example with opposite sentiments
    InputExample(texts=['My first sentence', 'My second sentence'], label=0.8),
    InputExample(texts=['Another pair', 'Unrelated sentence'], label=0.03),
    # Add more examples with varying sentences and labels
]

# Define your train dataset, the dataloader, and the train loss
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
train_loss = losses.CosineSimilarityLoss(model)

# Tune the model
model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=10, warmup_steps=100)


#
example1 = InputExample(texts=['I am from India.', 'I am not from India.'])
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