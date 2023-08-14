import pandas as pd
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

# Read the CSV file
df = pd.read_csv("training_data.csv")

# Convert 'Similarity' column to float explicitly
df['Similarity'] = df['Similarity'].astype(float)

# Create InputExample objects from the data
train_examples = [InputExample(texts=[text1, text2], label=similarity) for text1, text2, similarity in zip(df['Text1'], df['Text2'], df['Similarity'])]

# Define the model. Either from scratch or by loading a pre-trained model
model = SentenceTransformer('distilbert-base-nli-mean-tokens')

# Define your train dataset, the dataloader, and the train loss
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
train_loss = losses.CosineSimilarityLoss(model)


# Tune the model
model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=10, warmup_steps=100)

# Save the fine-tuned model
model.save("./third_party/fine_tuned_model")
