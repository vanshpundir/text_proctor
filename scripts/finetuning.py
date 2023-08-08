import pandas as pd
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

# Read the CSV file
df = pd.read_csv("training_data.csv")
df.fillna(0,inplace = True)
# Extract the 'Text1', 'Text2', and 'Similarity' columns from the dataframe
# Extract the 'Text1', 'Text2', and 'Similarity' columns from the dataframe
texts1 = df['Text1'].astype(str).tolist()
texts2 = df['Text2'].astype(str).tolist()
similarities = df['Similarity'].tolist()  # Convert the similarities to float
similarities.remove("Similarity")
similarities = df['Similarity'].astype(float).tolist()
# Create InputExample objects from the data
train_examples = [InputExample(texts=[text1, text2], label=float(similarity)) for text1, text2, similarity in zip(texts1, texts2, similarities)]


# Define the model. Either from scratch or by loading a pre-trained model
model = SentenceTransformer('distilbert-base-nli-mean-tokens')

# Define your train dataset, the dataloader, and the train loss
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
train_loss = losses.CosineSimilarityLoss(model)

# Tune the model
model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=10, warmup_steps=100)

model.save("./third_party/fine_tuned_model")
