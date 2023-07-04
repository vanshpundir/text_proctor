from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity
import yaml
from yaml.loader import SafeLoader

# Open the file and load the file
with open('config/ai_config.yaml') as f:
    data = yaml.load(f, Loader=SafeLoader)
    device = data['text_proctor']['device']
# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


# Sentences we want sentence
sentences = ["I am vansh from chitkara university and not from chandigarh"]

tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')


# Load model from HuggingFace Hub
model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
# tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')

#
# model = AutoModel.from_pretrained('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
model.to(torch.device(device))

# Tokenize sentences
encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
encoded_input = encoded_input.to(torch.device(device))

# Compute token embeddings for sentences
with torch.no_grad():
    model_output = model(**encoded_input)

# Perform pooling for sentences
sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

# Normalize sentence embeddings
sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

# Source sentence
source_sentence = """I am from chandigarh"""

# Tokenize the source sentence
encoded_source = tokenizer(source_sentence, padding=True, truncation=True, return_tensors='pt')
encoded_source = encoded_source.to(torch.device(device))

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

#https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2