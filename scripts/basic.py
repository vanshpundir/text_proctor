
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity
import yaml
from yaml.loader import SafeLoader


class SentenceSimilarity:
    def __init__(self):
        with open(config_path) as f:
            data = yaml.load(f, Loader=SafeLoader)
        self.tokenizer = AutoTokenizer.from_pretrained(data['text_proctor']['model_name'])
        self.model = AutoModel.from_pretrained(data['text_proctor']['model_name'])
        self.device = data['text_proctor']['device']



    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def normalising_embeddings(self, embeddings):
        return F.normalize(embeddings, p=2, dim=1)

    def encode_sentences(self, sentences):
        encoded_input = self.tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
        encoded_input = encoded_input.to(torch.device(self.device))
        return encoded_input

    def compute_embeddings(self, encoded_input):
        with torch.no_grad():
            model_output = self.model(**encoded_input)
        sentence_embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])
        sentence_embeddings = self.normalising_embeddings(sentence_embeddings)
        return sentence_embeddings

    def compute_similarity(self, sentences, source_sentence):
        encoded_sentences = self.encode_sentences(sentences)
        encoded_source = self.encode_sentences([source_sentence])
        sentence_embeddings = self.compute_embeddings(encoded_sentences)
        source_embedding = self.compute_embeddings(encoded_source)

        source_embedding = source_embedding.cpu().numpy()
        sentence_embeddings = sentence_embeddings.cpu().numpy()

        similarity_scores = cosine_similarity(source_embedding, sentence_embeddings)

        return similarity_scores


# Usage
if __name__ == "__main__":
    config_path = 'config/ai_config.yaml'
    sentences = ["I am vansh from chitkara university and not from chandigarh",'hijklhlkjh']
    source_sentence = "I am from chandigarh"

    similarity_calculator = SentenceSimilarity()
    similarity_scores = similarity_calculator.compute_similarity(sentences, source_sentence)

    print("Similarity scores:")
    print(similarity_scores)