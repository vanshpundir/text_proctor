from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch

def perform_sentiment_analysis(sentence1, sentence2):
    # Load the tokenizer and model
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")

    # Tokenize and encode the sentences
    tokens = tokenizer([sentence1, sentence2], padding=True, truncation=True, return_tensors="pt")

    # Perform sentiment analysis
    with torch.no_grad():
        outputs = model(**tokens)

    logits = outputs.logits
    predicted_labels = torch.argmax(logits, dim=1)

    return predicted_labels[0].item(), predicted_labels[1].item()

if __name__ == "__main__":
    sentence1 = "I am from India."
    sentence2 = "I am not from India."

    label1, label2 = perform_sentiment_analysis(sentence1, sentence2)

    if label1 == label2:
        print("The two sentences have the same sentiment.")
    else:
        print("The two sentences have different sentiments.")
