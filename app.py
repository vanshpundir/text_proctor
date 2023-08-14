from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer, util
import yaml

# Load the YAML configuration
with open('config/ai_config.yaml', 'r') as file:
    config = yaml.safe_load(file)['text_proctor']

# Initialize the Flask app
app = Flask(__name__)

# Load the Sentence Transformer model
model = SentenceTransformer(config['model_name'], device=config['device'])

@app.route('/get_similarity', methods=['POST'])
def get_similarity():
    data = request.get_json()

    text1 = data.get('text1')
    text2 = data.get('text2')

    # Compute cosine similarity
    embeddings = model.encode([text1, text2], convert_to_tensor=True)
    similarity = util.pytorch_cos_sim(embeddings[0], embeddings[1]).item()

    return jsonify({'similarity': similarity})

if __name__ == '__main__':
    app.run(debug=True)
