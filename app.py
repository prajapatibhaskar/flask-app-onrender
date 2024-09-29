from flask import Flask, request, jsonify
import pickle
from flask_cors import CORS  # Import CORS
from tensorflow.keras.models import load_model

app = Flask(__name__)
CORS(app)

# Load the model and tokenizer
with open('resources/tfidf_tokenizer.pkl', 'rb') as f:
    tfidf_tokenizer = pickle.load(f)
loaded_model = load_model('resources/text_classification_model.h5')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    text = data.get('text')
    
    if not text or not text.strip():
        return jsonify({'error': 'No text provided'}), 400
    
    text_features = tfidf_tokenizer.transform([text])
    predictions = loaded_model.predict(text_features)
    predicted_label = int(predictions[0][0])
    
    return jsonify({'predicted_label': predicted_label})
    
if __name__ == '__main__':
    app.run(debug=True)
