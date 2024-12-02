from flask import Flask, request, jsonify
from transformers import pipeline

app = Flask(__name__)

# Load pre-trained model for zero-shot classification
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Route for home page (optional)
@app.route('/')
def home():
    return "Welcome to the chatbot API!"

# Route for prediction (used by the frontend)
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()  # Get JSON data from the request
    user_message = data.get('message')  # Extract the message

    if not user_message:
        return jsonify({'error': 'No message provided'}), 400

    # Define possible candidate labels for classification
    candidate_labels = ['transfer', 'stake', 'check balance']
    result = classifier(user_message, candidate_labels)

    # Return the predicted intent and confidence
    return jsonify({'intent': result['labels'][0], 'confidence': result['scores'][0]})

if __name__ == "__main__":
    app.run(debug=True)
