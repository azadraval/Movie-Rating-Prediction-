from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix
import numpy as np
from tensorflow.keras.models import load_model # type: ignore

app = Flask(__name__)

# Load the pre-trained model
model = load_model('your_model_file.h5')

# Load the TF-IDF vectorizer
vectorizer = TfidfVectorizer()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        review_text = request.form['review']
        
        # Preprocess input data
        preprocessed_text = vectorizer.transform([review_text])
        
        # Convert sparse matrix to dense NumPy array
        preprocessed_text = preprocessed_text.toarray()
        
        # Predict sentiment
        prediction = model.predict(preprocessed_text)
        
        # Convert prediction to human-readable format
        sentiment = "Positive" if prediction > 0.5 else "Negative"
        
        return render_template('result.html', sentiment=sentiment, review=review_text)

if __name__ == '__main__':
    app.run(debug=True)
