from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np

app = FastAPI()

# Load the trained model
model = joblib.load('spam_classifier_model.pkl')

# Feature extraction function
def extract_features(email_text):
    email_length = len(email_text)
    subject_length = len(email_text.split('\n')[0])  # Assuming the subject is the first line
    word_count = len(email_text.split())
    text_without_stopwords = email_text  # Assuming preprocessed text without stopwords is available
    email_length_log = np.log1p(email_length)
    return {
        'email_length': email_length,
        'subject_length': subject_length,
        'text_without_stopwords': text_without_stopwords,
        'word_count': word_count,
        'email_length_log': email_length_log
    }

@app.get("/", response_class=HTMLResponse)
async def main():
    return """
    <html>
        <head>
            <title>Spam Filter</title>
        </head>
        <body>
            <h1>Spam Filter Prediction</h1>
            <form action="/predict" method="post">
                <label>Email Text:</label><br>
                <textarea name="email_text" rows="10" cols="50"></textarea><br><br>
                <input type="submit" value="Predict">
            </form>
        </body>
    </html>
    """

@app.post("/predict", response_class=HTMLResponse)
async def predict(email_text: str = Form(...)):
    # Extract features from the input email text
    features = extract_features(email_text)
    
    # Convert to DataFrame
    input_data = pd.DataFrame([features])

    # Make predictions
    prediction = model.predict(input_data)

    result = "Spam" if prediction == 1 else "Ham"

    return f"""
    <html>
        <head>
            <title>Prediction Result</title>
        </head>
        <body>
            <h1>Prediction Result</h1>
            <p>The email is classified as: <strong>{result}</strong></p>
            <a href="/">Back to Home</a>
        </body>
    </html>
    """

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
