from flask import Flask, render_template, request
import pickle
import numpy as np
import re
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)

# Load model
model = load_model("best_emotion_bilstm_model.h5")

# Load tokenizer
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

MAX_SEQUENCE_LENGTH = 80

# Emotion labels (must match training order)
EMOTIONS = ["anger", "fear", "joy", "neutral", "sadness"]


def sentence_split(text):
    return re.split(r'(?<=[.!?])\s+', text.strip())


def predict_emotion(text):
    sentences = sentence_split(text)

    sequences = tokenizer.texts_to_sequences(sentences)

    padded = pad_sequences(
        sequences,
        maxlen=MAX_SEQUENCE_LENGTH,
        padding="post",
        truncating="post"
    )

    predictions = model.predict(padded)

    # Article-level aggregation
    avg_probs = np.mean(predictions, axis=0)

    article_result = {
        emotion: float(prob)
        for emotion, prob in zip(EMOTIONS, avg_probs)
    }

    dominant = EMOTIONS[np.argmax(avg_probs)]

    # Sentence-level breakdown
    sentence_results = []

    for sentence, pred in zip(sentences, predictions):
        sentence_results.append({
            "sentence": sentence,
            "anger": float(pred[0]),
            "fear": float(pred[1]),
            "joy": float(pred[2]),
            "neutral": float(pred[3]),
            "sadness": float(pred[4])
        })

    # Sentence-level aggregated distribution
    sentence_avg = {
        emotion: float(prob)
        for emotion, prob in zip(EMOTIONS, np.mean(predictions, axis=0))
    }

    return article_result, dominant, sentence_results, sentence_avg


@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    dominant = None
    sentence_results = None
    sentence_avg = None

    if request.method == "POST":
        text = request.form["news_text"]
        if text.strip():
            result, dominant, sentence_results, sentence_avg = predict_emotion(text)

    return render_template(
        "index.html",
        result=result,
        dominant=dominant,
        sentence_results=sentence_results,
        sentence_avg=sentence_avg
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
