[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1xLqnea0mIzmqt4fuAR3Dyy60uLEBpwmv)

# 🔥 Media Emotion Intelligence  
### Emotion Analysis of Geopolitical News using Deep Learning

🚀 **Live Demo:**  
https://rajeshreddi-media-analysis.hf.space

---

## 📌 Project Overview

This project analyzes emotional framing in real-world geopolitical news articles using Natural Language Processing (NLP) and deep learning.

It performs:

- Article-level emotion classification  
- Sentence-level emotion prediction  
- Aggregated emotion distribution visualization  
- Emotional trend progression across sentences  

The system is deployed as a cloud-based web application using Docker and HuggingFace Spaces.

---

## 🧠 Problem Statement

News media often frames geopolitical events with varying emotional tones.  
This project aims to:

- Quantify emotional polarity in news reporting  
- Analyze shifts in tone across conflict phases  
- Provide interpretable visual emotion analytics  

---

## ⚙️ Tech Stack

**Backend & ML**
- Python
- TensorFlow / Keras (BiLSTM Model)
- Scikit-learn
- NumPy
- Pandas

**Frontend**
- Flask
- Chart.js (Interactive Graphs)
- HTML / CSS (Custom Premium UI)

**Deployment**
- Docker
- HuggingFace Spaces (Cloud Hosting)

---

## 🏗 System Architecture

1. User inputs geopolitical news article  
2. Text is split into sentences  
3. Tokenization & padding applied  
4. BiLSTM model predicts emotion probabilities  
5. Sentence-level probabilities are aggregated  
6. Visualizations generated:
   - Article-level bar chart  
   - Sentence-level breakdown table  
   - Emotional trend line graph  

---

## 📊 Model Details

- Architecture: Bidirectional LSTM (BiLSTM)
- Embedding Layer → BiLSTM → Dropout → Dense (Softmax)
- Trained on GoEmotions dataset
- Selected Emotions:
  - Anger
  - Fear
  - Joy
  - Neutral
  - Sadness



## 📂 Repository Structure
media-emotion-analysis/
│
├── app.py
├── Dockerfile
├── requirements.txt
├── best_emotion_bilstm_model.h5
├── tokenizer.pkl
├── label_encoder.pkl
│
├── templates/
├── static/
│
└── Media_Emotion_Analysis_Geopolitical_News.ipynb
## 📈 Features

✔ Real-time emotion prediction  
✔ Sentence-level emotional breakdown  
✔ Aggregated emotion visualization  
✔ Emotional progression graph  
✔ Cloud-deployed interactive dashboard  

---

## 🧪 Dataset

- GoEmotions dataset used for training
- Custom curated geopolitical news dataset for analysis
- Processed and trained using Google Colab

---

## 🚀 Deployment

The application is containerized using Docker and deployed on HuggingFace Spaces for cloud-based inference.

Live App:
https://rajeshreddi-media-analysis.hf.space

---

## 🎯 Future Improvements

- Multi-label emotion classification  
- Transformer-based models (BERT / RoBERTa)  
- Comparative media bias analysis  
- API-based deployment  

---

## 👨‍💻 Author

**Rajesh Reddy**  
📧 rajesh.reddi06@gmail.com  
🌐 https://github.com/RajeshReddi
