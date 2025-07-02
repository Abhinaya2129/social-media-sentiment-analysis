# Social Media Sentiment Analysis 💬

This Django-based web application allows users to classify the sentiment of social media comments or tweets as **Positive**, **Negative**, or **Neutral** using a trained Random Forest model.

## 🔧 Features

- Upload and train model on a CSV of social media posts
- Predict sentiment from user-inputted text
- View accuracy and classification report
- Clean web UI built using Django

## 🧠 Tech Stack

- Python
- Django
- HTML + Bootstrap
- Machine Learning: `scikit-learn`, `TfidfVectorizer`, `RandomForestClassifier`
- Model persistence: `joblib`

## 🚫 Note
The trained model file (`sentiment_model.pkl`) is excluded from this repository to avoid GitHub's 100MB file size limit. You can retrain it using the "Train Model" feature on the website or upload it separately.

## 🚀 Deployment
Best deployed using [Render](https://render.com) or [Railway](https://railway.app) for Django apps.

### 📡 Live Demo

👉 **[Try the App on Render](https://social-media-sentiment-analysis-uns9.onrender.com)**

---

## 📂 Setup Instructions

1. Clone the repository:
   ```bash
   git clone https://github.com/Abhinaya2129/social-media-sentiment-analysis.git
   cd social-media-sentiment-analysis
