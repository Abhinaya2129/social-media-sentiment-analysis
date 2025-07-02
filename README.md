# 📊 Social Media Sentiment Analysis (Django)

This project is a web-based application built using **Django** that analyzes the sentiment of user-input social media text. It classifies the sentiment as **Positive**, **Negative**, or **Neutral** using a **machine learning model (Random Forest)** trained on real-world tweet data.

---

## 🌐 Features

- 📝 User input form to analyze any text/tweet
- 📈 Train the sentiment analysis model from the admin panel
- 📊 Displays model accuracy and classification report
- 💾 Saves trained model and vectorizer using `joblib`
- 🔐 Session-based user authentication
- 💬 Real-time sentiment prediction with user-friendly UI

---

## 🛠️ Tech Stack

- **Backend:** Django (Python)
- **Frontend:** HTML, CSS (Bootstrap), JavaScript
- **ML Model:** Random Forest Classifier (with TF-IDF)
- **Libraries:** `scikit-learn`, `pandas`, `joblib`
- **Deployment:** [Render](https://render.com)

---

## 🚀 Getting Started

### 🔗 Live Project

👉 [Click here to open the live app](https://social-media-sentiment-analysis.onrender.com)

> _If the site takes a few seconds to load, it’s because free Render services spin down after inactivity._

### 1. Clone the Repo

```bash
git clone https://github.com/Abhinaya2129/social-media-sentiment-analysis.git
cd social-media-sentiment-analysis
