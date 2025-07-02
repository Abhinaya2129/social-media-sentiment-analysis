import os
import pandas as pd
import joblib
from django.shortcuts import render,redirect
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

def base(request):
    return render(request,'base.html')

def user_login(request):
    if request.method == "POST":
        username = request.POST["username"]
        password = request.POST["password"]

        # Custom admin/admin check
        if username == "admin" and password == "admin":
            request.session['user'] = username  # Manual session login
            return redirect('home')
        else:
            return render(request, "login.html", {"error": "Invalid credentials"})

    return render(request, "login.html")

def home(request):
    if 'user' not in request.session:
        return redirect('login')
    return render(request, "home.html")



BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'sentiment_model.pkl')
VECTORIZER_PATH = os.path.join(BASE_DIR, 'vectorizer.pkl')


def train_model_view(request):
    if 'user' not in request.session:
        return redirect('login')

    success = False
    error = None
    accuracy = None
    report = None

    if request.method == "POST":
        try:
            print("🔹 Starting model training process...")

            # Load dataset
            print("📂 Loading dataset...")
            df = pd.read_csv(os.path.join(BASE_DIR, r'socialmedia_comments.csv\\socialmedia_comments.csv'), header=None,
                             names=['ID', 'Entity', 'Sentiment', 'Tweet'])
            print("✅ Dataset loaded. Total records:", len(df))

            # Preprocessing
            print("🧹 Cleaning data...")
            df.dropna(subset=['Tweet', 'Sentiment'], inplace=True)
            X = df['Tweet']
            y = df['Sentiment']
            print("✅ After cleaning:", len(df), "records")

            # Split data
            print("🔀 Splitting into train and test sets...")
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Vectorize
            print("🧠 Vectorizing tweets using TF-IDF...")
            vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
            X_train_vec = vectorizer.fit_transform(X_train)
            X_test_vec = vectorizer.transform(X_test)
            print("✅ Vectorization complete. Shape:", X_train_vec.shape)

            # Train model
            print("🏋️ Training RandomForest model...")
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train_vec, y_train)
            print("✅ Model training complete.")

            # Evaluate
            print("📈 Evaluating model...")
            y_pred = model.predict(X_test_vec)
            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred)
            print("✅ Accuracy:", accuracy)
            print("✅ Classification Report:\n", report)

            # Save
            print("💾 Saving model and vectorizer...")
            joblib.dump(model, MODEL_PATH)
            joblib.dump(vectorizer, VECTORIZER_PATH)
            print("✅ Model and vectorizer saved successfully.")

            success = True

        except Exception as e:
            error = str(e)
            print("❌ Error during training:", error)

    return render(request, "train.html", {
        "success": success,
        "error": error,
        "accuracy": accuracy,
        "report": report
    })


# Load model and vectorizer (run once at server start)
if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
else:
    model = None
    vectorizer = None
    
def predict_sentiment(request):
    sentiment = ""
    error = ""

    if request.method == "POST":
        tweet = request.POST.get("tweet", "").strip()

        if tweet == "":
            error = "⚠️ Please enter a tweet to analyze."
        elif model is not None and vectorizer is not None:
            input_vec = vectorizer.transform([tweet])
            sentiment = model.predict(input_vec)[0]
        else:
            error = "⚠️ Model not loaded. Please train the model first."

    return render(request, "predict.html", {
        "sentiment": sentiment,
        "error": error
    })
