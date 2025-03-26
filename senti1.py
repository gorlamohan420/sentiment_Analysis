# Sentiment Analysis Project - Restaurant Reviews

# Import necessary libraries
import numpy as np
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Download NLTK stopwords (if not already downloaded)
nltk.download('stopwords')

# 1. Data Loading and Preprocessing
def load_and_preprocess_data(file_path):
    """Load and preprocess the dataset safely"""
    try:
        # Load dataset with tab separator
        data = pd.read_csv(file_path, delimiter='\t', quoting=3, encoding="utf-8")

        # Convert 'Liked' column to numeric, replacing invalid values with NaN
        data['Liked'] = pd.to_numeric(data['Liked'], errors='coerce')

        # Drop rows with missing values
        data.dropna(inplace=True)

        # Convert 'Liked' back to integer
        data['Liked'] = data['Liked'].astype(int)

        # Display full dataset
        print(f"\nTotal Reviews Loaded: {len(data)}")
        print("\nFull Dataset:")
        print(data.to_string())  # Display the entire dataset

        print("\nClass distribution:")
        print(data['Liked'].value_counts())

        return data

    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

# 2. Text Cleaning Function
def clean_text(text):
    """Clean and preprocess text data"""
    ps = PorterStemmer()
    text = re.sub('[^a-zA-Z]', ' ', text)  # Remove special characters
    text = text.lower().split()  # Convert to lowercase and split
    text = [ps.stem(word) for word in text if word not in set(stopwords.words('english'))]  # Remove stopwords
    return ' '.join(text)

# 3. Feature Engineering
def create_features(data):
    """Convert text data into numerical feature vectors"""
    corpus = data['Review'].apply(clean_text)
    cv = CountVectorizer(max_features=1500)
    X = cv.fit_transform(corpus).toarray()
    y = data['Liked'].values  # Target variable
    return X, y, cv

# 4. Model Training
def train_model(X_train, y_train, alpha=0.2):
    """Train and return the classifier"""
    classifier = MultinomialNB(alpha=alpha)
    classifier.fit(X_train, y_train)
    return classifier

# 5. Model Evaluation
def evaluate_model(classifier, X_test, y_test):
    """Evaluate model performance"""
    y_pred = classifier.predict(X_test)

    print("\nEvaluation Metrics:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2%}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Confusion Matrix Visualization
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

# 6. Prediction Function
def predict_sentiment(classifier, cv, review_text):
    """Predict sentiment of a new review"""
    cleaned_review = clean_text(review_text)
    vectorized = cv.transform([cleaned_review]).toarray()
    prediction = classifier.predict(vectorized)[0]
    return "Positive" if prediction == 1 else "Negative"

# Main Execution
if __name__ == "__main__":
    # File path - Update this with your actual file path
    FILE_PATH = r'C:\Users\Sumithra\Desktop\code\liked.tsv'

    # Step 1: Load and preprocess data
    data = load_and_preprocess_data(FILE_PATH)

    if data is not None:
        # Step 2: Create features
        X, y, cv = create_features(data)

        # Step 3: Split data (40% for testing)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.4, random_state=0
        )

        # Step 4: Train model
        classifier = train_model(X_train, y_train)

        # Step 5: Evaluate model
        evaluate_model(classifier, X_test, y_test)

        # Step 6: Interactive predictions
        print("\nTest the model with custom reviews:")
        test_reviews = [
            "The food was absolutely wonderful!",
            "Service was terrible and food cold.",
            "Average experience, nothing special.",
            "Best steak I've ever had!",
            "Would not recommend to anyone."
        ]

        for review in test_reviews:
            sentiment = predict_sentiment(classifier, cv, review)
            print(f"\nReview: {review}")
            print(f"Predicted Sentiment: {sentiment}")