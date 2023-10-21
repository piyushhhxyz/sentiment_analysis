import pickle
from nltk.classify.scikitlearn import SklearnClassifier

# Load the trained classifier from the saved file
with open("sentiment_classifier.pkl", "rb") as file:
    classifier = pickle.load(file)

# Define a function to predict sentiment
def predict_sentiment(text):
    words = text.split()  # Tokenize the input text
    features = find_features(words)  # Extract features from the input text
    sentiment = classifier.classify(features)  # Use the trained classifier to predict sentiment
    return sentiment

# Example usage:
example_text = "I absolutely loved this movie, it was fantastic!"
sentiment = predict_sentiment(example_text)
print(f"Sentiment: {sentiment}")

example_text = "This movie was terrible, I hated it."
sentiment = predict_sentiment(example_text)
print(f"Sentiment: {sentiment}")
