import pickle
import nltk
import random
from nltk.corpus import movie_reviews
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

nltk.download('movie_reviews')

# Create a list of documents (reviews) and their corresponding labels
documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]


random.shuffle(documents) # Shuffle the documents for randomness


all_words = [w.lower() for w in movie_reviews.words()] # Extract words and convert them to lowercase

# Calculate word frequencies and select the top 2,000 most common words
all_words = nltk.FreqDist(all_words)
word_features = list(all_words.keys())[:2000]

# Define a function to extract features from a document
def find_features(document):
    words = set(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)
    return features

# Create feature sets
featuresets = [(find_features(rev), category) for (rev, category) in documents]

# Split the data into training and testing sets
train_set, test_set = train_test_split(featuresets, test_size=0.2, random_state=42)

# Train a logistic regression classifier
classifier = SklearnClassifier(LogisticRegression(max_iter=1000))  # Increase max_iter as needed
classifier.train(train_set)

# Make predictions
predictions = [classifier.classify(test[0]) for test in test_set]

# Calculate accuracy
accuracy = accuracy_score([test[1] for test in test_set], predictions)
print(f'Accuracy: {accuracy:.2f}')

# Save the trained classifier to a file
with open("sentiment_classifier.pkl", "wb") as file:
    pickle.dump(classifier, file)