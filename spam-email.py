# Required Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, f1_score, classification_report
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re

# Download NLTK resources
# nltk.download('stopwords')
# nltk.download('wordnet')

# Load dataset
data = pd.read_csv('spam_emails.csv')  # Ensure this file is in the same directory as your script

# Text Pre-processing Function
def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\W', ' ', text)  # Remove non-word characters
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with single space

    # Tokenization and Removing stopwords
    words = text.split()
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]

    return ' '.join(words)
print(data.columns)

def predict_email_category(email_text):
    preprocessed_text = preprocess_text(email_text)
    vectorized_text = vectorizer.transform([preprocessed_text]).toarray()
    prediction = model.predict(vectorized_text)
    return "Spam" if prediction[0] == 1 else "Ham"

# Apply Pre-processing to Email Texts
data['text'] = data['text'].apply(preprocess_text)

# Feature Extraction using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)  # Adjust the number of features as needed
X = vectorizer.fit_transform(data['text']).toarray()
y = data['label']  # Labels: 1 for spam, 0 for not spam

# Splitting the Dataset into Training and Test Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training the Naive Bayes Classifier
model = MultinomialNB()
model.fit(X_train, y_train)

# Making Predictions and Evaluating the Model
predictions = model.predict(X_test)

# Evaluation Metrics
conf_matrix = confusion_matrix(y_test, predictions)
f_score = f1_score(y_test, predictions)
report = classification_report(y_test, predictions)

# Output the Results
print("Confusion Matrix:\n", conf_matrix)
print("\nF1 Score:", f_score)
print("\nClassification Report:\n", report)

# # Example usage:
# email_to_check = "Your example email text goes here."
# prediction_result = predict_email_category(email_to_check)
# print(f"The email is classified as: {prediction_result}")
