import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load dataset
df = pd.read_csv("week4/d5_sentiment_classifier/data/sentiment.csv")

# Check data
print("Original shape:", df.shape)
print(df.isnull().sum())
print("Duplicates:", df.duplicated().sum())

# Remove missing values
df = df.dropna(subset=["text", "sentiment"])

# Convert text to lowercase
df["text"] = df["text"].str.lower()

# Remove extra spaces
df["text"] = df["text"].str.strip()
df["text"] = df["text"].str.replace(r"\s+", " ", regex=True)

# Remove punctuation and special characters
df["text"] = df["text"].str.replace(
    r"[^a-zA-Z\s]", "", regex=True
)

# Save cleaned dataset
df.to_csv("week4/d5_sentiment_classifier/data/sentiment_cleaned.csv", index=False)

print("Cleaned shape:", df.shape)
print(df.head())
print(df["sentiment"].value_counts())


### dependent and independent features
x = df['text']
y = df['sentiment']

### train test split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.10, random_state=42)

### tf-id vectorizer
tf_id = TfidfVectorizer()

x_train_vector = tf_id.fit_transform(x_train)

x_test_vector = tf_id.transform(x_test)

print("Training TF-IDF shape:", x_train_vector.shape)
print("Testing TF-IDF shape:", x_test_vector.shape)


### model creation
gradient = GradientBoostingClassifier(
    n_estimators=50,
    max_depth=5,
    random_state=42
)


### model training
gradient.fit(x_train_vector, y_train)

### model prediction
y_pred = gradient.predict(x_test_vector)


### model evaluation
score = accuracy_score(y_test, y_pred)

cr = classification_report(y_test, y_pred)

cm = confusion_matrix(y_test, y_pred)

print(10 * "*")
print('\nFinal report:')
print(10 * "*")

print(f'accuracy score: {score}\n')
print(f'classification report: {cr}\n')
print(f'confusion matrix: {cm}\n')