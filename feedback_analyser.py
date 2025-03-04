import numpy as np
import gensim.downloader as api
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split

# Load GloVe embeddings
word_vectors = api.load("glove-wiki-gigaword-100")  # 100D GloVe

def get_embedding_vector(word):
    """Return word embedding if available, otherwise return a zero vector."""
    try:
        return word_vectors[word]
    except KeyError:
        return np.zeros(100)  # Return zero vector for unknown words

def review_to_vector(review, embedding_dim=100):
    """Convert a review to an average word embedding vector."""
    words = review.split()
    word_embeddings = [get_embedding_vector(w) for w in words]
    if word_embeddings:
        return np.mean(word_embeddings, axis=0)
    else:
        return np.zeros(embedding_dim)

# Define input dimensions
embedding_dim = 100  # GloVe embedding dimension

# Larger dataset
reviews = [
    "Great quality and fast shipping. Highly recommend!",  # Positive
    "I love this! It's exactly what I was looking for.",  # Positive
    "Excellent customer service. Very satisfied!",  # Positive
    "This is amazing, I will buy again!",  # Positive
    "Not bad, but could be better. It’s average.",  # Neutral
    "The product is okay, does what it says, nothing special.",  # Neutral
    "It’s alright, nothing too impressive.",  # Neutral
    "Product was fine but delivery was slow.",  # Neutral
    "This is the worst product I have ever bought.",  # Negative
    "Very disappointed. Not as described at all.",  # Negative
    "Terrible experience. I want a refund.",  # Negative
    "This is garbage, completely useless.",  # Negative
]

labels = [2, 2, 2, 2, 1, 1, 1, 1, 0, 0, 0, 0]  # 2=Positive, 1=Neutral, 0=Negative

# Convert reviews to vectors
X = np.array([review_to_vector(r) for r in reviews])
y = np.array(labels)

# Split into train and test sets (now better balanced)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Simpler MLP Model to avoid overfitting
mlp_model = Sequential([
    Dense(64, activation='relu', input_shape=(embedding_dim,)),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(3, activation='softmax')  # 3 classes: Positive, Neutral, Negative
])

mlp_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
mlp_model.fit(X_train, y_train, epochs=20, batch_size=2, validation_data=(X_test, y_test))

# Evaluate the model
loss, accuracy = mlp_model.evaluate(X_test, y_test)
print(f'Test Accuracy: {accuracy:.2f}')

# Mapping labels to sentiment names
sentiment_labels = {0: "Negative", 1: "Neutral", 2: "Positive"}

# Predict sentiment for a new review
new_review = "Great quality and fast shipping. Highly recommend!"  
review_vector = review_to_vector(new_review)

# Ensure correct input shape for prediction
review_vector = np.expand_dims(review_vector, axis=0)

predicted_class = np.argmax(mlp_model.predict(review_vector))

# Print the result
print("Predicted Sentiment:", sentiment_labels[predicted_class])
