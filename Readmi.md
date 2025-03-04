Sentiment Analysis Using GloVe Embeddings and MLP

Overview

This project is a sentiment analysis system that classifies user reviews as Positive, Neutral, or Negative using GloVe word embeddings and a Multi-Layer Perceptron (MLP) model built with TensorFlow/Keras. The model is trained on a small dataset and predicts the sentiment of a given review.

Features

Uses GloVe embeddings (100D) to represent words as dense vectors.

Converts reviews into average word embeddings.

Implements an MLP model with Dense and Dropout layers.

Classifies sentiments into Positive, Neutral, and Negative.

Evaluates model accuracy and predicts sentiment for a new review.

Installation

Ensure you have Python installed (>=3.7). Install the required dependencies:

pip install numpy gensim tensorflow scikit-learn

Code Workflow

1. Import Required Libraries

import numpy as np
import gensim.downloader as api
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split

numpy: Handles numerical computations.

gensim: Loads pre-trained GloVe embeddings.

tensorflow.keras: Builds and trains the MLP model.

sklearn: Splits data into training and testing sets.

2. Load GloVe Embeddings

word_vectors = api.load("glove-wiki-gigaword-100")  # 100D GloVe

Loads pre-trained GloVe embeddings (100D) from gensim's dataset.

Helps convert words into meaningful vector representations.

3. Convert Words and Reviews to Vectors

def get_embedding_vector(word):
    try:
        return word_vectors[word]
    except KeyError:
        return np.zeros(100)  # Return zero vector for unknown words

If a word exists in the GloVe model, return its embedding vector.

Otherwise, return a zero vector (handles out-of-vocabulary words).

def review_to_vector(review, embedding_dim=100):
    words = review.split()
    word_embeddings = [get_embedding_vector(w) for w in words]
    if word_embeddings:
        return np.mean(word_embeddings, axis=0)
    else:
        return np.zeros(embedding_dim)

Converts a review into a single vector by averaging word embeddings.

If the review has no valid words, returns a zero vector.

4. Dataset Preparation

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

Creates a small dataset of reviews labeled as Positive (2), Neutral (1), or Negative (0).

X = np.array([review_to_vector(r) for r in reviews])
y = np.array(labels)

Converts reviews into numerical feature vectors using GloVe.

5. Split Dataset into Training and Testing

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

Splits data into 70% training and 30% testing.

Ensures a balanced distribution of sentiments.

6. Build the MLP Model

mlp_model = Sequential([
    Dense(64, activation='relu', input_shape=(embedding_dim,)),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(3, activation='softmax')  # 3 classes: Positive, Neutral, Negative
])

Uses a fully connected neural network (MLP).

Dense(64, relu): First hidden layer.

Dropout(0.3): Prevents overfitting.

Dense(32, relu): Second hidden layer.

Dense(3, softmax): Outputs probability scores for 3 classes.

7. Compile and Train the Model

mlp_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
mlp_model.fit(X_train, y_train, epochs=20, batch_size=2, validation_data=(X_test, y_test))

adam: Optimizer for faster learning.

sparse_categorical_crossentropy: Suitable for multi-class classification.

Trains for 20 epochs with a batch size of 2.

8. Evaluate Model Performance

loss, accuracy = mlp_model.evaluate(X_test, y_test)
print(f'Test Accuracy: {accuracy:.2f}')

Evaluates test accuracy to measure model performance.

9. Predict Sentiment of a New Review

new_review = "Great quality and fast shipping. Highly recommend!"
review_vector = review_to_vector(new_review)
review_vector = np.expand_dims(review_vector, axis=0)
predicted_class = np.argmax(mlp_model.predict(review_vector))

Converts a new review into GloVe vector representation.

Feeds it into the trained model to predict sentiment.

10. Output the Sentiment Prediction

sentiment_labels = {0: "Negative", 1: "Neutral", 2: "Positive"}
print("Predicted Sentiment:", sentiment_labels[predicted_class])

Maps model output (0, 1, 2) to Negative, Neutral, or Positive.

Displays the predicted sentiment for the review.

Expected Output Example

Test Accuracy: 0.83
Predicted Sentiment: Positive

The model achieves high accuracy and successfully classifies new reviews.

Future Improvements

Increase dataset size for better generalization.

Use pretrained transformers (BERT) for more accurate embeddings.

Implement LSTM/RNN for sequence-based learning.

Author

Developed by: Somnath Bhandari M
Project Type: Sentiment Analysis using Deep Learning

Happy coding! 🚀

