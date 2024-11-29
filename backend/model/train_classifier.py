# backend/model/retrain_model.py
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import numpy as np
import os

def retrain_classifier(X_train, y_train, X_test, y_test, save_dir):
    """ Retrain a classifier with the embeddings and cluster labels """
    classifier = LogisticRegression(max_iter=1000, random_state=42)
    classifier.fit(X_train, y_train)  # Train the classifier on the embeddings and labels
    
    # Predict and evaluate accuracy
    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")

    # Save the trained classifier
    classifier_path = os.path.join(save_dir, "classifier.pkl")
    joblib.dump(classifier, classifier_path)
    print(f"Classifier saved to {classifier_path}")

    return classifier

def load_data_and_train():
    """ Load data and train the classifier """
    # Load the precomputed embeddings and cluster labels
    embeddings = np.load('backend/data/embeddings_train.npy')  # Load embeddings
    cluster_labels = np.load('backend/data/cluster_labels.npy')  # Load cluster labels
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(embeddings, cluster_labels, test_size=0.2, random_state=42)

    # Train the classifier and save it
    save_dir = 'backend/model/'
    retrain_classifier(X_train, y_train, X_test, y_test, save_dir)

# Run the training process
load_data_and_train()
