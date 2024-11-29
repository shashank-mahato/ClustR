# active_learning/feedback_handler.py
import json
import os

def save_feedback(feedback_data, feedback_file="backend/data/feedback.json"):
    """ Save human feedback for later retraining """
    with open(feedback_file, "a") as f:
        f.write(f"{json.dumps(feedback_data)}\n")
    print("Feedback saved successfully!")

def detect_low_confidence(classifier, document_embedding, threshold=0.7):
    """ Check if the classifier's confidence is below the threshold """
    prediction_prob = classifier.predict_proba(document_embedding)[0]
    max_confidence = max(prediction_prob)
    if max_confidence < threshold:
        return True, prediction_prob
    return False, prediction_prob
