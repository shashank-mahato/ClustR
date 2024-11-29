# backend/utils.py
import re
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from transformers import AutoTokenizer, AutoModel
import torch

# Load pre-trained BERT model and tokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModel.from_pretrained('bert-base-uncased')

def load_data():
    """Load the 20 Newsgroups dataset."""
    newsgroups_train = fetch_20newsgroups(subset='train')
    return newsgroups_train.data, newsgroups_train.target

def clean_text(text):
    """Clean text by removing special characters, converting to lowercase, and normalizing spaces."""
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove special characters
    text = text.lower()                      # Convert to lowercase
    text = re.sub(r'\s+', ' ', text)         # Replace multiple spaces with a single space
    return text

def generate_embedding(text):
    """Generate BERT embeddings for a given text."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    embedding = outputs.last_hidden_state.mean(dim=1)  # Mean pooling to get a fixed-size embedding
    return embedding

def generate_embeddings_and_save():
    """Load data, clean it, generate embeddings, and save embeddings."""
    # Load dataset
    newsgroups_data, _ = load_data()

    # Clean text data
    cleaned_data = [clean_text(doc) for doc in newsgroups_data]

    # Generate embeddings
    embeddings = [generate_embedding(doc) for doc in cleaned_data]

    # Convert list of embeddings to numpy array and save
    embeddings_np = np.vstack([embedding.numpy() for embedding in embeddings])

    # Save the embeddings to a file
    save_dir = "backend/data/"
    np.save(save_dir + "embeddings_train.npy", embeddings_np)
    print(f"Embeddings saved to {save_dir}embeddings_train.npy")

# Run the function to generate and save embeddings
generate_embeddings_and_save()
