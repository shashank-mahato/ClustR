from flask import Flask, json, request, jsonify, render_template, redirect, url_for
import numpy as np
import joblib
from groq import Groq
import os
import torch
from transformers import BertTokenizer, BertModel
import logging
from werkzeug.utils import secure_filename
import PyPDF2
import docx
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pdfplumber
from pdf2image import convert_from_path
import pytesseract

app = Flask(__name__)

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

GROQ_API_KEY = "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" 
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'docx'}

UPLOADS_FOLDER = 'backend/uploads'
NEW_DATA_FOLDER = 'backend/new_data'
MOD_UPLOADS_FOLDER = 'backend/mod_uploads'

if not os.path.exists(UPLOADS_FOLDER):
    os.makedirs(UPLOADS_FOLDER)

if not os.path.exists(NEW_DATA_FOLDER):
    os.makedirs(NEW_DATA_FOLDER)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_pdf(file_path):
    """Extract text from a PDF file using pdfplumber or OCR."""
    try:
        with pdfplumber.open(file_path) as pdf:
            text = ""
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:  
                    text += page_text
                else:  
                    images = convert_from_path(file_path)
                    for image in images:
                        text += pytesseract.image_to_string(image)

        return text
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {e}")
        return ""

def extract_text_from_docx(file_path):
    """Extract text from a DOCX file."""
    try:
        doc = docx.Document(file_path)
        text = ""
        for para in doc.paragraphs:
            text += para.text + "\n"
        return text
    except Exception as e:
        logger.error(f"Error extracting text from DOCX: {e}")
        return ""

def clean_extracted_text(text):
    """Clean the extracted text."""
    text = ' '.join(text.split())
    text = text.replace("\n", " ").replace("\r", " ").replace("  ", " ").strip()
    return text

logger.debug("Loading classifier model...")
classifier = joblib.load("backend/model/classifier.pkl")
logger.debug("Classifier model loaded successfully.")

logger.debug("Loading cluster topics...")
cluster_topics = joblib.load("backend/model/cluster_topics.pkl")
logger.debug("Cluster topics loaded successfully.")

logger.debug("Loading BERT model and tokenizer...")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")
logger.debug("BERT model and tokenizer loaded successfully.")

def generate_embedding(text):
    """Generate BERT embeddings for a given text."""
    logger.debug(f"Generating embeddings for document: {text[:50]}...")  
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    embedding = outputs.last_hidden_state.mean(dim=1) 
    logger.debug("Embedding generated successfully.")
    return embedding

def save_to_new_data(document, embedding, label):
    """Save the processed document and embedding to new_data folder."""
    try:
        label_folder = os.path.join(NEW_DATA_FOLDER, label)
        if not os.path.exists(label_folder):
            os.makedirs(label_folder)

        text_filename = os.path.join(label_folder, f"{secure_filename(document[:20])}.txt")
        with open(text_filename, 'w') as f:
            f.write(document)

        embedding_filename = os.path.join(label_folder, f"{secure_filename(document[:20])}_embedding.npy")
        np.save(embedding_filename, embedding)

        logger.debug(f"Saved document and embedding to {label_folder}")
    except Exception as e:
        logger.error(f"Error saving processed document and embedding: {e}")

client = Groq(
    api_key=GROQ_API_KEY,
)

def get_cluster_topics_from_new_data():
    """Fetch cluster names from the new_data directory."""
    try:
        cluster_names = [
            name for name in os.listdir(NEW_DATA_FOLDER)
            if os.path.isdir(os.path.join(NEW_DATA_FOLDER, name))
        ]
        return cluster_names
    except Exception as e:
        logger.error(f"Error fetching cluster topics: {str(e)}")
        return []
    
def generate_new_topic_with_llm(document):
    """Use Groq AI to generate a new topic for the document."""
    logger.debug(f"Generating new topic suggestion for document: {document[:50]}...") 
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"Suggest which class does it belong to, "
                        f"E.g., Professional Letter, Resume, Science, Computers, etc. "
                        f"Give only the most meaningful 2-3 classes with an explanation: {document}"
                    )
                }
            ],
            model="llama3-8b-8192",
            stream=False,
        )

        topic_suggestion = chat_completion.choices[0].message.content.strip()
        logger.debug(f"Full response from AI: {topic_suggestion}")

        return topic_suggestion, "" 

    except Exception as e:
        logger.error(f"Error generating topic with Groq AI: {str(e)}")
        return f"Error generating topic with Groq AI: {str(e)}", ""


def retrain_classifier_incrementally(new_embeddings, new_labels):
    """Update the Logistic Regression classifier with new data."""
    logger.debug("Loading pre-trained classifier...")
    try:
        classifier = joblib.load("backend/model/classifier.pkl")
        logger.debug(f"Updating classifier with {len(new_labels)} new samples...")

        logger.debug(f"New embeddings shape: {np.array(new_embeddings).shape}")
        logger.debug(f"New labels: {new_labels}")

        unique_classes = np.unique(new_labels)
        if len(unique_classes) < 2:
            logger.error("At least two classes are required for retraining.")
            return False  

        classifier.fit(new_embeddings, new_labels)

        joblib.dump(classifier, "backend/model/classifier.pkl")
        logger.debug("Classifier updated and saved successfully.")

        reload_classifier()

        return True
    except Exception as e:
        logger.error(f"Error during retraining: {str(e)}")
        return False


def retrain_model_with_new_feedback_incremental():
    """Incrementally update classifier with new data stored in `new_data` folder."""
    try:
        logger.debug("Starting incremental retraining with new data from `new_data` folder...")
        
        new_embeddings = []
        new_labels = []

        for label in os.listdir(NEW_DATA_FOLDER):
            label_folder = os.path.join(NEW_DATA_FOLDER, label)

            if os.path.isdir(label_folder):
                for filename in os.listdir(label_folder):
                    if filename.endswith("_embedding.npy"):
                        embedding_path = os.path.join(label_folder, filename)
                        embedding = np.load(embedding_path)

                        new_embeddings.append(embedding)
                        new_labels.append(label)  

        if not new_embeddings:
            logger.error("No valid data found for retraining.")
            return

        new_embeddings = np.vstack(new_embeddings)
        new_labels = np.array(new_labels)

        logger.debug(f"New embeddings before retraining: {new_embeddings.shape}")
        logger.debug(f"New labels before retraining: {new_labels}")

        if retrain_classifier_incrementally(new_embeddings, new_labels):
            update_cluster_topics(new_labels)  
            reload_classifier()  
            logger.debug("Incremental retraining completed successfully.")
        else:
            logger.error("Incremental retraining failed.")

    except Exception as e:
        logger.error(f"Error during incremental retraining: {str(e)}")


def update_cluster_topics(new_labels):
    """Update the topic mapping after retraining."""
    logger.debug("Updating cluster-topic mapping...")
    new_topic_mapping = {}
    for idx, label in enumerate(new_labels):
        new_topic_mapping[idx] = label

    joblib.dump(new_topic_mapping, "backend/model/cluster_topics.pkl")
    logger.debug("Updated cluster topics saved successfully.")

def reload_classifier():
    """Reload the classifier after it has been retrained."""
    global classifier
    classifier = joblib.load("backend/model/classifier.pkl")
    logger.debug("Classifier reloaded successfully.")

def fetch_documents_from_mod_uploads():
    """Fetch documents from the mod_uploads directory."""
    mod_uploads_path = 'backend/mod_uploads'
    logger.debug(f"Checking directory: {mod_uploads_path}")

    try:
        if not os.path.exists(mod_uploads_path):
            logger.error(f"Directory {mod_uploads_path} does not exist.")
            return []

        documents = [
            filename for filename in os.listdir(mod_uploads_path)
            if os.path.isfile(os.path.join(mod_uploads_path, filename))
        ]

        logger.debug(f"Files found in {mod_uploads_path}: {documents}")
        return documents

    except Exception as e:
        logger.error(f"Error fetching documents: {str(e)}")
        return []
@app.route("/")
def homepage():
    """Render homepage to choose User or Moderator."""
    return render_template("home.html")

@app.route("/user")
def user_home():
    """Render user home page for document upload."""
    return render_template("upload.html")

@app.route("/moderator")
def moderator_home_redirect():
    """Redirect the Moderator link to the actual moderator dashboard."""
    return redirect(url_for("moderator_dashboard"))

@app.route("/upload_document", methods=["POST"])
def upload_document():
    """Handle the document upload."""
    return render_template("upload.html")

@app.route("/submit_document", methods=["POST"])
def submit_document():
    """Endpoint to receive a document, classify it, and handle low-confidence predictions."""
    try:
        if 'document' not in request.files:
            logger.error("No document file part")
            return jsonify({"error": "No document file part"}), 400

        file = request.files['document']
        if file.filename == '':
            logger.error("No file selected")
            return jsonify({"error": "No file selected"}), 400

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)

            uploads_path = os.path.join(UPLOADS_FOLDER, filename)
            mod_uploads_path = os.path.join('backend/mod_uploads', filename)

            file.save(uploads_path)

            document = ""
            if filename.endswith('.txt'):
                with open(uploads_path, 'r') as f:
                    document = f.read()
            elif filename.endswith('.pdf'):
                document = extract_text_from_pdf(uploads_path)
            elif filename.endswith('.docx'):
                document = extract_text_from_docx(uploads_path)

            if not document:
                logger.error("Document is empty.")
                return jsonify({"error": "Document is empty"}), 400

            embedding = generate_embedding(document)
            embedding = np.array(embedding.numpy()).reshape(1, -1)

            prediction_probabilities = classifier.predict_proba(embedding)[0]
            confidence_score = max(prediction_probabilities)
            predicted_label = np.argmax(prediction_probabilities)

            logger.debug(f"Prediction confidence score: {confidence_score * 100:.2f}%")

            if confidence_score < 0.80:
                with open(mod_uploads_path, 'wb') as f:
                    file.seek(0)  
                    f.write(file.read())  
                logger.debug(f"Low confidence document saved to {mod_uploads_path}")

                return redirect(url_for('redirect_to_moderator', document=document, confidence_score=confidence_score))

            else:
                logger.debug(f"Document classified with topic: {cluster_topics.get(predicted_label, 'Unknown Topic')}")
                save_to_new_data(document, embedding, cluster_topics.get(predicted_label, 'Unknown Topic'))

                return redirect(url_for("classified_success", topic=cluster_topics.get(predicted_label, 'Unknown Topic'), confidence_score=f"{confidence_score * 100:.2f}%"))

        else:
            logger.error("Invalid file format.")
            return jsonify({"error": "Invalid file format, supported formats are: txt, pdf, docx"}), 400

    except Exception as e:
        logger.error(f"Error processing document: {e}")
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500



@app.route("/classified_success")
def classified_success():
    """Display success message when document is classified."""
    topic = request.args.get('topic', 'Unknown Topic')
    confidence_score = request.args.get('confidence_score', '0')
    return render_template("classified_success.html", topic=topic, confidence_score=confidence_score)

@app.route("/redirect_to_moderator")
def redirect_to_moderator():
    """Redirect to the moderator for low-confidence predictions."""
    document = request.args.get('document')
    confidence_score = request.args.get('confidence_score')
    return render_template("redirect_to_moderator.html", document=document, confidence_score=confidence_score)


@app.route("/moderator_dashboard")
def moderator_dashboard():
    """Render moderator dashboard to review uploaded documents."""
    MOD_UPLOADS_FOLDER = 'backend/mod_uploads'

    try:
        if not os.path.exists(MOD_UPLOADS_FOLDER):
            logger.error(f"Directory {MOD_UPLOADS_FOLDER} not found.")
            return render_template("moderator_dashboard.html", documents=[], message="No documents available for review.")

        documents = [
            filename for filename in os.listdir(MOD_UPLOADS_FOLDER)
            if os.path.isfile(os.path.join(MOD_UPLOADS_FOLDER, filename))
        ]

        if not documents:
            logger.debug("No documents available for review.")
            return render_template("moderator_dashboard.html", documents=[], message="No documents available for review.")

        logger.debug(f"Fetched documents for review: {documents}")
        return render_template("moderator_dashboard.html", documents=documents)

    except Exception as e:
        logger.error(f"Error fetching documents from mod_uploads: {str(e)}")
        return jsonify({"error": "Error fetching documents."}), 500
    

def get_cluster_topics_from_new_data():
    """Fetch cluster names from the new_data directory."""
    try:
        cluster_names = [
            name for name in os.listdir(NEW_DATA_FOLDER)
            if os.path.isdir(os.path.join(NEW_DATA_FOLDER, name))
        ]
        return cluster_names
    except Exception as e:
        logger.error(f"Error fetching cluster topics: {str(e)}")
        return []

@app.route("/review_document/<string:document_id>", methods=["GET", "POST"])
def review_document(document_id):
    """Handle the review of a selected document by the moderator."""
    document_path = os.path.join(MOD_UPLOADS_FOLDER, document_id)

    try:
        if request.method == "GET":
            if not os.path.exists(document_path):
                logger.error(f"Document not found: {document_path}")
                return jsonify({"error": "Document not found."}), 404

            document_content = ""
            if document_path.endswith('.pdf'):
                document_content = extract_text_from_pdf(document_path)
            elif document_path.endswith('.docx'):
                document_content = extract_text_from_docx(document_path)
            elif document_path.endswith('.txt'):
                with open(document_path, 'r', encoding='utf-8', errors='ignore') as file:
                    document_content = file.read()

            document_content = clean_extracted_text(document_content)

            topic_suggestion, _ = generate_new_topic_with_llm(document_content)

            return render_template(
                "review_document.html",
                document=document_content,
                document_id=document_id,
                topic_suggestion=topic_suggestion
            )

        if request.method == "POST":
            new_topic_cluster = request.form.get("new_topic_cluster")
            document_content = request.form.get("document_content")

            if not new_topic_cluster:
                logger.error("No topic entered.")
                return jsonify({"error": "No topic entered."}), 400

            embedding = generate_embedding(document_content)

            save_to_new_data(document_content, embedding, new_topic_cluster)

            new_embeddings = []
            new_labels = []

            for label in os.listdir(NEW_DATA_FOLDER):
                label_folder = os.path.join(NEW_DATA_FOLDER, label)
                if os.path.isdir(label_folder):
                    for filename in os.listdir(label_folder):
                        if filename.endswith("_embedding.npy"):
                            embedding_path = os.path.join(label_folder, filename)
                            embedding = np.load(embedding_path)
                            new_embeddings.append(embedding)
                            new_labels.append(label) 

            if not new_embeddings:
                logger.error("No valid data found for retraining.")
                return jsonify({"error": "No valid data for retraining."}), 500

            new_embeddings = np.vstack(new_embeddings)
            new_labels = np.array(new_labels)

            try:
                classifier = joblib.load("backend/model/classifier.pkl")
                classifier.fit(new_embeddings, new_labels)
                joblib.dump(classifier, "backend/model/classifier.pkl")
                logger.debug("Classifier updated and saved successfully.")
            except Exception as e:
                logger.error(f"Error updating classifier: {e}")
                return jsonify({"error": f"Error updating classifier: {e}"}), 500

            try:
                cluster_topics = {idx: label for idx, label in enumerate(np.unique(new_labels))}
                joblib.dump(cluster_topics, "backend/model/cluster_topics.pkl")
                logger.debug("Cluster topics updated and saved successfully.")
            except Exception as e:
                logger.error(f"Error updating cluster topics: {e}")
                return jsonify({"error": f"Error updating cluster topics: {e}"}), 500

            if os.path.exists(document_path):
                os.remove(document_path)
                logger.debug(f"Document {document_id} deleted from {MOD_UPLOADS_FOLDER}")

            return render_template("review_result.html", new_topic=new_topic_cluster)

    except Exception as e:
        logger.error(f"Error during document review: {str(e)}")
        return jsonify({"error": "An error occurred during review."}), 500

@app.route("/discard_document/<string:document_id>", methods=["POST"])
def discard_document(document_id):
    """Handle the discarding of a document by the moderator."""
    MOD_UPLOADS_FOLDER = 'backend/mod_uploads'
    document_path = os.path.join(MOD_UPLOADS_FOLDER, document_id)

    try:
        if not os.path.exists(document_path):
            logger.error(f"Document not found: {document_path}")
            return jsonify({"error": "Document not found."}), 404

        os.remove(document_path)
        logger.debug(f"Document {document_id} discarded successfully.")

        return redirect(url_for('moderator_dashboard', message="Document discarded successfully."))

    except Exception as e:
        logger.error(f"Error discarding document: {str(e)}")
        return jsonify({"error": "An error occurred while discarding the document."}), 500


@app.route("/review_result", methods=["GET", "POST"])
def review_result():
    """Display the result after moderator confirms the topic."""
    
    new_topic = request.args.get('new_topic', 'Unknown Topic')  
    message = ""  
    retraining_message = "Retraining initiated with the new topic."
    
    new_data_path = 'backend/new_data' 
    
    topic_exists = False

    for label in os.listdir(new_data_path):
        label_folder = os.path.join(new_data_path, label)
        if os.path.isdir(label_folder) and label == new_topic:
            topic_exists = True
            break
    
    if topic_exists:
        message = f"Topic '{new_topic}' has been added to the already existing cluster."
    else:
        message = f"New topic '{new_topic}' has been created and added."

    return render_template(
        "review_result.html",
        new_topic=new_topic,
        message=message,
        retraining_message=retraining_message
    )

@app.route("/status", methods=["GET"])
def status():
    """Check system status."""
    return jsonify({"status": "System running", "model_status": "Ready"}), 200

if __name__ == "__main__":
    logger.debug("Starting Flask application...")
    app.run(host="0.0.0.0", port=5000, debug=True)
