# 🌟 **ClustR** 🌟
### *Adaptive Self-Bootstrapping Document Classification System*  

![Python](https://img.shields.io/badge/-Python_3.10-blue?style=for-the-badge&logo=python&logoColor=white)  
![Flask](https://img.shields.io/badge/-Flask_2.0.3-green?style=for-the-badge&logo=flask&logoColor=white)  
![BERT](https://img.shields.io/badge/-BERT-orange?style=for-the-badge)  

---

## 🚀 **Overview**  

**ClustR** is an **Adaptive Document Classification System** combining **BERT embeddings**, **Active Learning**, and **human feedback** to tackle dynamic document trends with minimal manual intervention.  

✨ *Empowering Document Management: Evolving Intelligence Through Human-Centric Feedback.*  

---

## 📜 **Abstract**  

In today's data-driven world, unstructured data management is vital. ClustR employs **state-of-the-art transformers**, **active learning pipelines**, and **incremental retraining** to classify and cluster documents seamlessly, adapting dynamically to emerging topics.  

### **Key Features**:  
1️⃣ **Dynamic Topic Discovery**: Automatically identifies new document categories.  
2️⃣ **Confidence-Based Active Learning**: Flags uncertain predictions for review.  
3️⃣ **Human-in-the-Loop**: Moderators provide corrections for real-time retraining.  
4️⃣ **Iterative Refinement**: Models improve continuously with new data.  
5️⃣ **Customizable and Scalable**: Designed for various industries like research, legal, and knowledge management.  

---

## 🛠 **Tech Stack**  

### **🧑‍💻 Languages & Frameworks**  
- 🐍 **Python 3.10**  
- 🌐 **Flask** (Web Framework)  

### **📚 Libraries**  
- 🤗 **Transformers**: BERT for contextual embeddings  
- ⚙️ **Scikit-learn**: Logistic Regression Classifier  
- 🔥 **PyTorch**: Embedding generation  
- 📄 **PDFPlumber**, **PyPDF2**, **pytesseract**: Document text extraction  
- 🏗 **Joblib**: Model storage  

### **⚙️ Tools**  
- 🧠 **Groq AI API**: Topic suggestions via LLMs  
- 🎨 **Jinja2**: Template rendering  

---

## 🔍 **How It Works**  

1️⃣ **📂 Upload Document**: Users upload `.txt`, `.pdf`, or `.docx` files.  
2️⃣ **🧠 Semantic Embedding**: BERT generates embeddings capturing document essence.  
3️⃣ **🤖 Auto-Classification**: Documents with confidence > 80% are auto-categorized.  
4️⃣ **🧑‍⚖️ Moderator Feedback**: Low-confidence documents are flagged for human review.  
5️⃣ **♻️ Real-Time Retraining**: Incorporates moderator corrections to enhance model accuracy dynamically.  

---

## 🏗 **Setup Instructions**  

### **💡 Pre-requisites**  
Ensure you have the following installed:  
- 🐍 Python 3.10+  
- 🛠 pip (Python package manager)  

### **⚙️ Installation**  


#### 1. Clone the repository
```bash
git clone https://github.com/your_username/ClustR.git
cd ClustR
```
#### 2. Install dependencies
```bash
pip install -r requirements.txt
```
#### 3. Download Pretrained BERT Weights
```bash
transformers-cli download bert-base-uncased
```
#### 4. Set up environment variables
###### Create a .env file and configure it with necessary variables:
```bash
echo "GROQ_API_KEY=your_groq_api_key" > .env
```
#### 5. Run the Application
```bash
python app.py
```
#### 6. Access the application
###### Open your browser and navigate to:
```bash
http://localhost:5000
```
---

## 💻 **Usage**  

1️⃣ **User Dashboard**: Upload documents for classification.  
2️⃣ **Moderator Dashboard**: Review flagged documents and provide corrections.  
3️⃣ **Dynamic Adaptation**: Observe the system evolving with real-time updates.  

---

## 🌟 **Key Features**  

- **📈 Adaptive Learning**: Incorporates feedback to refine classification.  
- **📂 Dynamic Topic Discovery**: Uncovers hidden patterns in unstructured data.  
- **💬 Human-AI Collaboration**: Combines automation with expert input for optimal results.  
- **⚡ Scalable & Customizable**: Designed for industries like research, law, and knowledge management.  

---

## 📊 **Expected Impact**  

ClustR is designed to **revolutionize document classification** by reducing manual efforts, accelerating categorization, and ensuring adaptability to evolving topics.  
✨ Industries that benefit: **Legal**, **Research**, **Knowledge Management**  

---

   ## 👥 Contributors
   - **Shashank Mahato**
     - [LinkedIn](https://www.linkedin.com/in/shashank-mahato/) 🌐
   - **Shuddhasattwa Majumder**
     - [LinkedIn](https://www.linkedin.com/in/shuddhasattwa-majumder/) 🌐

---

