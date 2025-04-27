from flask import Flask, render_template, request, redirect, url_for, flash
import google.generativeai as genai
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import os
import logging
import pickle

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.secret_key = 'supersecretkey'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure Generative AI model
model = genai.GenerativeModel('gemini-2.0-flash-exp')
genai.configure(api_key="AIzaSyBcEheIeNQwlVnAKq067CBZhqaycVhSEkw")

# Global variable to store the vector store
vector_store = None

# Load existing vector store if available
if os.path.exists('vector_store.pkl'):
    with open('vector_store.pkl', 'rb') as f:
        vector_store = pickle.load(f)

# Define your 404 error handler to redirect to the index page
@app.errorhandler(404)
def page_not_found(e):
    return redirect(url_for('index'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    global vector_store
    try:
        if 'pdf_files' not in request.files:
            flash("No file part")
            return redirect(url_for('index'))
        
        files = request.files.getlist('pdf_files')
        documents = []

        for file in files:
            if file.filename == '':
                flash("No selected file")
                return redirect(url_for('index'))
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)
            
            pdf_loader = PyPDFLoader(file_path)
            documents.extend(pdf_loader.load())
        
        # Create embeddings using HuggingFaceEmbeddings
        embeddings = HuggingFaceEmbeddings()
        
        if vector_store is None:
            # Create a new vector store if it doesn't exist
            vector_store = FAISS.from_documents(documents, embeddings)
        else:
            # Add new documents to the existing vector store
            vector_store.add_documents(documents)
        
        # Save the updated vector store
        with open('vector_store.pkl', 'wb') as f:
            pickle.dump(vector_store, f)
        
        flash("PDFs uploaded and processed successfully. The knowledge base is ready.")
        return redirect(url_for('index'))
    except Exception as e:
        logger.error("An error occurred while processing the PDFs: %s", e)
        flash("An error occurred while processing the PDFs.")
        return redirect(url_for('index'))

@app.route('/ask', methods=['POST'])
def ask():
    global vector_store
    question = request.form['prompt']
    context = "" # Initialize context as empty

    if vector_store is not None:
        # Retrieve relevant documents based on the question if vector_store exists
        relevant_docs = vector_store.similarity_search(question)
        context = " ".join([doc.page_content for doc in relevant_docs])
        custom_prompt = f"You are the best doctor. Only provide medical-related answers. Context: {context} Question: {question}"
    else:
        # If no vector store, ask the question without context
        custom_prompt = f"You are the best doctor. Only provide medical-related answers. Question: {question}"
        flash("Knowledge base is empty. Answering based on general knowledge.") # Optional: Inform the user

    response = model.generate_content(custom_prompt)
    
    cheer_up_line = "Remember, every day is a new beginning. Here's a little story to cheer you up: Once upon a time, in a small village, there lived a kind old man who always had a smile on his face. Despite facing many hardships, he believed in the power of positivity and kindness. One day, his smile and positive attitude saved the village from a terrible disaster. His story reminds us that a positive attitude can change the world. Keep smiling!"

    if response.text:
        return f"{response.text}\n\n{cheer_up_line}"
    else:
        return f"Sorry, but I think Gemini didn't want to answer that!\n\n{cheer_up_line}"

if __name__ == '__main__':
    app.run(debug=True)
