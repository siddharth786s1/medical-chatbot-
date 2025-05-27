from flask import Flask, render_template, request, redirect, url_for, flash
import google.generativeai as genai
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import os
import logging
import pickle
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.secret_key = 'supersecretkey'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configure Generative AI model
try:
    # Use the API key from the .env file
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found in environment variables")
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-1.5-flash')
    logger.info("Generative AI model configured successfully.")
except Exception as e:
    logger.error(f"Error configuring Generative AI model: {e}", exc_info=True)
    model = None

# Global variable to store the vector store
vector_store = None
VECTOR_STORE_PATH = 'vector_store.pkl'

# Helper to get embeddings (to avoid re-initializing constantly if not necessary)
# However, for pickling FAISS, the embedding function's state is critical.
_hf_embeddings_instance = None
def get_huggingface_embeddings_instance():
    global _hf_embeddings_instance
    if _hf_embeddings_instance is None:
        logger.info("Initializing HuggingFaceEmbeddings instance...")
        _hf_embeddings_instance = HuggingFaceEmbeddings()
        logger.info("HuggingFaceEmbeddings instance initialized.")
    return _hf_embeddings_instance

logger.info(f"Attempting to load vector store from: {VECTOR_STORE_PATH}")
if os.path.exists(VECTOR_STORE_PATH):
    try:
        with open(VECTOR_STORE_PATH, 'rb') as f:
            loaded_object = pickle.load(f)
        if isinstance(loaded_object, FAISS):
            vector_store = loaded_object
            logger.info(f"Successfully loaded FAISS vector_store from {VECTOR_STORE_PATH}. Type: {type(vector_store)}.")
            # Critical check: FAISS needs its embedding_function to work after unpickling for similarity_search
            if hasattr(vector_store, 'embedding_function') and callable(vector_store.embedding_function):
                logger.info("Loaded FAISS object has a callable embedding_function.")
            else:
                logger.warning("Loaded FAISS object is MISSING or has a non-callable embedding_function. Similarity search will likely fail.")
                logger.info("Attempting to re-attach HuggingFaceEmbeddings to loaded FAISS object.")
                # This is a common patch if the embedding function doesn't survive pickling well.
                # Ensure the HuggingFaceEmbeddings are compatible with what was used to create the index.
                try:
                    vector_store.embedding_function = get_huggingface_embeddings_instance()
                    logger.info("Re-attached embedding function successfully.")
                except Exception as attach_e:
                    logger.error(f"Failed to re-attach embedding function: {attach_e}", exc_info=True)

            if hasattr(vector_store, 'index') and vector_store.index is not None:
                 logger.info(f"Loaded FAISS index has {vector_store.index.ntotal} documents.")
            else:
                 logger.warning("Loaded FAISS object has no index or index is None.")
        elif loaded_object is None:
            logger.warning(f"Pickle file {VECTOR_STORE_PATH} contained None. vector_store remains None.")
            vector_store = None
        else:
            logger.warning(f"Pickle file {VECTOR_STORE_PATH} did not contain a FAISS instance. Contained: {type(loaded_object)}. vector_store remains None.")
            vector_store = None
            # Optionally remove the invalid file if it's causing persistent issues
            # try:
            #     os.remove(VECTOR_STORE_PATH)
            #     logger.info(f"Removed invalid pickle file: {VECTOR_STORE_PATH}")
            # except Exception as rm_err:
            #     logger.error(f"Error removing invalid pickle file {VECTOR_STORE_PATH}: {rm_err}")
    except EOFError:
        logger.error(f"EOFError while loading {VECTOR_STORE_PATH}. File may be empty/corrupted. vector_store remains None.")
        if os.path.exists(VECTOR_STORE_PATH):
            try: os.remove(VECTOR_STORE_PATH); logger.info(f"Removed corrupted file: {VECTOR_STORE_PATH}")
            except Exception as rm_e: logger.error(f"Error removing corrupted file {VECTOR_STORE_PATH}: {rm_e}")
        vector_store = None
    except pickle.UnpicklingError as e:
        logger.error(f"UnpicklingError while loading {VECTOR_STORE_PATH}: {e}. vector_store remains None.")
        if os.path.exists(VECTOR_STORE_PATH):
            try: os.remove(VECTOR_STORE_PATH); logger.info(f"Removed unpickleable file: {VECTOR_STORE_PATH}")
            except Exception as rm_e: logger.error(f"Error removing unpickleable file {VECTOR_STORE_PATH}: {rm_e}")
        vector_store = None
    except Exception as e:
        logger.error(f"Unexpected error loading {VECTOR_STORE_PATH}: {e}. vector_store remains None.", exc_info=True)
        if os.path.exists(VECTOR_STORE_PATH): # Attempt to remove if general error
            try: os.remove(VECTOR_STORE_PATH); logger.info(f"Removed problematic file after general error: {VECTOR_STORE_PATH}")
            except Exception as rm_e: logger.error(f"Error removing problematic file {VECTOR_STORE_PATH}: {rm_e}")
        vector_store = None
else:
    logger.info(f"{VECTOR_STORE_PATH} not found. vector_store will be initialized on first upload.")


@app.errorhandler(404)
def page_not_found(e):
    logger.warning(f"404 error encountered for path: {request.path}")
    return redirect(url_for('index'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    global vector_store
    logger.info("Attempting to upload files.")
    try:
        if 'pdf_files' not in request.files:
            flash("No file part")
            logger.warning("Upload attempt with no file part.")
            return redirect(url_for('index'))
        
        files = request.files.getlist('pdf_files')
        if not files or all(file.filename == '' for file in files):
            flash("No selected file")
            logger.warning("Upload attempt with no selected file.")
            return redirect(url_for('index'))

        documents = []
        for file_obj in files: # Renamed 'file' to 'file_obj' to avoid conflict with 'open'
            if file_obj and file_obj.filename != '':
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], file_obj.filename)
                file_obj.save(file_path)
                logger.info(f"Saved uploaded file: {file_path}")
                
                pdf_loader = PyPDFLoader(file_path)
                loaded_docs = pdf_loader.load()
                documents.extend(loaded_docs)
                logger.info(f"Loaded {len(loaded_docs)} documents from {file_obj.filename}.")
        
        if not documents:
            flash("No documents could be loaded from the provided PDF(s).")
            logger.warning("No documents were loaded from the PDF(s).")
            return redirect(url_for('index'))

        current_embeddings = get_huggingface_embeddings_instance()
        
        if vector_store is None:
            logger.info("No existing vector_store, creating new one.")
            vector_store = FAISS.from_documents(documents, current_embeddings)
            if vector_store and hasattr(vector_store, 'index') and vector_store.index is not None:
                logger.info(f"Created new vector_store. Type: {type(vector_store)}. Docs: {vector_store.index.ntotal}.")
            else:
                logger.error("Failed to create new vector_store or it has no index.")
        else:
            logger.info("Existing vector_store found, adding documents.")
            # Ensure the existing vector_store has a working embedding function
            if not (hasattr(vector_store, 'embedding_function') and callable(vector_store.embedding_function)):
                logger.warning("Existing vector_store was missing embedding_function before adding docs. Re-attaching.")
                vector_store.embedding_function = current_embeddings
            vector_store.add_documents(documents) # Modifies in place
            if hasattr(vector_store, 'index') and vector_store.index is not None:
                logger.info(f"Added documents. Existing vector_store now has docs: {vector_store.index.ntotal}.")
            else:
                logger.error("vector_store has no index after adding documents.")
        
        if vector_store:
            logger.info(f"Preparing to save vector_store. Type: {type(vector_store)}. Docs: {vector_store.index.ntotal if hasattr(vector_store, 'index') and vector_store.index is not None else 'N/A'}.")
            if not (hasattr(vector_store, 'embedding_function') and callable(vector_store.embedding_function)):
                logger.warning("CRITICAL: vector_store is MISSING callable embedding_function before pickling. This will likely cause issues on load.")
            else:
                logger.info("vector_store has callable embedding_function before pickling.")
            try:
                with open(VECTOR_STORE_PATH, 'wb') as f:
                    pickle.dump(vector_store, f)
                logger.info(f"Saved updated vector_store to {VECTOR_STORE_PATH}")
            except Exception as e:
                logger.error(f"Error saving vector_store to {VECTOR_STORE_PATH}: {e}", exc_info=True)
        else:
            logger.error("vector_store is None before save attempt in upload. This should not happen if creation/update was successful.")
        
        flash("PDFs uploaded and processed successfully. The knowledge base is ready.")
        return redirect(url_for('index'))
    except Exception as e:
        logger.error(f"An error occurred while processing the PDFs: {e}", exc_info=True)
        flash("An error occurred while processing the PDFs.")
        return redirect(url_for('index'))

@app.route('/ask', methods=['POST'])
def ask():
    global vector_store
    question = request.form.get('prompt', '').strip()
    logger.info(f"Received question: {question}")
    
    context = ""
    custom_prompt = ""
    
    # If a vector store exists (from uploaded PDFs), retrieve relevant text
    if vector_store is not None:
        try:
            relevant_docs = vector_store.similarity_search(question, k=3)
            if relevant_docs:
                context = " ".join([doc.page_content for doc in relevant_docs])
                logger.info(f"Extracted context from PDFs with {len(relevant_docs)} documents.")
            else:
                logger.info("No matching PDF content found for the question.")
        except Exception as e:
            logger.error(f"Error during similarity search: {e}", exc_info=True)

    # Create a prompt that instructs the AI to use the extracted PDF text (if available)
    if context:
        custom_prompt = (
            f"You are a medical assistant. Based solely on the following extracted details from PDFs:\n\n"
            f"{context}\n\n"
            f"Answer the following question in simple and clear terms:\n\n"
            f"Question: {question}"
        )
    else:
        custom_prompt = (
            f"You are a medical assistant. No PDF details are available. "
            f"Answer the following question based on general medical knowledge:\n\n"
            f"Question: {question}"
        )
    
    logger.info(f"Generated prompt (first 500 chars): {custom_prompt[:500]}...")
    
    try:
        response = model.generate_content(custom_prompt)
        if response and response.text:
            logger.info("Successfully received response from LLM.")
            return response.text
        else:
            logger.warning("LLM response was empty or invalid.")
            return "Sorry, I was unable to generate a response to that."
    except Exception as e:
        logger.error(f"Error generating content from LLM: {e}", exc_info=True)
        return "Sorry, an error occurred while trying to get an answer."

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
