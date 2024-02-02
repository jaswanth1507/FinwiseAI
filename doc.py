from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os
from PyPDF2 import PdfReader
import docx
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from flask_cors import CORS
import json

app = Flask(__name__)
cors = CORS(app)


# Load environment variables or set OpenAI API key directly
load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', 'your-openai-api-key-here')


# Initialize global variables
global conversation

conversation=None


def get_pdf_text(pdf_file):
    reader = PdfReader(pdf_file)
    text = ''.join([page.extract_text() or '' for page in reader.pages])
    return text

def get_docx_text(docx_file):
    doc = docx.Document(docx_file)
    return ' '.join([para.text for para in doc.paragraphs if para.text])

def get_text_chunks(text):
    splitter = CharacterTextSplitter(separator="\n", chunk_size=900, chunk_overlap=100, length_function=len)
    return splitter.split_text(text)

def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings()
    vectorstore = FAISS.from_texts(text_chunks, embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model_name='gpt-3.5-turbo', temperature=0)
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=vectorstore.as_retriever(), memory=memory)
    return chain

def get_files_text(uploaded_files):
    text = ""
    for file in uploaded_files:
        if file.filename.endswith('.pdf'):
            text += get_pdf_text(file)
        elif file.filename.endswith('.docx'):
            text += get_docx_text(file)
    return text

@app.route('/upload', methods=['POST'])
def upload_files():
    global conversation
    if 'file' not in request.files:
        return jsonify(error="No file part"), 400
    files = request.files.getlist('file')
    if not files:
        return jsonify(error="No files selected"), 400

    files_text = get_files_text(files)
    text_chunks = get_text_chunks(files_text)
    vectorstore = get_vectorstore(text_chunks)
    conversation = get_conversation_chain(vectorstore)

    return jsonify(message="Files processed successfully"), 200

@app.route('/ask', methods=['POST'])
def ask_question():
    global conversation
    if not conversation:
        return jsonify({"error": "No files processed. Please upload files first."}), 400
    data = request.get_json()
    if not data or 'question' not in data:
        return jsonify({"error": "No question provided"}), 400
    user_question = data['question']

    response = conversation({'question': user_question})

    if isinstance(response, str):
        # If response is a string, include it directly
        return jsonify({"response": response}), 200
    else:
        # Assuming the response is an object with a content attribute
        return jsonify({"response": response['answer']}), 200


def serialize_human_message(msg):
    # Assuming the msg object has a .content attribute; adjust as necessary
    return {"content": msg.content}

if __name__ == '__main__':
    app.run(host="0.0.0.0",port="5000",debug=True)