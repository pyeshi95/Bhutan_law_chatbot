import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from transformers import pipeline

# 1. Load and split documents
def load_documents():
    loader = PyPDFLoader("data/Bhutan_Constitution.pdf")
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_documents(documents)

# 2. Create embeddings and vector database
def create_vector_db(chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = Chroma.from_documents(chunks, embeddings, persist_directory="db")
    vectordb.persist()
    return vectordb

# 3. Load model (local Llama or Mistral)
def load_llm():
    generator = pipeline("text-generation", model="mistralai/Mistral-7B-Instruct-v0.2", device=-1)
    return HuggingFacePipeline(pipeline=generator)

# 4. Build QA system
def build_qa(vectordb, llm):
    retriever = vectordb.as_retriever(search_kwargs={"k": 3})
    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("📚 Bhutan Law Chatbot")
st.write("Ask me questions about Bhutanese laws!")

if "qa" not in st.session_state:
    docs = load_documents()
    vectordb = create_vector_db(docs)
    llm = load_llm()
    st.session_state.qa = build_qa(vectordb, llm)

query = st.text_input("Your Question:")
if query:
    response = st.session_state.qa.run(query)
    st.write("**Answer:**", response)
