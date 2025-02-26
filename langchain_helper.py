import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def process_pdf(pdf_file_path):
    """Loads and processes a PDF file into a FAISS vectorstore."""
    loader = PyPDFLoader(pdf_file_path)
    docs = loader.load()

    # Setting up text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    # Splitting into chunks
    chunks = text_splitter.split_documents(docs)

    # Embedding and storing in FAISS
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)

    return vectorstore


def answer_question(vectorstore, question):
    """Retrieves an answer from the vectorstore based on the question."""
    llm = ChatOpenAI(model_name='gpt-4o-mini')
    retriever = vectorstore.as_retriever()

    # Retrieval-based QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever
    )

    response = qa_chain.invoke({"query": question})

    return response["result"]
