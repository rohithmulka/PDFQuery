import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()


def process_pdf(pdf_file_path):
    loader = PyPDFLoader(pdf_file_path)
    docs = loader.load()

    # setting up splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    # splitting into chunks
    chunks = text_splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings()
    vectorestore = FAISS.from_documents(chunks, embeddings)

    return vectorestore


def answer_question(vectorstore, question):
    llm = ChatOpenAI(model_name='gpt-3.5-turbo')
    # Load your FAISS vector store (make sure it's initialized)
    retriever = vectorstore.as_retriever()

    # Set up a Retrieval QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",  # "map_reduce" or "refine" are other options
        retriever=retriever
    )

    response = qa_chain.invoke({"query": question})

    return response["result"]


st.title("PDF Query")

# Initialize session state variables if they don't exist
if 'pdf_processed' not in st.session_state:
    st.session_state.pdf_processed = False
    st.session_state.vectorstore = None
if 'question' not in st.session_state:
    st.session_state.question = ""

# Process another PDF button - place it at the top if PDF was already processed
if st.session_state.pdf_processed:
    if st.button("Process another PDF"):
        # Reset all session state variables
        st.session_state.pdf_processed = False
        st.session_state.vectorstore = None
        st.session_state.question = ""
        # Use an empty key to force Streamlit to create a new file uploader widget
        st.session_state.widget_key = st.session_state.get('widget_key', 0) + 1
        st.rerun()

# Generate a unique key for the file uploader based on session state
uploader_key = f"pdf_uploader_{st.session_state.get('widget_key', 0)}"
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf", key=uploader_key)

# Process PDF when file is uploaded
if uploaded_file is not None and not st.session_state.pdf_processed:
    # Save the uploaded file temporarily
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.getvalue())

    # Process the PDF
    with st.spinner("Processing PDF..."):
        st.session_state.vectorstore = process_pdf("temp.pdf")
        st.session_state.pdf_processed = True

    st.success("PDF processed successfully!")

# Only show question input if PDF has been processed
if st.session_state.pdf_processed:
    # Use the session state to get/set the question value
    def update_question():
        st.session_state.question = st.session_state.question_input


    # Question input with the current session state value
    question = st.text_input("Ask a question about the PDF content:",
                             key="question_input",
                             value=st.session_state.question,
                             on_change=update_question)

    if question:
        answer_container = st.container()
        with answer_container:
            with st.spinner("Searching for answer..."):
                answer = answer_question(st.session_state.vectorstore, question)

            st.subheader("Relevant Information:")
            st.write(answer)