import streamlit as st
from langchain_helper import process_pdf, answer_question

st.title("PDF Query")

# Initialize session state variables
if 'pdf_processed' not in st.session_state:
    st.session_state.pdf_processed = False
    st.session_state.vectorstore = None
if 'question' not in st.session_state:
    st.session_state.question = ""

# Button to reset processing
if st.session_state.pdf_processed:
    if st.button("Process another PDF"):
        st.session_state.pdf_processed = False
        st.session_state.vectorstore = None
        st.session_state.question = ""
        st.session_state.widget_key = st.session_state.get('widget_key', 0) + 1
        st.rerun()

# File uploader with dynamic key
uploader_key = f"pdf_uploader_{st.session_state.get('widget_key', 0)}"
uploaded_file = st.file_uploader("Upload a PDF file", type="pdf", key=uploader_key)

# Process PDF if uploaded
if uploaded_file is not None and not st.session_state.pdf_processed:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.getvalue())

    with st.spinner("Processing PDF..."):
        st.session_state.vectorstore = process_pdf("temp.pdf")
        st.session_state.pdf_processed = True

    st.success("PDF processed successfully!")

# Question input only if PDF is processed
if st.session_state.pdf_processed:
    def update_question():
        st.session_state.question = st.session_state.question_input

    question = st.text_input("Ask a question about the PDF content:",
                             key="question_input",
                             value=st.session_state.question,
                             on_change=update_question)

    if question:
        with st.spinner("Searching for answer..."):
            answer = answer_question(st.session_state.vectorstore, question)

        st.subheader("Relevant Information:")
        st.write(answer)
