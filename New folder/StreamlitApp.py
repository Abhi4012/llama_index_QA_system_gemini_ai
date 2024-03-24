import os
import streamlit as st
from llama_index.core import SimpleDirectoryReader
from QAWithPDF.data_ingestion import load_data
from QAWithPDF.embedding import download_gemini_embedding
from QAWithPDF.model_api import load_model

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

def main():
    st.set_page_config(page_title="QA with Documents")
    
    # Setting background color and padding
    st.markdown(
        """
        <style>
        .reportview-container {
            background: linear-gradient(135deg, #c3e7fd, #f8d6fd);
            color: #333333;
            padding: 1rem;
        }
        .st-bb {
            border-radius: 12px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1), 0 1px 3px rgba(0, 0, 0, 0.08);
            padding: 20px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    # Setting custom font
    st.markdown("<style>h1 {color: #1E90FF; font-family: Arial, sans-serif;}</style>", unsafe_allow_html=True)
    
    # Setting header
    st.title("QA with Documents (Information Retrieval)")
    
    # Uploading document
    doc = st.file_uploader("Upload your document")
    
    # Text input for user question
    user_question = st.text_input("Ask your question")
    
    # Retrieve Google API key from environment variable
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    
    # Processing button
    if st.button("Submit & Process"):
        with st.spinner("Processing..."):
            document = load_data(doc)
            model = load_model()
            query_engine = download_gemini_embedding(model, document, api_key=GOOGLE_API_KEY)
                
            response = query_engine.query(user_question)
                
            # Displaying response
            st.markdown(f"<div class='st-bb'>{response.response}</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
