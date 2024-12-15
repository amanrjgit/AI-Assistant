# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 22:29:14 2024

@author: Aman Jaiswar
"""

import streamlit as st
import os
import json
import sys
import traceback

# Add src directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from rag_model import PersonalAssistantRAG
from utils import create_custom_styles

# Configuration
INDEX_PATH = 'data/questions.index'
ANSWERS_PATH = 'data/answers.json'

# Page Configuration
st.set_page_config(
    page_title="Personal Knowledge Assistant", 
    page_icon="🧠", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply Custom Styles
create_custom_styles()

# Initialize Assistant
@st.cache_resource
def get_rag_assistant():
    return PersonalAssistantRAG(
        index_path=INDEX_PATH,
        answers_path=ANSWERS_PATH
    )

# Main App
def main():
    # Title
    st.title("🧠 Personal Knowledge Assistant")
    st.subheader("Your Intelligent Q&A Companion")

    # Sidebar for Q&A Management
    with st.sidebar:
        st.header("Knowledge Management")
        
        # Q&A Input Section
        with st.form(key='qa_form'):
            st.subheader("Add New Q&A Pair")
            new_question = st.text_input("Question")
            new_answer = st.text_area("Answer")
            submit_button = st.form_submit_button(label='Add Q&A Pair')

            if submit_button and new_question and new_answer:
                try:
                    # Get RAG assistant
                    assistant = get_rag_assistant()
                    
                    # Add new Q&A pair
                    assistant.bulk_add({new_question: new_answer})
                    st.success("Q&A Pair Added Successfully!")
                except Exception as e:
                    st.error(f"Error adding Q&A pair: {e}")

        # File Upload for Bulk Q&A
        uploaded_file = st.file_uploader("Upload Q&A JSON", type=['json'])
        if uploaded_file is not None:
            try:
                # Get RAG assistant
                assistant = get_rag_assistant()
                
                # Read the uploaded JSON file
                qa_data = json.load(uploaded_file)
                
                # Add to RAG model
                assistant.bulk_add(qa_data)
                st.success(f"Added {len(qa_data)} Q&A Pairs!")
            except Exception as e:
                st.error(f"Error uploading Q&A file: {e}")

        # View Current Knowledge Base
        st.subheader("Current Knowledge Base")
        try:
            with open(ANSWERS_PATH, 'r') as f:
                current_qa = json.load(f)
            
            for q, a in current_qa.items():
                with st.expander(q[:50] + "..." if len(q) > 50 else q):
                    st.write(a)
        except (FileNotFoundError, json.JSONDecodeError):
            st.info("No Q&A pairs added yet.")

    # Main Chat Interface
    st.markdown('<div class="main-container">', unsafe_allow_html=True)

    # Query Input
    query = st.text_input("Ask a question from your knowledge base", 
                          placeholder="What would you like to know?")

    # Advanced Options
    col1, col2 = st.columns([3, 1])
    with col1:
        top_k = st.slider("Number of Top Results", min_value=1, max_value=5, value=1)
    
    # Query handling
    if query:
        try:
            # Get RAG assistant
            assistant = get_rag_assistant()
            
            # Perform RAG query
            result = assistant.query(query, top_k=top_k)
            
            # Display result
            st.markdown("### 🤖 Assistant's Response")
            st.write(result['answer'])
            
            # Show confidence and additional details
            with st.expander("Query Details"):
                st.write(f"**Confidence Score:** {result['distance']:.4f}")
                st.write(f"**Matched Index:** {result['index']}")
        
        except Exception as e:
            st.error("An error occurred during query processing.")
            st.error(traceback.format_exc())

    st.markdown('</div>', unsafe_allow_html=True)

# Run the app
if __name__ == "__main__":
    main()