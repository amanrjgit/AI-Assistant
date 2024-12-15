# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 22:28:27 2024

@author: Aman Jaiswar
"""

import streamlit as st
import os
import json
from rag_model import PersonalAssistantRAG

# Configuration
INDEX_PATH = 'data/questions.index'
ANSWERS_PATH = 'data/answers.json'

# Ensure data directory exists
os.makedirs('data', exist_ok=True)

# Initialize session state
if 'assistant' not in st.session_state:
    st.session_state.assistant = PersonalAssistantRAG(
        index_path=INDEX_PATH if os.path.exists(INDEX_PATH) else None,
        answers_path=ANSWERS_PATH if os.path.exists(ANSWERS_PATH) else None
    )

# Page configuration
st.set_page_config(
    page_title="Personal Knowledge Assistant", 
    page_icon="🧠", 
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
.main-container {
    max-width: 900px;
    margin: 0 auto;
    padding: 20px;
    background-color: #f9f9f9;
    border-radius: 15px;
}
.stTextInput > div > div > input {
    background-color: white;
    border: 2px solid #e0e0e0;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

# Main App
st.title("🧠 Personal Knowledge Assistant")

# Sidebar for Q&A Management
st.sidebar.title("Knowledge Management")

# Q&A Input Section
with st.sidebar.form(key='qa_form'):
    st.subheader("Add New Q&A Pair")
    new_question = st.text_input("Question")
    new_answer = st.text_area("Answer")
    submit_button = st.form_submit_button(label='Add Q&A Pair')

    if submit_button and new_question and new_answer:
        # Check if we have existing QA pairs
        try:
            with open(ANSWERS_PATH, 'r') as f:
                existing_qa = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            existing_qa = {}
        
        # Add new Q&A pair
        existing_qa[new_question] = new_answer
        
        # Save updated QA pairs
        with open(ANSWERS_PATH, 'w') as f:
            json.dump(existing_qa, f, indent=2)
        
        # Update RAG model
        st.session_state.assistant.bulk_add({new_question: new_answer})
        st.session_state.assistant.save_index(INDEX_PATH, ANSWERS_PATH)
        
        st.sidebar.success("Q&A Pair Added Successfully!")

# File Upload for Bulk Q&A
uploaded_file = st.sidebar.file_uploader("Upload Q&A JSON", type=['json'])
if uploaded_file is not None:
    # Read the uploaded JSON file
    qa_data = json.load(uploaded_file)
    
    # Add to RAG model
    st.session_state.assistant.bulk_add(qa_data)
    st.session_state.assistant.save_index(INDEX_PATH, ANSWERS_PATH)
    
    st.sidebar.success(f"Added {len(qa_data)} Q&A Pairs!")

# Advanced Options
with st.sidebar.expander("Advanced Options"):
    top_k = st.slider("Number of Top Results", min_value=1, max_value=5, value=1)
    st.session_state.top_k = top_k

# Main Chat Interface
st.markdown('<div class="main-container">', unsafe_allow_html=True)

# Chat input
query = st.text_input("Ask a question from your knowledge base", 
                      placeholder="What would you like to know?")

# Query handling
if query:
    # Perform RAG query
    result = st.session_state.assistant.query(query, top_k=st.session_state.get('top_k', 1))
    
    # Display result
    st.markdown("### 🤖 Assistant's Response")
    st.write(result['answer'])
    
    # Show confidence and additional details
    with st.expander("Query Details"):
        st.write(f"**Confidence Score:** {result['distance']:.4f}")
        st.write(f"**Matched Index:** {result['index']}")

st.markdown('</div>', unsafe_allow_html=True)

# Optional: View Current Knowledge Base
# st.sidebar.title("Current Knowledge Base")
# try:
#     with open(ANSWERS_PATH, 'r') as f:
#         current_qa = json.load(f)
    
#     for q, a in current_qa.items():
#         st.sidebar.expander(q).write(a)
# except (FileNotFoundError, json.JSONDecodeError):
#     st.sidebar.info("No Q&A pairs added yet.")
