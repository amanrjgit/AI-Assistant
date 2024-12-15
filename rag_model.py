# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 22:25:20 2024

@author: Aman Jaiswar
"""

import os
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import Dict, List, Union

class PersonalAssistantRAG:
    def __init__(
        self, 
        embedding_model: str = 'all-MiniLM-L6-v2', 
        index_path: str = None, 
        answers_path: str = None
    ):
        """
        Initialize the RAG system with FAISS indexing
        
        Args:
            embedding_model (str): Sentence Transformer model name
            index_path (str, optional): Path to existing FAISS index
            answers_path (str, optional): Path to existing answers JSON
        """
        # Ensure data directory exists
        os.makedirs('data', exist_ok=True)
        
        # Default paths if not provided
        self.index_path = index_path or 'data/questions.index'
        self.answers_path = answers_path or 'data/answers.json'
        
        # Initialize embedding model
        self.model = SentenceTransformer(embedding_model)
        
        # FAISS index and answers
        self.index = None
        self.answers = {}
        
        # Load existing index and answers if they exist
        self.load_index_and_answers()
        
    def load_index_and_answers(self):
        """
        Load existing FAISS index and answers if they exist
        """
        try:
            if os.path.exists(self.index_path):
                self.index = faiss.read_index(self.index_path)
            
            if os.path.exists(self.answers_path):
                with open(self.answers_path, 'r') as f:
                    self.answers = json.load(f)
        except Exception as e:
            print(f"Error loading index or answers: {e}")
            self.index = None
            self.answers = {}
        
    def create_index(self, questions: List[str]):
        """
        Create a FAISS index from a list of questions
        
        Args:
            questions (list): List of questions to index
        """
        # Encode questions
        embeddings = self.model.encode(questions)
        
        # Create FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)
        
    def add_qa_pairs(self, questions: List[str], answers: List[str]):
        """
        Add question-answer pairs to the system
        
        Args:
            questions (list): List of questions
            answers (list): Corresponding list of answers
        """
        # Ensure questions and answers are the same length
        assert len(questions) == len(answers), "Questions and answers must be the same length"
        
        # Create or update FAISS index
        self.create_index(questions)
        
        # Store answers
        self.answers = dict(enumerate(answers))
        
        # Save the updated index and answers
        self.save()
        
    def save(self):
        """
        Save the current FAISS index and answers
        """
        if self.index is not None:
            faiss.write_index(self.index, self.index_path)
        
        if self.answers:
            with open(self.answers_path, 'w') as f:
                json.dump(self.answers, f, indent=2)
        
    def query(self, query: str, top_k: int = 1) -> Dict[str, Union[str, float, int]]:
        """
        Retrieve the most similar question's answer
        
        Args:
            query (str): Input query
            top_k (int, optional): Number of top results to retrieve
        
        Returns:
            dict: Retrieved answer details
        """
        if self.index is None or not self.answers:
            return {
                'answer': "No indexed questions available.",
                'distance': float('inf'),
                'index': -1
            }
        
        # Encode query
        query_embedding = self.model.encode([query])
        
        # Search for most similar questions
        distances, top_indices = self.index.search(query_embedding, top_k)
        
        # Retrieve the top answer
        top_index = top_indices[0][0]
        answer = self.answers.get(str(top_index), "I don't have an answer for that.")
        
        return {
            'answer': answer,
            'distance': distances[0][0],
            'index': top_index
        }
    
    def bulk_add(self, qa_dict: Dict[str, str]):
        """
        Add multiple question-answer pairs from a dictionary
        
        Args:
            qa_dict (dict): Dictionary with questions as keys and answers as values
        """
        questions = list(qa_dict.keys())
        answers = list(qa_dict.values())
        
        self.add_qa_pairs(questions, answers)