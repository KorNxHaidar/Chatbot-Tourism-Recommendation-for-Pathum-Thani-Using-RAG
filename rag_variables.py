import sys
import re
import pandas as pd
import json
import os
import dotenv

from pathlib import Path
from typing import overload
from langchain_core.documents import Document
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain_cohere import CohereEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from sentence_transformers import SentenceTransformer
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.vectorstores import VectorStore
from openai import OpenAI

dotenv.load_dotenv()
# os.chdir("C:\\Users\\Korn\\Desktop\\dsi314\\Chatbot-Tourism-Recommendation-for-Pathum-Thani-Using-RAG")

#HuggingFaceEmbeddings
model_name = "BAAI/bge-m3"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': True}
embeddings = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)
load_path = "./vectorstore_directory"
vectorstore = FAISS.load_local(load_path, embeddings, allow_dangerous_deserialization=True)
print("Vector Store loaded successfully!")

# Retrieve and generate using the relevant snippets of the blog.
retriever = vectorstore.as_retriever()
print("Retriver successfully!")

typhoon_prompt = PromptTemplate(
    input_variables=["context","question"],
    template = """
    You are an intelligent assistant for question-answering tasks about Pathum Thani, Thailand. 
    Analyze the context and question carefully to provide an appropriate response.

    Context: {context}

    Scenario Detection:
    1. If the question is about a tourist attraction:
    Response Format:
    - Name of the attraction 
    - Description of the place (e.g., unique features, activities available) 
    - Opening and closing hours (if available)
    - Additional information (e.g., transportation tips, entrance fees, or special advice)
    - Rating Score (1-5 stars)
    - Total Reviews
    - Example Review (one authentic review)

    2. If the question is about a restaurant:
    Response Format:
    - Name of the attraction 
    - Description of the place (e.g., unique features, activities available) 
    - Opening and closing hours (if available)
    - Additional information (e.g., transportation tips, entrance fees, or special advice)
    - Rating Score (1-5 stars)
    - Total Reviews
    - Example Review (one authentic review)

    3. For general conversational questions:
    Respond naturally in Thai, addressing the specific query without a fixed format.

    Question: {question}

    Important Guidelines:
    - Always respond in Thai language
    - Provide accurate and helpful information
    - If no information is available, respond with "I don't know."
    """,
)
print("Prompt is setting now!")
