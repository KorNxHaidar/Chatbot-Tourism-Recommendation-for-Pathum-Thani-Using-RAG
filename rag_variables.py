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
os.chdir("C:\\Users\\Korn\\Desktop\\dsi314\\Chatbot-Tourism-Recommendation-for-Pathum-Thani-Using-RAG")

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
    template="""
    You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question.
    You should answer the question in Thai language only.
    Context: {context}

    You are an expert travel guide specializing in tourist attractions in Pathum Thani, Thailand.
    The user has the following question:
    Question: {question}

    Please provide a helpful response with the following details:
    1. Name of the attraction
    2. Description of the place (e.g., unique features, activities available)
    3. Opening and closing hours
    4. Additional information (e.g., transportation tips, entrance fees, or special advice)
    
    If you don't know the answer, simply say, "I don't know."
    """,
)
print("Prompt is setting now!")