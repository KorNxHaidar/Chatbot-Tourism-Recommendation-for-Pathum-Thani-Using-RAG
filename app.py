from openai import OpenAI
import streamlit as st
import os
import dotenv
import uuid
import time
from langchain.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from rag_variables import (
    vectorstore, 
    retriever, 
    typhoon_prompt, 
    embeddings,
)

dotenv.load_dotenv()

typhoon_token = os.getenv("TYPHOON_API_KEY")
client = OpenAI(
    api_key=typhoon_token,
    base_url="https://api.opentyphoon.ai/v1" 
)

# Set up Streamlit
st.set_page_config(
    page_title="Typhoon LLM", 
    page_icon="📚", 
    layout="centered", 
    initial_sidebar_state="expanded"
)

# --- Initial Setup ---
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []

def type_effect(message, speed=0.05):
    """แสดงข้อความที่ค่อยๆ เพิ่มทีละตัว"""
    full_message = ""
    message_placeholder = st.empty()
    for i in range(len(message)):
        full_message += message[i]
        message_placeholder.markdown(f"<h1 style='text-align: center; font-size: 4.5em; font-weight: bold;'>{full_message}</h1>", unsafe_allow_html=True)
        time.sleep(speed) 

    time.sleep(1.5) 
    for i in range(len(message), -1, -1):
        message_placeholder.markdown(f"<h1 style='text-align: center; font-size: 4em; font-weight: bold;'>{message[:i]}</h1>", unsafe_allow_html=True)
        time.sleep(speed) 

# เพิ่มการเช็คว่าเคยทำ type_effect แล้วหรือยัง
if 'type_effect_done' not in st.session_state:
    st.session_state.type_effect_done = False

# เช็คว่าถ้ายังไม่เคยทำ type_effect และไม่มีข้อความใน session_state.messages
if not st.session_state.messages and not st.session_state.type_effect_done:
    type_effect("What can I help with?", speed=0.05)
    st.session_state.type_effect_done = True

# Function to generate a response using the Typhoon LLM with type effect
def generate_response(context, chat_history, question, message_placeholder):
    history = "\n".join([f"{entry['role'].capitalize()}: {entry['content']}" for entry in chat_history])
    prompt = typhoon_prompt.format(context=context, question=question) 
    chat_completion = client.chat.completions.create(
        model="typhoon-v1.5x-70b-instruct",
        messages=[{"role": "user", "content": prompt}],
    )

    full_response = chat_completion.choices[0].message.content
    
    # Display the response with type effect
    for i in range(len(full_response)):
        message_placeholder.markdown(full_response[:i+1])
        time.sleep(0.02) 

    return full_response

# Function to answer the user's question
def answer_question(user_question, chat_history):
    retrieved_contexts = retriever.get_relevant_documents(user_question)
    context = "\n".join([doc.page_content for doc in retrieved_contexts])
    response = generate_response(context=context, chat_history=chat_history, question=user_question)
    return response 

# --- Main Chat App ---
# Handle incoming user input and chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Main chat flow where the assistant's response will be shown with type effect
if prompt := st.chat_input("Message Typhoon Chat"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        messages = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
        full_response = generate_response(prompt, messages, prompt, message_placeholder)
        st.session_state.messages.append({"role": "assistant", "content": full_response})
