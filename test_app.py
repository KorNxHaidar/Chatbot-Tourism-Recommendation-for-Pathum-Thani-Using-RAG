from openai import OpenAI
import streamlit as st
import os
import dotenv
import uuid
import time

# โหลด environment variables (API key)
dotenv.load_dotenv()

typhoon_token = os.getenv("TYPHOON_API_KEY")
# ตั้งค่า Typhoon API
client = OpenAI(
    api_key=typhoon_token,
    base_url="https://api.opentyphoon.ai/v1"  # URL ของ Typhoon API
)

# ตั้งค่า Streamlit
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

# การแสดงข้อความที่ค่อยๆ เพิ่มทีละตัว
def type_effect(message, speed=0.05):
    """แสดงข้อความที่ค่อยๆ เพิ่มทีละตัว"""
    full_message = ""
    message_placeholder = st.empty()  # Placeholder ที่ใช้แสดงข้อความ
    for i in range(len(message)):
        full_message += message[i]
        message_placeholder.markdown(f"<h1 style='text-align: center; font-size: 4.5em; font-weight: bold;'>{full_message}</h1>", unsafe_allow_html=True)
        time.sleep(speed)  # รอเวลาเพื่อให้แสดงตัวอักษรแต่ละตัว

    # หลังจากข้อความทั้งหมดแสดงเสร็จแล้ว รอระยะเวลาและค่อยๆ ทำให้ข้อความหายไปทีละตัว
    time.sleep(1.5)  # รอเวลาหลังจากข้อความแสดงหมดแล้ว
    for i in range(len(message), -1, -1):
        message_placeholder.markdown(f"<h1 style='text-align: center; font-size: 4em; font-weight: bold;'>{message[:i]}</h1>", unsafe_allow_html=True)
        time.sleep(speed)  # รอเวลาให้ข้อความหายไปทีละตัว

# เพิ่มการเช็คว่าเคยทำ type_effect แล้วหรือยัง
if 'type_effect_done' not in st.session_state:
    st.session_state.type_effect_done = False

# เช็คว่าถ้ายังไม่เคยทำ type_effect และไม่มีข้อความใน session_state.messages
if not st.session_state.messages and not st.session_state.type_effect_done:
    type_effect("What can I help with?", speed=0.05)
    # ตั้งค่าว่า type_effect ทำงานไปแล้ว
    st.session_state.type_effect_done = True



# Main Chat app ใช้ Typhoon
def stream_typhoon_response(messages):
    response = client.chat.completions.create(
        model="typhoon-v1.5x-70b-instruct",
        messages=messages,
        temperature=0.3,
        stream=True
    )
    for chunk in response:
        content = chunk.choices[0].delta.content
        if content:
            yield content

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


if prompt := st.chat_input("Message Typhoon Chat"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        messages = [
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state.messages
        ]

        for chunk in stream_typhoon_response(messages):
            full_response += chunk
            message_placeholder.markdown(full_response)

