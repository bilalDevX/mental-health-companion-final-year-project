import streamlit as st
import requests
import time

# FastAPI Backend URL
FASTAPI_URL = "http://127.0.0.1:8000"

# Custom Theme - Light & Dark Mode
st.markdown(
    """
    <style>
        body {font-family: 'Arial', sans-serif;}
        .stApp {background-color: #f9f9f9;}
        
        .user-bubble {
            background-color: #4CAF50;
            padding: 10px;
            border-radius: 10px;
            color: white;
            margin-bottom: 5px;
        }
        .ai-bubble {
            background-color: #2196F3;
            padding: 10px;
            border-radius: 10px;
            color: white;
            margin-bottom: 5px;
        }
        
        .sidebar-title {
            font-size: 22px;
            font-weight: bold;
            color: #333;
        }
        .sidebar-history {
            background-color: #e3e3e3;
            padding: 10px;
            border-radius: 8px;
        }

        @media (prefers-color-scheme: dark) {
            .stApp {background-color: #181818; color: white;}
            .sidebar-history {background-color: #333; color: white;}
            .user-bubble {background-color: #00897B;}
            .ai-bubble {background-color: #1565C0;}
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Sidebar - Chat History
st.sidebar.markdown("<p class='sidebar-title'>📜 Chat History</p>", unsafe_allow_html=True)

def get_chat_history():
    response = requests.get(f"{FASTAPI_URL}/chat_history")
    if response.status_code == 200:
        return response.json()
    return []

chat_history = get_chat_history()
for chat in chat_history:
    color = "#FF5733" if chat["crisis_detected"] else "#1E88E5"
    with st.sidebar.expander(f"🗨 {chat['text']}"):
        st.markdown(f"<p style='color:{color};'><b>Emotion:</b> {chat['primary_emotion'].capitalize()}</p>", unsafe_allow_html=True)
        if chat["crisis_detected"]:
            st.error("🚨 Crisis Detected!")
        else:
            st.success("✅ No crisis detected.")

st.title("🧠 AI Mental Health Companion")
st.write("💬 Talk to the AI, and it will analyze your emotions and provide support.")

user_input = st.text_area("Type your message:", height=150)

if st.button("Send"):
    if user_input.strip():
        with st.spinner("Analyzing emotions..."):
            response = requests.post(f"{FASTAPI_URL}/analyze_emotions", json={"text": user_input})

        if response.status_code == 200:
            data = response.json()
            st.subheader("🤖 AI's Response")
            st.markdown(f"<div class='ai-bubble'>{data['ai_response']}</div>", unsafe_allow_html=True)
            st.success("✅ No crisis detected." if not data["crisis_detected"] else "🚨 Crisis Detected! Seek Help.")
        else:
            st.error("❌ Error fetching response from API.")

st.write("---")
st.caption("🤖 Powered by FastAPI & Streamlit")
