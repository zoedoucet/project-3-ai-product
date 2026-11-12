"""The app you demo on Friday. Run it from the repo root:

    streamlit run app/streamlit_app.py

This is a skeleton: replace every TODO with your own product. Day 3 of Week 8
(Streamlit, from notebook to app) shows each piece.
"""

import sys

import streamlit as st

sys.path.append(".")                       # so `src` is importable when you run from the repo root
from src.functions import *                # noqa: E402,F401  the same functions as your notebooks
import json
import os
from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from dotenv import load_dotenv

st.set_page_config(page_title="Tree Health Investigator", page_icon="🔎", layout ="centered")
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background-color: #FAF9F6;
        color: #263B2A;
    }

    /* Main content spacing */
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 1rem;
    }

    /* Title */
    h1 {
        color: #263F2A;
        font-family: Georgia, serif;
        text-align: center;
    }

    /* Subtitle */
    .stApp p {
        color: #4F5750;
    }

    /* Assistant messages: pale sage green */
    [data-testid="stChatMessage"] {
        background-color: #E8EDE5;
        border-radius: 16px;
        padding: 0.75rem 1rem;
    }

    /* User messages: light grey */
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
        background-color: #DCDCDC;
    }

    /* Question input */
    div[data-testid="stTextInput"] input {
        background-color: #FFFFFF;
        border-radius: 12px;
    }

    /* Buttons */
    div.stButton > button {
        background-color: #D0DDCA;
        color: #263B2A;
        border: none;
        border-radius: 10px;
    }

    div.stButton > button:hover {
        background-color: #B7C9B0;
        color: #263B2A;
        border: none;
    }

    /* Send button */
    div[data-testid="stHorizontalBlock"] > div:last-child div.stButton > button {
        background-color: #DCE8D6;
        color: #263B2A;
    }

    div[data-testid="stHorizontalBlock"] > div:last-child div.stButton > button:hover {
        background-color: #C6D9BF;
        color: #263B2A;
    }

    div[data-testid="stHorizontalBlock"] > div:last-child div.stButton > button:hover {
        background-color: #4F704A;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
    <h1 style="text-align: center; width: 100%; margin: 0;">
        Tree Health Investigator
    </h1>""",unsafe_allow_html=True)

st.markdown(""" 
 <p style="text-align: center; color: var(--text-color); opacity: 0.7;">
        An AI-powered assistant for understanding insect-related tree health problems,<br>
        grounded in forest entomology research.
    </p>""",  unsafe_allow_html=True)

if st.button("ℹ️ About & How to Use"):
    st.switch_page("pages/1_About.py")
    
@st.cache_resource
def load_core():
    """Load the saved model, or build the pipeline, ONCE per session (not on every rerun).

    LLM track:   return the Chroma collection and anything else the pipeline needs.
                 Read the key with st.secrets["OPENAI_API_KEY"] or from .env, never from the code.
    """
    root = Path(__file__).resolve().parent.parent

    # loading the saved pipeline
    with open(root/"models"/"rag_config.json", encoding = "utf-8")as file:
        config = json.load(file)

    # loading the embedding model
    model = SentenceTransformer(config["embedding_model"])

    # connecting to ChromaDB
    chroma_client = chromadb.PersistentClient(path=str(root / "models" / "chroma"))
    collection = chroma_client.get_collection(name=config["collection_name"])

    # loading API Key
    load_dotenv(root / ".env")
    api_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")

    if not api_key:
        raise ValueError("OPENAI_API_KEY is missing")

    client = OpenAI(api_key=api_key)
    
    return model, collection, client, config


model, collection, client, config = load_core()

# creating conversation history
if "messages" not in st.session_state:
    st.session_state.messages =[
        {"role":"assistant",
         "content": "Hello, I am the Tree Health Investigator. How can I help you?"}
    ]

# creating evidence memory
if "evidence" not in st.session_state:
    st.session_state.evidence = None

# displaying all messages in the current conversation
with st.container(height= 300, border = True):
    for message in st.session_state.messages:
        with st.chat_message(message["role"],
                             avatar=":material/forest:" if message["role"] == "assistant" else "👤"):
            st.markdown(message["content"])

def send_question():
    st.session_state.submitted_question = st.session_state.question_input
    st.session_state.question_input = ""
    
# Question field
question = st.text_input(
    "Ask a question about your tree",
    placeholder="Describe your tree's symptoms...",
    label_visibility="collapsed",
    key="question_input")

# Buttons underneath the question field
left_col, spacer_col, right_col = st.columns([2, 5, 1])

with left_col:
    # New Investigation button (clears the chat)
    if st.button("New Investigation"):
        st.session_state.messages = [{
            "role": "assistant",
            "content": "Hello, I am the Tree Health Investigator. How can I help you?"
        }]
        st.session_state.evidence = None
        st.rerun()

with right_col:
    # Send button
    send = st.button("Send", on_click= send_question)


# Chatbot interaction
if st.session_state.get("submitted_question"):

    question = st.session_state.submitted_question
    st.session_state.submitted_question = None

    if not question.strip():
        st.warning("Please enter a question.")

    else:
        # Saving the user's question
        st.session_state.messages.append({
            "role": "user",
            "content": question})

        # Generating the answer
        with st.spinner("Investigating your question..."):
            answer, evidence = ask_tree_assistant(
                question=question,
                model=model,
                collection=collection,
                client=client,
                k=config["k"],
                history=st.session_state.messages[:-1],
                previous_results=st.session_state.evidence,
                return_results=True)

            st.session_state.evidence = evidence

        # Saving the assistant's answer
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer
        })
        # Displaying the updated conversation
        st.rerun()