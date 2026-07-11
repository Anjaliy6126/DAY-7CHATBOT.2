# -*- coding: utf-8 -*-
"""
🧺 Samsung Washing Machine Manual Chatbot
RAG-powered assistant with LangChain, Chroma & OpenAI — dark themed, multi-screen Streamlit app.
"""

import streamlit as st
import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import UnstructuredHTMLLoader
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

# ----------------------------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Samsung Washing Machine Chatbot",
    page_icon="🧺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------
# DARK THEME CSS
# ----------------------------------------------------------------------
st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Poppins', sans-serif;
}

#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* App background */
.stApp {
    background: linear-gradient(160deg, #0b1120 0%, #101a2b 45%, #0c1424 100%);
    color: #eef1ff;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a 0%, #0b1120 100%);
    border-right: 1px solid rgba(96,165,250,0.2);
}
section[data-testid="stSidebar"] * {
    color: #e2e8f0 !important;
}

/* Sidebar nav buttons */
section[data-testid="stSidebar"] .stButton>button {
    width: 100%;
    text-align: left;
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    color: #e2e8f0 !important;
    border-radius: 10px;
    padding: 0.6em 1em;
    margin-bottom: 8px;
    font-weight: 600;
    transition: 0.2s;
}
section[data-testid="stSidebar"] .stButton>button:hover {
    background: linear-gradient(90deg, #2563eb, #1d4ed8);
    border-color: #3b82f6;
    transform: translateX(3px);
}

/* Headings */
h1, h2, h3 {
    background: linear-gradient(90deg, #60a5fa, #a78bfa, #f472b6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 800 !important;
}

p, li, span, label, .stMarkdown {
    color: #d9e1f3 !important;
}

/* Hero banner */
.hero-banner {
    padding: 26px 30px;
    border-radius: 18px;
    background: linear-gradient(120deg, rgba(96,165,250,0.15), rgba(167,139,250,0.10));
    border: 1px solid rgba(255,255,255,0.08);
    box-shadow: 0 8px 28px rgba(0,0,0,0.35);
    margin-bottom: 22px;
}

/* Cards */
.custom-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 20px 22px;
    margin-bottom: 18px;
    box-shadow: 0 6px 22px rgba(0,0,0,0.3);
}

/* Text input */
.stTextInput input {
    background-color: #1a2540 !important;
    color: #f1f5f9 !important;
    border-radius: 10px !important;
    border: 1px solid #334155 !important;
}

/* Main action button */
div.stButton > button {
    background: linear-gradient(90deg, #2563eb, #7c3aed);
    color: white !important;
    border: none;
    border-radius: 12px;
    padding: 0.6em 1.6em;
    font-weight: 700;
    transition: 0.25s;
    box-shadow: 0 4px 16px rgba(37,99,235,0.35);
}
div.stButton > button:hover {
    transform: translateY(-2px) scale(1.01);
    box-shadow: 0 8px 22px rgba(124,58,237,0.5);
}

/* Answer box */
.answer-box {
    background: linear-gradient(135deg, rgba(37,99,235,0.15), rgba(124,58,237,0.10));
    border: 1px solid rgba(96,165,250,0.3);
    border-radius: 16px;
    padding: 20px 22px;
    margin-top: 16px;
    font-size: 16px;
    line-height: 1.6;
}

.footer-note {
    text-align: center;
    color: #94a3b8;
    font-size: 13px;
    margin-top: 30px;
    opacity: 0.8;
}

hr { border-color: rgba(255,255,255,0.1); }

</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------
# API KEY
# ----------------------------------------------------------------------
api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")

# ----------------------------------------------------------------------
# RAG CHAIN BUILDER
# ----------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def build_rag(_api_key):
    loader = UnstructuredHTMLLoader(
        file_path="How to use the various modes of the washing machine _ Samsung LEVANT.html"
    )
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=_api_key)
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    retriever = vectorstore.as_retriever()

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=_api_key)

    prompt = ChatPromptTemplate.from_template("""
You are an assistant for question-answering tasks.
Use the retrieved context to answer the question.
If the answer is not present in the context, say:
"I don't know."
Keep your answer concise.
Question:
{question}
Context:
{context}
Answer:
""")

    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
    )
    return rag_chain

# ----------------------------------------------------------------------
# SIDEBAR NAVIGATION
# ----------------------------------------------------------------------
if "screen" not in st.session_state:
    st.session_state.screen = "💬 Chat"

with st.sidebar:
    st.markdown("### 🧺 Navigation")
    st.markdown("---")

    nav_options = [
        "💬 Chat",
        "📖 About the Manual",
        "⚙️ Settings",
        "ℹ️ About this App",
    ]
    for opt in nav_options:
        if st.button(opt, key=f"nav_{opt}"):
            st.session_state.screen = opt

    st.markdown("---")
    st.markdown("### 🔑 API Status")
    if api_key:
        st.success("Connected ✅")
    else:
        st.error("No API key found ❌")

    st.markdown("---")
    st.caption("Made with 💙 using Streamlit, LangChain & OpenAI")

screen = st.session_state.screen

# ----------------------------------------------------------------------
# HERO HEADER (shown on every screen)
# ----------------------------------------------------------------------
st.markdown("""
<div class="hero-banner">
    <div style="display:flex; align-items:center; gap:16px;">
        <div style="font-size:46px;">🧺✨</div>
        <div>
            <h1 style="margin-bottom:0;">Samsung Washing Machine Manual Chatbot</h1>
            <p style="margin-top:4px; font-size:15px; color:#d9e1f3;">🤖 Ask anything about your washing machine's modes, cycles & settings</p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------
# SCREEN: CHAT
# ----------------------------------------------------------------------
if screen == "💬 Chat":
    if not api_key:
        st.error("⚠️ Please configure your OPENAI_API_KEY in Streamlit secrets or environment variables.")
        st.stop()

    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    st.markdown("#### 🔍 Ask your question")
    question = st.text_input(
        "Your question",
        placeholder="e.g. What is the cycle for DRUM CLEAN?",
        label_visibility="collapsed"
    )
    ask_clicked = st.button("🔮 Get Answer")
    st.markdown('</div>', unsafe_allow_html=True)

    if ask_clicked:
        if question.strip():
            with st.spinner("🔎 Searching the manual..."):
                chain = build_rag(api_key)
                answer = chain.invoke(question)
            st.toast("✅ Answer ready!", icon="🧺")
            st.markdown(f"""
            <div class="answer-box">
            <b>🤖 Answer:</b><br><br>{answer.content}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.warning("⚠️ Please type a question first.")

# ----------------------------------------------------------------------
# SCREEN: ABOUT THE MANUAL
# ----------------------------------------------------------------------
elif screen == "📖 About the Manual":
    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    st.markdown("#### 📖 Source Document")
    st.markdown("""
    This chatbot is grounded in the official Samsung guide:
    **"How to use the various modes of the washing machine — Samsung LEVANT"**

    It covers:
    - 🌀 Wash cycle modes (Cotton, Synthetics, Delicates, etc.)
    - 🧼 Drum Clean cycle
    - ⏱️ Time-based settings & delay options
    - 🌡️ Temperature & spin speed configuration
    """)
    st.markdown('</div>', unsafe_allow_html=True)

# ----------------------------------------------------------------------
# SCREEN: SETTINGS
# ----------------------------------------------------------------------
elif screen == "⚙️ Settings":
    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    st.markdown("#### ⚙️ App Configuration")
    st.markdown(f"""
    - **Model:** `gpt-4o-mini`
    - **Embeddings:** `text-embedding-3-small`
    - **Chunk size:** 1000 (overlap: 200)
    - **Vector store:** Chroma (in-memory)
    - **API key status:** {"✅ Configured" if api_key else "❌ Missing"}
    """)
    st.info("💡 Set `OPENAI_API_KEY` in Streamlit Cloud's Secrets manager — never hardcode it in the script.")
    st.markdown('</div>', unsafe_allow_html=True)

# ----------------------------------------------------------------------
# SCREEN: ABOUT THIS APP
# ----------------------------------------------------------------------
elif screen == "ℹ️ About this App":
    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    st.markdown("#### ℹ️ About")
    st.markdown("""
    A **Retrieval-Augmented Generation (RAG)** chatbot built to answer questions
    about a Samsung washing machine manual — no more digging through PDFs! 🧺

    **Stack:**
    - 🦜 LangChain for orchestration
    - 🎨 Chroma for vector storage
    - 🧠 OpenAI embeddings + GPT-4o-mini for generation
    - ⚡ Streamlit for the interface
    """)
    st.markdown('</div>', unsafe_allow_html=True)

# ----------------------------------------------------------------------
# FOOTER
# ----------------------------------------------------------------------
st.markdown("---")
st.markdown("""
<div class="footer-note">
🧺 Samsung Washing Machine Manual Chatbot &nbsp;|&nbsp; Built with LangChain + Streamlit 💙
</div>
""", unsafe_allow_html=True)
