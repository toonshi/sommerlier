import streamlit as st
from rag_query import query_knowledge_base
from pdf_loader import load_pdfs_to_chroma
import os
from dotenv import load_dotenv
import time
import shutil
from pinecone import Pinecone
from datetime import datetime
from user_profiles import UserProfile

# Load environment variables
load_dotenv()

# Page config
st.set_page_config(
    page_title="Study Assistant",
    page_icon="📚",
    layout="wide"
)

# Title
st.title("Study Assistant: Learn Smarter")

# Initialize session states
if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'setup_complete' not in st.session_state:
    st.session_state.setup_complete = False

if 'use_gpt_knowledge' not in st.session_state:
    st.session_state.use_gpt_knowledge = True

if 'current_user' not in st.session_state:
    st.session_state.current_user = None

if 'current_topic' not in st.session_state:
    st.session_state.current_topic = None

if 'user_profile' not in st.session_state:
    st.session_state.user_profile = None

# Create three columns: chat, sidebar, and progress
chat_col, sidebar_col, progress_col = st.columns([2, 1, 1])

def save_interaction(query, response):
    """Save user interaction and update progress"""
    if st.session_state.current_user and st.session_state.current_topic:
        st.session_state.user_profile.log_interaction(
            question=query,
            answer=response,
            context_used="GPT" if st.session_state.use_gpt_knowledge else "Document",
            topic=st.session_state.current_topic
        )

with chat_col:
    # Chat interface
    st.write("### Chat with your documents")
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # Chat input
    if st.session_state.current_user and st.session_state.current_topic:
        if prompt := st.chat_input("Ask a question about your documents"):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Get bot response
            with st.chat_message("assistant"):
                response = query_knowledge_base(
                    prompt, 
                    use_gpt_knowledge=st.session_state.use_gpt_knowledge
                )
                st.write(response)
                
                # Save interaction
                save_interaction(prompt, response)
                
                # Add assistant message to chat history
                st.session_state.messages.append(
                    {"role": "assistant", "content": response}
                )
    else:
        st.info("Please select a username and topic to start chatting!")

with sidebar_col:
    st.sidebar.title("Study Settings")
    
    # User Profile Management
    st.sidebar.markdown("### User Profile")
    username = st.sidebar.text_input("Username")
    if username and username != st.session_state.current_user:
        st.session_state.current_user = username
        st.session_state.user_profile = UserProfile(username)
    
    # Topic Selection
    if st.session_state.current_user:
        st.sidebar.markdown("### Topic")
        current_topic = st.sidebar.text_input("Current Topic", key="topic_input")
        if current_topic != st.session_state.current_topic:
            st.session_state.current_topic = current_topic
    
    # Knowledge Mode Toggle
    st.sidebar.markdown("### Knowledge Mode")
    st.session_state.use_gpt_knowledge = st.sidebar.toggle(
        "Use GPT Knowledge",
        value=st.session_state.use_gpt_knowledge,
        help="Toggle between using only document context or including GPT knowledge"
    )
    
    # Document Upload
    st.sidebar.markdown("### Document Upload")
    uploaded_files = st.sidebar.file_uploader(
        "Upload PDF documents", 
        accept_multiple_files=True,
        type=['pdf']
    )
    
    if uploaded_files:
        if st.sidebar.button("Process Documents"):
            with st.spinner("Processing documents..."):
                # Save uploaded files
                if not os.path.exists("uploads"):
                    os.makedirs("uploads")
                
                for uploaded_file in uploaded_files:
                    file_path = os.path.join("uploads", uploaded_file.name)
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                
                # Load documents to ChromaDB
                load_pdfs_to_chroma("uploads")
                st.session_state.setup_complete = True
                st.sidebar.success("Documents processed successfully!")

with progress_col:
    st.markdown("### Learning Progress")
    
    if st.session_state.current_user and st.session_state.user_profile:
        # Get user stats
        stats = st.session_state.user_profile.get_progress_stats()
        
        # Display overall stats
        st.metric("Total Questions Asked", stats["total_questions"])
        st.metric("Understanding Score", f"{stats['understanding_score']:.1f}%")
        
        # Display topic progress
        st.markdown("#### Topics Progress")
        for topic, topic_stats in stats["topics"].items():
            with st.expander(topic):
                st.write(f"Questions Asked: {topic_stats['questions_asked']}")
                if topic_stats['last_interaction']:
                    st.write(f"Last Studied: {topic_stats['last_interaction']}")
    else:
        st.info("Please log in to see your progress!")
