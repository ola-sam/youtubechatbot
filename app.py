import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.document_loaders.parsers import OpenAIWhisperParser
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.blob_loaders.youtube_audio import YoutubeAudioLoader
from youtube_transcript_api import YouTubeTranscriptApi
from urllib.parse import urlparse, parse_qs
import shutil
import os
from dotenv import load_dotenv

load_dotenv()

# Page settings
st.set_page_config(layout="wide")

# OpenAI API Key
# OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

# Model selection
model_options = ["gpt-3.5-turbo", "gpt-4o-mini"]
selected_model = st.sidebar.selectbox("Select LLM Model", model_options, index=0)

# Langchain Initialization
llm = ChatOpenAI(api_key=OPENAI_API_KEY, model_name=selected_model)
output_parser = StrOutputParser()
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful and engaging assistant ready to chat."),
    ("user", "{question}\n\nBased on the video content: {content}")
])
chain = prompt | llm | output_parser

# App Header
st.title("🎬 Chat with YouTube Videos")
st.markdown("Enter a YouTube URL to extract content and have a conversation about it!")

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "content" not in st.session_state:
    st.session_state.content = ""

@st.cache_data
def get_transcript_content(url):
    try:
        parsed_url = urlparse(url)
        video_id = parse_qs(parsed_url.query).get('v', [None])[0]
        if not video_id:
            st.error("Invalid YouTube URL. Please provide a valid URL.")
            return ""
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        return " ".join([t["text"] for t in transcript])
    except Exception as e:
        st.error(f"Failed to get transcript: {e}")
        return ""

@st.cache_data
def video_to_text(url):
    try:
        save_dir = "./temp/"
        loader = GenericLoader(YoutubeAudioLoader([url], save_dir), OpenAIWhisperParser())
        docs = loader.load()
        result = " ".join(doc.page_content for doc in docs)
        shutil.rmtree(save_dir)  # Clean up temporary files
        return result
    except Exception as e:
        st.error(f"Failed to process video audio: {e}")
        return ""

def display_video_and_transcript(url):
    col1, col2 = st.columns([6, 4])
    with col1:
        st.video(url, start_time=0)
        with st.expander("Video Transcript", expanded=False):
            st.write(st.session_state.content)
    return col2

def display_chat_interface():
    st.subheader("Chat with your video")
    chat_container = st.container()
    with chat_container:
        chat_placeholder = st.empty()
        chat_placeholder.markdown("""
            <div style="height:400px; overflow-y:auto; border:1px solid #ddd; padding:10px; border-radius:10px; background-color:#f9f9f9;">
        """, unsafe_allow_html=True)
        
        for question, answer in st.session_state.chat_history:
            with st.chat_message("user"):
                st.write(question)
            with st.chat_message("assistant"):
                st.write(answer)

        chat_placeholder.markdown("</div>", unsafe_allow_html=True)
    
    if st.button("Reset Chat", key="reset_chat"):
        st.session_state.chat_history = []
        st.rerun()

def main_page():
    url = st.text_input("Enter YouTube URL:")
    if st.button("Load Video", type="primary"):
        with st.spinner("Extracting content..."):
            content = get_transcript_content(url) or video_to_text(url)
            st.session_state.content = content
            st.toast("Video content loaded successfully!", icon="🎉")

    if st.session_state.content:
        col2 = display_video_and_transcript(url)
        
        with col2:
            display_chat_interface()
            
            # Suggested prompts
            user_input = ""
            suggested_prompts = {
                "Explain the content in simple terms": "Explain the content in this video in simple terms",
                "Give me a summary": "Give me a summary",
                "Give me real-life examples": "Give me real-life examples"
            }
            for label, prompt_text in suggested_prompts.items():
                if st.button(label):
                    user_input = prompt_text
            
            if not user_input:
                user_input = st.chat_input("Ask me anything about the video:")
            
            if user_input:
                with st.spinner("Thinking..."):
                    try:
                        response = chain.invoke({"content": st.session_state.content, "question": user_input})
                        st.session_state.chat_history.append((user_input, response))
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error generating response: {e}")

if __name__ == '__main__':
    main_page()
