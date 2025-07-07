import streamlit as st
import requests
import io
import time
import json
import base64
import pandas as pd
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any
import hashlib
import zipfile
from PIL import Image
import numpy as np

# --- Configuration ---
# IMPORTANT: For deployment, change this to your deployed FastAPI backend URL
# Example: API_BASE = "https://your-backend-url.onrender.com/api"
# In app.py
API_BASE = "https://adarshdivase-ai-toolkit-backend.hf.space/api"

st.set_page_config(
    page_title="AI Services Toolkit Pro",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for better styling ---
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .feature-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
        color: #333333;
    }
    
    .metric-card {
        background: linear-gradient(45deg, #667eea, #764ba2);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
    }
    
    .success-message {
        background: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 5px;
        border: 1px solid #c3e6cb;
    }
    
    .error-message {
        background: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 5px;
        border: 1px solid #f5c6cb;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: #f0f2f6;
        border-radius: 10px 10px 0 0;
        color: #262730;
        font-weight: 500;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #667eea;
        color: white;
    }
    
    .upload-section {
        border: 2px dashed #667eea;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        background: #f8f9ff;
    }
    
    .stats-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    .sidebar-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #e9ecef;
        margin-bottom: 1rem;
    }

    /* Fixed content display boxes with better contrast */
    .content-display {
        background: #ffffff;
        color: #333333;
        padding: 1.5rem;
        border-radius: 10px;
        border: 2px solid #667eea;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .original-text {
        background: #fff3cd;
        color: #856404;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #ffc107;
        margin: 0.5rem 0;
    }
    
    .translated-text {
        background: #d1ecf1;
        color: #0c5460;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #17a2b8;
        margin: 0.5rem 0;
    }
    
    .generated-content {
        background: #f8f9fa;
        color: #495057;
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #dee2e6;
        margin: 1rem 0;
        line-height: 1.6;
    }
    
    .transcription-result {
        background: #e7f3ff;
        color: #004085;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #007bff;
        margin: 1rem 0;
    }

    /* Sentiment analysis specific styling */
    .sentiment-positive {
        background: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        border: 2px solid #28a745;
    }
    
    .sentiment-negative {
        background: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        border: 2px solid #dc3545;
    }
</style>
""", unsafe_allow_html=True)

# --- Session State Initialization ---
# Initialize all session state variables to ensure persistence across reruns
if 'history' not in st.session_state:
    st.session_state.history = []
if 'favorites' not in st.session_state:
    st.session_state.favorites = []
if 'api_calls_count' not in st.session_state:
    st.session_state.api_calls_count = 0
if 'user_preferences' not in st.session_state:
    st.session_state.user_preferences = {
        'theme': 'Light',
        'default_voice': 'Female',
        'default_language': 'English',
        'auto_save': True
    }
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'translation_history' not in st.session_state:
    st.session_state.translation_history = []

# --- Helper Functions ---
@st.cache_data(ttl=3) # Cache backend status for 3 seconds to avoid excessive API calls
def get_backend_status():
    """Checks the backend status by pinging the /status endpoint."""
    try:
        response = requests.get(f"{API_BASE}/status", timeout=5)
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        data = response.json()
        return data.get("models_loaded", False), None
    except requests.exceptions.ConnectionError:
        return False, "Connection Error: Backend server not running or unreachable."
    except requests.exceptions.Timeout:
        return False, "Timeout Error: Backend server took too long to respond."
    except requests.exceptions.RequestException as e:
        return False, f"An unexpected error occurred: {e}"

def log_to_history(service: str, input_data: str, output_data: str, success: bool = True):
    """
    Logs API calls to the session history.
    Input and output are truncated for display in the history table.
    """
    st.session_state.history.append({
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'service': service,
        'input': input_data[:100] + "..." if len(input_data) > 100 else input_data,
        'output': output_data[:100] + "..." if len(output_data) > 100 else output_data,
        'success': success
    })
    st.session_state.api_calls_count += 1

def display_spinner_and_message(message):
    """Displays a spinner and a message for better UX during processing."""
    with st.spinner(message):
        time.sleep(0.5)  # Brief pause for UX to show spinner

def display_error(error_message):
    """Displays an error message in a styled box."""
    st.error(f"🚨 Error: {error_message}")

def display_success(message):
    """Displays a success message in a styled box."""
    st.success(f"✅ {message}")

def create_download_link(data, filename, text):
    """Creates an HTML download link for given data."""
    b64 = base64.b64encode(data).decode()
    return f'<a href="data:application/octet-stream;base64,{b64}" download="{filename}">{text}</a>'

def add_to_favorites(item_type: str, content: dict):
    """Adds an item to the session favorites list."""
    st.session_state.favorites.append({
        'type': item_type,
        'content': content,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })

# --- AI Service Components ---

def sentiment_analysis_component():
    """Streamlit component for Sentiment Analysis."""
    st.header("🎭 Sentiment Analysis")
    st.write("Determine the emotional tone of text with advanced analytics.")

    text_input = st.text_area(
        "Enter text to analyze sentiment:", 
        height=150, 
        placeholder="Enter your text here...",
        help="Type or paste any text to analyze its emotional sentiment"
    )
        
    if st.button("🔍 Analyze Sentiment", type="primary", use_container_width=True):
        if not text_input.strip():
            st.warning("Please enter text to analyze.")
            return

        with st.spinner("Analyzing sentiment..."):
            try:
                # Call FastAPI backend for sentiment analysis
                response = requests.post(
                    f"{API_BASE}/sentiment/analyze", 
                    json={"text": text_input},
                    timeout=30
                )
                response.raise_for_status()
                result = response.json()
                
                st.subheader("📊 Analysis Results")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    sentiment_class = "sentiment-positive" if result["label"] == "POSITIVE" else "sentiment-negative"
                    st.markdown(f"""
                    <div class="{sentiment_class}">
                        <h3>{result['label']}</h3>
                        <p>{result['score']:.1%} Confidence</p>
                    </div>
                    """, unsafe_allow_html=True)
                with col2:
                    st.metric("Confidence Score", f"{result['score']:.1%}")
                with col3:
                    polarity = "High" if result['score'] > 0.7 else "Moderate" if result['score'] > 0.5 else "Low"
                    st.metric("Polarity Strength", polarity)
                
                # Visualization
                fig = go.Figure(data=[
                    go.Bar(name='Positive', x=['Sentiment'], y=[result['score']], marker_color='green'),
                    go.Bar(name='Negative', x=['Sentiment'], y=[1-result['score']], marker_color='red')
                ])
                fig.update_layout(
                    title="Sentiment Distribution",
                    barmode='stack',
                    height=400,
                    showlegend=True
                )
                st.plotly_chart(fig, use_container_width=True)
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("⭐ Save to Favorites", key="save_sentiment"):
                        add_to_favorites("Sentiment Analysis", {
                            'text': text_input,
                            'result': result
                        })
                        st.success("Saved to favorites!")
                
                log_to_history("Sentiment Analysis", text_input, str(result))
                
            except requests.exceptions.RequestException as e:
                display_error(f"Could not analyze sentiment: {e}")
                log_to_history("Sentiment Analysis", text_input, str(e), False)
            except Exception as e:
                display_error(f"An unexpected error occurred: {e}")
                log_to_history("Sentiment Analysis", text_input, str(e), False)

def text_summarization_component():
    """Streamlit component for Text Summarization."""
    st.header("📄 Text Summarization")
    st.write("Generate concise summaries of your text.")

    text_input = st.text_area(
        "Paste text to summarize:", 
        height=300, 
        placeholder="Paste your long text here...",
        help="Enter a longer text document to get a concise summary"
    )
        
    if st.button("📝 Generate Summary", type="primary", use_container_width=True):
        if not text_input.strip():
            st.warning("Please provide text to summarize.")
            return

        with st.spinner("Generating summary..."):
            try:
                # Call FastAPI backend for summarization
                response = requests.post(
                    f"{API_BASE}/summarization/summarize", 
                    json={"text": text_input},
                    timeout=60
                )
                response.raise_for_status()
                result = response.json()
                summary = result.get("summary_text", "No summary generated.")
                
                st.subheader("📝 Summary")
                st.markdown(f'<div class="content-display">{summary}</div>', unsafe_allow_html=True)
                
                st.subheader("📊 Summary Statistics")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Original Length", f"{len(text_input)} chars")
                with col2:
                    st.metric("Summary Length", f"{len(summary)} chars")
                with col3:
                    compression_ratio = len(summary) / len(text_input) if text_input else 0
                    st.metric("Compression Ratio", f"{compression_ratio:.1%}")
                with col4:
                    reading_time = len(summary.split()) / 200  # Average reading speed
                    st.metric("Reading Time", f"{reading_time:.1f} min")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.download_button(
                        label="📥 Download Summary",
                        data=summary,
                        file_name="summary.txt",
                        mime="text/plain",
                        use_container_width=True
                    )
                with col2:
                    if st.button("⭐ Save to Favorites", key="save_summary"):
                        add_to_favorites("Text Summary", {
                            'original_text': text_input[:200] + "...",
                            'summary': summary,
                        })
                        st.success("Saved to favorites!")
                
                log_to_history("Text Summarization", text_input[:100], summary)
                
            except requests.exceptions.RequestException as e:
                display_error(f"Could not summarize text: {e}")
                log_to_history("Text Summarization", text_input[:100], str(e), False)
            except Exception as e:
                display_error(f"An unexpected error occurred: {e}")
                log_to_history("Text Summarization", text_input[:100], str(e), False)

def text_generation_component():
    """Streamlit component for Creative Text Generation."""
    st.header("✍️ Creative Text Generation")
    st.write("Generate creative content with AI assistance.")

    # Use a simple text area without complex session state management
    text_input = st.text_area(
        "Enter your prompt:", 
        height=150, 
        placeholder="Start your creative prompt here...",
        help="Provide a starting prompt for creative text generation"
    )
        
    if st.button("🚀 Generate Content", type="primary", use_container_width=True):
        if not text_input.strip():
            st.warning("Please enter a prompt.")
            return

        with st.spinner("Generating creative content..."):
            try:
                # Call FastAPI backend for text generation
                response = requests.post(
                    f"{API_BASE}/generation/generate", 
                    json={"text": text_input},
                    timeout=60
                )
                response.raise_for_status()
                result = response.json()
                generated_text = result.get("generated_text", "No text generated.")
                
                st.subheader("📖 Generated Content")
                st.markdown(f'<div class="generated-content">{generated_text}</div>', unsafe_allow_html=True)
                
                st.subheader("📊 Content Analysis")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Words", len(generated_text.split()))
                with col2:
                    st.metric("Characters", len(generated_text))
                with col3:
                    sentences = generated_text.split('.')
                    st.metric("Sentences", len([s for s in sentences if s.strip()]))
                with col4:
                    avg_words = len(generated_text.split()) / len([s for s in sentences if s.strip()]) if sentences else 0
                    st.metric("Avg Words/Sentence", f"{avg_words:.1f}")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.download_button(
                        label="📥 Download Text",
                        data=generated_text,
                        file_name="generated_content.txt",
                        mime="text/plain",
                        use_container_width=True
                    )
                with col2:
                    if st.button("⭐ Save to Favorites", key="save_generation"):
                        add_to_favorites("Generated Text", {
                            'prompt': text_input,
                            'content': generated_text
                        })
                        st.success("Saved to favorites!")
                
                log_to_history("Text Generation", text_input, generated_text)
                
            except requests.exceptions.RequestException as e:
                display_error(f"Could not generate text: {e}")
                log_to_history("Text Generation", text_input, str(e), False)
            except Exception as e:
                display_error(f"An unexpected error occurred: {e}")
                log_to_history("Text Generation", text_input, str(e), False)

def image_captioning_component():
    """Streamlit component for Image Captioning."""
    st.header("🖼️ Image Captioning")
    st.write("Generate descriptive captions for your images.")

    uploaded_file = st.file_uploader(
        "Choose an image...", 
        type=["jpg", "jpeg", "png", "bmp", "tiff"], 
        help="Supported formats: JPG, JPEG, PNG, BMP, TIFF"
    )
        
    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
        
        file_size = len(uploaded_file.read()) / 1024
        uploaded_file.seek(0)
        st.info(f"📄 File: {uploaded_file.name} | Size: {file_size:.1f} KB | Type: {uploaded_file.type}")
        
        if st.button("🔍 Analyze Image", type="primary", use_container_width=True):
            with st.spinner("Analyzing image with AI..."):
                try:
                    # Call FastAPI backend for image captioning
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                    response = requests.post(f"{API_BASE}/image/caption", files=files, timeout=60)
                    response.raise_for_status()
                    result = response.json()
                    caption = result.get("generated_text", "No caption generated.")
                    
                    st.subheader("🔍 Analysis Results")
                    st.markdown(f'<div class="content-display"><h4>📝 Image Caption</h4><p>{caption}</p></div>', unsafe_allow_html=True)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        analysis_report = f"""Image Analysis Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
File: {uploaded_file.name}

Caption: {caption}
"""
                        st.download_button(
                            label="📥 Download Report",
                            data=analysis_report,
                            file_name="image_analysis_report.txt",
                            mime="text/plain",
                            use_container_width=True
                        )
                    with col2:
                        if st.button("⭐ Save to Favorites", key="save_image_analysis"):
                            add_to_favorites("Image Analysis", {
                                'filename': uploaded_file.name,
                                'caption': caption,
                            })
                            st.success("Saved to favorites!")
                    
                    log_to_history("Image Analysis", uploaded_file.name, caption)
                    
                except requests.exceptions.RequestException as e:
                    display_error(f"Could not analyze image: {e}")
                    log_to_history("Image Analysis", uploaded_file.name, str(e), False)
                except Exception as e:
                    display_error(f"An unexpected error occurred: {e}")
                    log_to_history("Image Analysis", uploaded_file.name, str(e), False)

def translation_component():
    """Streamlit component for Language Translation."""
    st.header("🌍 Language Translation")
    st.write("Translate text between English and French.")

    text_input = st.text_area(
        "Enter text to translate (English to French only):", 
        height=200, 
        placeholder="Enter English text here...",
        help="Enter English text to translate to French"
    )
        
    if st.button("🔄 Translate", type="primary", use_container_width=True):
        if not text_input.strip():
            st.warning("Please enter text to translate.")
            return

        with st.spinner("Translating text..."):
            try:
                # Call FastAPI backend for translation
                response = requests.post(
                    f"{API_BASE}/translation/translate", 
                    json={"text": text_input},
                    timeout=30
                )
                response.raise_for_status()
                result = response.json()
                translated_text = result.get("translated_text", "Translation failed.")
                
                st.session_state.translation_history.append({
                    'original': text_input,
                    'translated': translated_text,
                    'source_lang': 'English',
                    'target_lang': 'French',
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })
                
                st.subheader("🔄 Translation Results")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f'<div class="original-text"><h5>📝 Original (English)</h5><p>{text_input}</p></div>', unsafe_allow_html=True)
                with col2:
                    st.markdown(f'<div class="translated-text"><h5>🔄 Translation (French)</h5><p>{translated_text}</p></div>', unsafe_allow_html=True)
                
                st.subheader("📊 Translation Metrics")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Original Length", f"{len(text_input)} chars")
                with col2:
                    st.metric("Translated Length", f"{len(translated_text)} chars")

                col1, col2 = st.columns(2)
                with col1:
                    st.download_button(
                        label="📥 Download Translation",
                        data=f"Original: {text_input}\n\nTranslation: {translated_text}",
                        file_name="translation.txt",
                        mime="text/plain",
                        use_container_width=True
                    )
                with col2:
                    if st.button("⭐ Save to Favorites", key="save_translation_result"):
                        add_to_favorites("Translation", {
                            'original': text_input,
                            'translated': translated_text,
                            'source_lang': 'English',
                            'target_lang': 'French'
                        })
                        st.success("Saved to favorites!")
                
                log_to_history("Language Translation", "English → French", translated_text)
                
            except requests.exceptions.RequestException as e:
                display_error(f"Could not translate text: {e}")
                log_to_history("Language Translation", text_input[:100], str(e), False)
            except Exception as e:
                display_error(f"An unexpected error occurred: {e}")
                log_to_history("Language Translation", text_input[:100], str(e), False)

def speech_to_text_component():
    """Streamlit component for Speech-to-Text."""
    st.header("🎤 Speech to Text")
    st.write("Convert spoken audio into written text.")
    st.warning("This feature uses a self-hosted model, which may require initial download time on the backend.")

    st.subheader("📁 Audio File Upload")
    uploaded_audio = st.file_uploader(
        "Upload an audio file (.wav, .mp3, .flac, .m4a)", 
        type=["wav", "mp3", "flac", "m4a"], 
        help="Supported formats: WAV, MP3, FLAC, M4A"
    )
    
    if uploaded_audio:
        st.audio(uploaded_audio, format=uploaded_audio.type)
        file_size = len(uploaded_audio.read()) / 1024
        st.info(f"📄 File: {uploaded_audio.name} | Size: {file_size:.1f} KB | Type: {uploaded_audio.type}")
        uploaded_audio.seek(0) # Reset file pointer after reading size
    
    if st.button("🔊 Convert to Text", type="primary", use_container_width=True):
        if uploaded_audio is None:
            st.warning("Please upload an audio file to transcribe.")
            return

        with st.spinner("Converting speech to text..."):
            try:
                # Call FastAPI backend for STT
                files = {"file": (uploaded_audio.name, uploaded_audio.getvalue(), uploaded_audio.type)}
                response = requests.post(f"{API_BASE}/stt", files=files, timeout=120)
                response.raise_for_status()
                result = response.json()
                transcription = result.get("transcribed_text", "Could not transcribe audio.")
                
                word_count = len(transcription.split())
                
                st.subheader("📝 Transcription Results")
                st.markdown(f'<div class="transcription-result">{transcription}</div>', unsafe_allow_html=True)
                display_success("Audio transcribed successfully!")

                st.subheader("📊 Transcription Statistics")
                st.metric("Word Count", word_count)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.download_button(
                        label="📥 Download Transcript",
                        data=transcription,
                        file_name="transcript.txt",
                        mime="text/plain",
                        use_container_width=True
                    )
                with col2:
                    if st.button("⭐ Save to Favorites", key="save_stt_result"):
                        add_to_favorites("Speech to Text", {
                            'audio_source': uploaded_audio.name,
                            'transcription': transcription,
                        })
                        st.success("Saved to favorites!")
                
                log_to_history("Speech to Text", uploaded_audio.name, transcription)
                
            except requests.exceptions.RequestException as e:
                display_error(f"Could not transcribe audio: {e}")
                log_to_history("Speech to Text", uploaded_audio.name, str(e), False)
            except Exception as e:
                display_error(f"An unexpected error occurred: {e}")
                log_to_history("Speech to Text", uploaded_audio.name, str(e), False)

def text_to_speech_component():
    """Streamlit component for Text-to-Speech."""
    st.header("🔊 Text to Speech")
    st.write("Convert text to natural-sounding speech.")

    text_input = st.text_area(
        "Enter text to convert to speech:", 
        height=200, 
        value="Hello! This is a sample text for text-to-speech conversion.",
        placeholder="Enter your text here...",
        help="Enter text to convert to speech audio"
    )
        
    word_count = len(text_input.split()) if text_input else 0
    estimated_duration = word_count / 150 * 60  # Average speaking rate (seconds)

    if st.button("🎤 Generate Speech", type="primary", use_container_width=True):
        if not text_input.strip():
            st.warning("Please enter text to convert.")
            return

        with st.spinner("Generating speech..."):
            try:
                # Call FastAPI backend for TTS
                response = requests.post(
                    f"{API_BASE}/tts", 
                    json={"text": text_input},
                    timeout=60
                )
                response.raise_for_status()

                # Check if response is JSON (error) or audio content
                content_type = response.headers.get('content-type', '')
                if 'application/json' in content_type:
                    error_data = response.json()
                    raise Exception(f"TTS API Error: {error_data.get('detail', 'Unknown error')}")
                
                audio_bytes = response.content
                
                if len(audio_bytes) == 0:
                    raise Exception("No audio content received from the server")
                
                st.subheader("🎵 Generated Audio")
                st.audio(audio_bytes, format='audio/wav')
                display_success("Speech generated successfully!")
                
                st.subheader("📊 Audio Information")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Estimated Duration", f"{estimated_duration:.1f}s")
                with col2:
                    st.metric("File Size", f"{len(audio_bytes
