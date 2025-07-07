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

    /* Adjust color for generated text and translated text outputs for better visibility */
    /* This targets the specific markdown divs with light backgrounds */
    div[data-testid="stMarkdown"] > div > div[style*="background: #f8f9fa;"],
    div[data-testid="stMarkdown"] > div > div[style*="background: #e3f2fd;"],
    div[data-testid="stMarkdown"] > div > div[style*="background: #e8f5e8;"],
    div[data-testid="stMarkdown"] > div > div[style*="background: #f0f8ff;"] {
        color: #333333; /* Dark grey for better contrast on light backgrounds */
    }
    /* Specific adjustment for sentiment analysis text within the colored box */
    div[data-testid="stMarkdown"] > div > div[style*="background: green;"],
    div[data-testid="stMarkdown"] > div > div[style*="background: red;"] {
        color: white; /* Keep white text for sentiment analysis result for consistency with gradient cards */
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
# Inputs for components that use st.rerun to persist values
if 'translation_text_input' not in st.session_state:
    st.session_state.translation_text_input = ""
if 'translation_source' not in st.session_state:
    st.session_state.translation_source = "Auto-detect"
if 'translation_target' not in st.session_state:
    st.session_state.translation_target = "English"
if 'generation_text_input' not in st.session_state:
    st.session_state.generation_text_input = "Once upon a time,"
if 'qa_context_input' not in st.session_state:
    st.session_state.qa_context_input = ""
if 'qa_question_input' not in st.session_state:
    st.session_state.qa_question_input = ""
if 'chat_input' not in st.session_state:
    st.session_state.chat_input = ""


# --- Helper Functions ---
@st.cache_data(ttl=3) # Cache backend status for 3 seconds to avoid excessive API calls
def get_backend_status():
    """Checks the backend status by pinging the /status endpoint."""
    try:
        response = requests.get(f"{API_BASE}/status", timeout=2)
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
        key="sentiment_text_input",
        placeholder="Enter your text here..."
    )
        
    if st.button("🔍 Analyze Sentiment", key="sentiment_button", use_container_width=True):
        if not text_input.strip():
            st.warning("Please enter text to analyze.")
            return

        display_spinner_and_message("Analyzing sentiment...")
        try:
            # Call FastAPI backend for sentiment analysis
            response = requests.post(f"{API_BASE}/sentiment/analyze", json={"text": text_input})
            response.raise_for_status()
            result = response.json()
            
            st.subheader("📊 Analysis Results")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                sentiment_color = "green" if result["label"] == "POSITIVE" else "red"
                st.markdown(f"<div style='text-align: center; padding: 1rem; background: {sentiment_color}; color: white; border-radius: 10px;'>"
                            f"<h3>{result['label']}</h3><p>{result['score']:.1%} Confidence</p></div>", 
                            unsafe_allow_html=True)
            with col2:
                st.metric("Confidence Score", f"{result['score']:.1%}")
            with col3:
                polarity = "High" if result['score'] > 0.7 else "Moderate" if result['score'] > 0.5 else "Low"
                st.metric("Polarity Strength", polarity)
            
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
        key="summarization_text_input",
        placeholder="Paste your long text here..."
    )
        
    if st.button("📝 Generate Summary", key="summarization_button", use_container_width=True):
        if not text_input.strip():
            st.warning("Please provide text to summarize.")
            return

        display_spinner_and_message("Generating summary...")
        try:
            processed_text = text_input
            
            # Call FastAPI backend for summarization
            response = requests.post(f"{API_BASE}/summarization/summarize", json={"text": processed_text})
            response.raise_for_status()
            result = response.json()
            summary = result.get("summary_text", "No summary generated.")
            
            st.subheader("📝 Summary")
            st.info(summary)
            
            st.subheader("📊 Summary Statistics")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Original Length", f"{len(processed_text)} chars")
            with col2:
                st.metric("Summary Length", f"{len(summary)} chars")
            with col3:
                compression_ratio = len(summary) / len(processed_text) if processed_text else 0
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
                        'original_text': processed_text[:200] + "...",
                        'summary': summary,
                    })
                    st.success("Saved to favorites!")
            
            log_to_history("Text Summarization", processed_text[:100], summary)
            
        except requests.exceptions.RequestException as e:
            display_error(f"Could not summarize text: {e}")
            log_to_history("Text Summarization", processed_text[:100], str(e), False)
        except Exception as e:
            display_error(f"An unexpected error occurred: {e}")
            log_to_history("Text Summarization", processed_text[:100], str(e), False)

def text_generation_component():
    """Streamlit component for Creative Text Generation."""
    st.header("✍️ Creative Text Generation")
    st.write("Generate creative content with AI assistance.")

    text_input = st.text_area(
        "Enter your prompt:", 
        value=st.session_state.get('generation_text_input', "Once upon a time,"), 
        height=150, 
        key="generation_text_input",
        placeholder="Start your creative prompt here..."
    )
        
    if st.button("🚀 Generate Content", key="generation_button", use_container_width=True):
        if not text_input.strip():
            st.warning("Please enter a prompt.")
            return

        display_spinner_and_message("Generating creative content...")
        try:
            # Call FastAPI backend for text generation
            response = requests.post(f"{API_BASE}/generation/generate", json={"text": text_input})
            response.raise_for_status()
            result = response.json()
            generated_text = result.get("generated_text", "No text generated.")
            
            st.subheader("📖 Generated Content")
            st.markdown(f"<div style='background: #f8f9fa; padding: 1.5rem; border-radius: 10px; border-left: 4px solid #667eea;'>{generated_text}</div>", unsafe_allow_html=True)
            
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
        key="image_upload",
        help="Supported formats: JPG, JPEG, PNG, BMP, TIFF"
    )
        
    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
        
        file_size = len(uploaded_file.read()) / 1024
        uploaded_file.seek(0)
        st.info(f"📄 File: {uploaded_file.name} | Size: {file_size:.1f} KB | Type: {uploaded_file.type}")
        
        if st.button("🔍 Analyze Image", key="captioning_button", use_container_width=True):
            display_spinner_and_message("Analyzing image with AI...")
            try:
                # Call FastAPI backend for image captioning
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                response = requests.post(f"{API_BASE}/image/caption", files=files)
                response.raise_for_status()
                result = response.json()
                caption = result.get("generated_text", "No caption generated.")
                
                st.subheader("🔍 Analysis Results")
                
                st.markdown(f"<div style='background: #e3f2fd; padding: 1rem; border-radius: 10px; border-left: 4px solid #2196f3;'>"
                            f"<h4>📝 Image Caption</h4><p>{caption}</p></div>", unsafe_allow_html=True)
                
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
        key="translation_text_input",
        placeholder="Enter English text here...",
        value=st.session_state.translation_text_input # Use session state for persistence
    )
        
    if st.button("🔄 Translate", key="translation_button", use_container_width=True):
        if not text_input.strip():
            st.warning("Please enter text to translate.")
            return

        display_spinner_and_message("Translating text...")
        try:
            original_text_to_translate = text_input
            
            # Call FastAPI backend for translation
            response = requests.post(f"{API_BASE}/translation/translate", json={"text": original_text_to_translate})
            response.raise_for_status()
            translated_text = response.json().get("translated_text", "Translation failed.")
            
            st.session_state.translation_history.append({
                'original': original_text_to_translate,
                'translated': translated_text,
                'source_lang': 'English',
                'target_lang': 'French',
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
            
            st.subheader("🔄 Translation Results")
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"<div style='background: #f8f9fa; padding: 1rem; border-radius: 10px; border-left: 4px solid #dc3545;'>"
                            f"<h5>📝 Original (English)</h5><p>{original_text_to_translate}</p></div>", unsafe_allow_html=True)
            with col2:
                st.markdown(f"<div style='background: #f8f9fa; padding: 1rem; border-radius: 10px; border-left: 4px solid #28a745;'>"
                            f"<h5>🔄 Translation (French)</h5><p>{translated_text}</p></div>", unsafe_allow_html=True)
            
            st.subheader("📊 Translation Metrics")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Original Length", f"{len(original_text_to_translate)} chars")
            with col2:
                st.metric("Translated Length", f"{len(translated_text)} chars")

            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    label="📥 Download Translation",
                    data=f"Original: {original_text_to_translate}\n\nTranslation: {translated_text}",
                    file_name="translation.txt",
                    mime="text/plain",
                    use_container_width=True
                )
            with col2:
                if st.button("⭐ Save to Favorites", key="save_translation_result"):
                    add_to_favorites("Translation", {
                        'original': original_text_to_translate,
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
        key="stt_audio_upload",
        help="Supported formats: WAV, MP3, FLAC, M4A"
    )
    
    if uploaded_audio:
        st.audio(uploaded_audio, format=uploaded_audio.type)
        file_size = len(uploaded_audio.read()) / 1024
        st.info(f"📄 File: {uploaded_audio.name} | Size: {file_size:.1f} KB | Type: {uploaded_audio.type}")
        uploaded_audio.seek(0) # Reset file pointer after reading size
    
    source_audio_data = None
    source_filename = "audio_input.wav"
    source_mime = "audio/wav"

    if uploaded_audio:
        source_audio_data = uploaded_audio.getvalue()
        source_filename = uploaded_audio.name
        source_mime = uploaded_audio.type

    if st.button("🔊 Convert to Text", key="stt_convert_button", use_container_width=True):
        if source_audio_data is None:
            st.warning("Please upload an audio file to transcribe.")
            return

        display_spinner_and_message("Converting speech to text...")
        try:
            # Call FastAPI backend for STT
            files = {"file": (source_filename, source_audio_data, source_mime)}
            response = requests.post(f"{API_BASE}/stt", files=files)
            response.raise_for_status()
            result = response.json()
            transcription = result.get("transcribed_text", "Could not transcribe audio.")
            
            word_count = len(transcription.split())
            
            st.subheader("📝 Transcription Results")
            st.markdown(f"<div style='background: #f0f8ff; padding: 1.5rem; border-radius: 10px; border-left: 4px solid #007bff;'>{transcription}</div>", unsafe_allow_html=True)
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
                        'audio_source': source_filename,
                        'transcription': transcription,
                    })
                    st.success("Saved to favorites!")
            
            log_to_history("Speech to Text", source_filename, transcription)
            
        except requests.exceptions.RequestException as e:
            display_error(f"Could not transcribe audio: {e}")
            log_to_history("Speech to Text", source_filename, str(e), False)
        except Exception as e:
            display_error(f"An unexpected error occurred: {e}")
            log_to_history("Speech to Text", source_filename, str(e), False)

def text_to_speech_component():
    """Streamlit component for Text-to-Speech."""
    st.header("🔊 Text to Speech")
    st.write("Convert text to natural-sounding speech.")

    text_input = st.text_area(
        "Enter text to convert to speech:", 
        height=200, 
        key="tts_text_input",
        value="Hello! This is a sample text for text-to-speech conversion.",
        placeholder="Enter your text here..."
    )
        
    word_count = len(text_input.split()) if text_input else 0
    estimated_duration = word_count / 150 * 60  # Average speaking rate (seconds)

    if st.button("🎤 Generate Speech", key="tts_generate_button", use_container_width=True):
        if not text_input.strip():
            st.warning("Please enter text to convert.")
            return

        display_spinner_and_message("Generating speech...")
        try:
            # Call FastAPI backend for TTS
            response = requests.post(f"{API_BASE}/tts", json={"text": text_input})
            response.raise_for_status()

            audio_bytes = response.content
            
            st.subheader("🎵 Generated Audio")
            st.audio(audio_bytes, format='audio/wav')
            display_success("Speech generated successfully!")
            
            st.subheader("📊 Audio Information")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Estimated Duration", f"{estimated_duration:.1f}s")
            with col2:
                st.metric("File Size", f"{len(audio_bytes) / (1024*1024):.2f} MB") 
            
            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    label="📥 Download Audio",
                    data=audio_bytes,
                    file_name="generated_speech.wav",
                    mime="audio/wav",
                    use_container_width=True
                )
            with col2:
                if st.button("⭐ Save to Favorites", key="save_tts_result"):
                    add_to_favorites("Text to Speech", {
                        'text': text_input[:100] + "...",
                    })
                    st.success("Saved to favorites!")
            
            log_to_history("Text to Speech", text_input[:100], f"Generated speech")
            
        except requests.exceptions.RequestException as e:
            display_error(f"Could not generate speech: {e}")
            log_to_history("Text to Speech", text_input[:100], str(e), False)
        except Exception as e:
            display_error(f"An unexpected error occurred: {e}")
            log_to_history("Text to Speech", text_input[:100], str(e), False)

def history_and_analytics_component():
    """Streamlit component for displaying History and Analytics."""
    st.header("📊 History & Analytics")
    st.write("View your usage history and analytics across all AI services.")

    st.subheader("📈 Usage Statistics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total API Calls", st.session_state.api_calls_count)
    with col2:
        successful_calls = len([h for h in st.session_state.history if h['success']])
        st.metric("Successful Calls", successful_calls)
    with col3:
        services_used = len(set(h['service'] for h in st.session_state.history))
        st.metric("Services Used", services_used)
    with col4:
        favorites_count = len(st.session_state.favorites)
        st.metric("Favorites", favorites_count)
    
    if st.session_state.history:
        st.subheader("📊 Service Usage Distribution")
        service_counts = {}
        for entry in st.session_state.history:
            service = entry['service']
            service_counts[service] = service_counts.get(service, 0) + 1
        
        fig = px.pie(
            values=list(service_counts.values()),
            names=list(service_counts.keys()),
            title="API Calls by Service"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("📅 Usage Timeline")
        timeline_data = []
        for entry in st.session_state.history:
            timeline_data.append({
                'timestamp': entry['timestamp'],
                'service': entry['service'],
                'success': entry['success']
            })
        
        if timeline_data:
            timeline_df = pd.DataFrame(timeline_data)
            timeline_df['timestamp'] = pd.to_datetime(timeline_df['timestamp'])
            
            fig = px.scatter(
                timeline_df,
                x='timestamp',
                y='service',
                color='success',
                title="API Calls Timeline",
                color_discrete_map={True: 'green', False: 'red'}
            )
            st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("📋 Call History")
    if st.session_state.history:
        history_df = pd.DataFrame(st.session_state.history)
        
        col1, col2 = st.columns(2)
        with col1:
            service_filter = st.selectbox(
                "Filter by service:", 
                ["All"] + list(set(h['service'] for h in st.session_state.history)),
                key="history_service_filter"
            )
        with col2:
            status_filter = st.selectbox("Filter by status:", ["All", "Success", "Failed"], key="history_status_filter")
        
        filtered_history = st.session_state.history.copy()
        if service_filter != "All":
            filtered_history = [h for h in filtered_history if h['service'] == service_filter]
        if status_filter != "All":
            success_value = status_filter == "Success"
            filtered_history = [h for h in filtered_history if h['success'] == success_value]
        
        if filtered_history:
            st.dataframe(pd.DataFrame(filtered_history), use_container_width=True)
            
            col1, col2 = st.columns(2)
            with col1:
                csv_data = pd.DataFrame(filtered_history).to_csv(index=False)
                st.download_button(
                    label="📥 Download History (CSV)",
                    data=csv_data,
                    file_name="ai_services_history.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            with col2:
                if st.button("🗑️ Clear History", use_container_width=True, key="clear_history_button"):
                    st.session_state.history = []
                    st.session_state.api_calls_count = 0
                    st.success("History cleared!")
                    st.rerun()
        else:
            st.info("No history matches the selected filters.")
    else:
        st.info("No history available yet. Start using the AI services to see your history here.")

def favorites_component():
    """Streamlit component for displaying Favorites."""
    st.header("⭐ Favorites")
    st.write("View and manage your saved favorite results.")

    if st.session_state.favorites:
        for i, item in enumerate(st.session_state.favorites):
            st.subheader(f"{item['type']} - {item['timestamp']}")
            st.json(item['content'])
            if st.button(f"Remove from Favorites {i}", key=f"remove_fav_{i}"):
                st.session_state.favorites.pop(i)
                st.rerun()
    else:
        st.info("No favorites saved yet.")

# --- Main Application Layout ---

st.sidebar.title("🛠️ AI Services Toolkit Pro")
st.sidebar.write("A suite of powerful, self-hosted AI models for advanced tasks.")

# Backend Status in Sidebar
models_loaded, status_error = get_backend_status()
if models_loaded:
    st.sidebar.success("Backend Status: ● Ready")
else:
    if status_error:
        st.sidebar.error(f"Backend Status: ● Error ({status_error})")
    else:
        st.sidebar.warning("Backend Status: ● Loading Models...")
        st.sidebar.info("Please ensure the backend server is running and models are loaded.")
        st.sidebar.markdown("Run the backend with: `python merged_backend.py`")

st.markdown('<div class="main-header"><h1>🤖 AI Services Toolkit Pro Dashboard</h1></div>', unsafe_allow_html=True)

# Use st.tabs for navigation
tab_sentiment, tab_summarization, tab_generation, tab_captioning, tab_translation, tab_tts, tab_stt, tab_history, tab_favorites, tab_settings = st.tabs([
    "Sentiment Analysis", "Text Summarization", "Text Generation",
    "Image Captioning", "Translation", "Text-to-Speech", "Speech-to-Text",
    "History", "Favorites", "Settings"
])

# Render components in their respective tabs
with tab_sentiment:
    sentiment_analysis_component()

with tab_summarization:
    text_summarization_component()

with tab_generation:
    text_generation_component()

with tab_captioning:
    image_captioning_component()

with tab_translation:
    translation_component()

with tab_tts:
    text_to_speech_component()

with tab_stt:
    speech_to_text_component()

with tab_history:
    history_and_analytics_component()

with tab_favorites:
    favorites_component()

with tab_settings:
    st.header("⚙️ User Settings")
    st.write("Customize your AI Toolkit experience.")
    
    st.subheader("General Preferences")
    st.session_state.user_preferences['theme'] = st.selectbox(
        "App Theme:", ["Light", "Dark"], 
        index=["Light", "Dark"].index(st.session_state.user_preferences['theme']),
        key="settings_theme_select"
    )
    st.session_state.user_preferences['auto_save'] = st.checkbox(
        "Auto-save results to history", 
        value=st.session_state.user_preferences['auto_save'],
        key="settings_auto_save_checkbox"
    )

    st.subheader("API Configuration (Advanced)")
    current_api_base_input = API_BASE
    new_api_base_from_input = st.text_input("Backend API Base URL:", value=current_api_base_input, key="settings_api_base_input")
    
    if new_api_base_from_input != API_BASE:
        st.warning(f"API Base URL changed from {API_BASE} to {new_api_base_from_input}. This change will apply on next rerun.")
        globals()['API_BASE'] = new_api_base_from_input
        if st.button("Apply API Change and Rerun", key="apply_api_change"):
            st.rerun()

st.markdown("---")
st.markdown("Developed with FastAPI and Streamlit.")
