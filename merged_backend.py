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

# --- Configuration ---
API_BASE = "http://localhost:8000/api"

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
</style>
""", unsafe_allow_html=True)

# --- Session State Initialization ---
if 'history' not in st.session_state:
    st.session_state.history = []
if 'favorites' not in st.session_state:
    st.session_state.favorites = []
if 'api_calls_count' not in st.session_state:
    st.session_state.api_calls_count = 0
if 'user_preferences' not in st.session_state:
    st.session_state.user_preferences = {}

# --- Helper Functions ---
@st.cache_data(ttl=3)
def get_backend_status():
    """Checks the backend status."""
    try:
        response = requests.get(f"{API_BASE}/status", timeout=2)
        response.raise_for_status()
        data = response.json()
        return data.get("models_loaded", False), None
    except requests.exceptions.ConnectionError:
        return False, "Connection Error: Backend server not running or unreachable."
    except requests.exceptions.Timeout:
        return False, "Timeout Error: Backend server took too long to respond."
    except requests.exceptions.RequestException as e:
        return False, f"An unexpected error occurred: {e}"

def log_to_history(service: str, input_data: str, output_data: str, success: bool = True):
    """Log API calls to history."""
    st.session_state.history.append({
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'service': service,
        'input': input_data[:100] + "..." if len(input_data) > 100 else input_data,
        'output': output_data[:100] + "..." if len(output_data) > 100 else output_data,
        'success': success
    })
    st.session_state.api_calls_count += 1

def display_spinner_and_message(message):
    """Displays a spinner and a message."""
    st.info(f"⏳ {message}")

def display_error(error_message):
    """Displays an error message."""
    st.error(f"🚨 Error: {error_message}")

def display_success(message):
    """Displays a success message."""
    st.success(f"✅ {message}")

def create_download_link(data, filename, text):
    """Create a download link for data."""
    b64 = base64.b64encode(data).decode()
    return f'<a href="data:application/octet-stream;base64,{b64}" download="{filename}">{text}</a>'

# --- Enhanced AI Service Components ---

def sentiment_analysis_component():
    st.header("🎭 Sentiment Analysis")
    st.write("Determine the emotional tone of text with advanced analytics.")

    # Advanced options
    col1, col2 = st.columns([3, 1])
    
    with col1:
        text_input = st.text_area("Enter text to analyze sentiment:", height=150, key="sentiment_text_input")
    
    with col2:
        st.subheader("Options")
        detailed_analysis = st.checkbox("Detailed Analysis", value=False)
        emotion_detection = st.checkbox("Emotion Detection", value=False)
        
    # Batch analysis
    st.subheader("Batch Analysis")
    uploaded_file = st.file_uploader("Upload CSV for batch sentiment analysis", type=['csv'], key="sentiment_batch")
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("Preview:", df.head())
        text_column = st.selectbox("Select text column:", df.columns)

    if st.button("Analyze Sentiment", key="sentiment_button"):
        if not text_input.strip() and not uploaded_file:
            st.warning("Please enter text or upload a file to analyze.")
            return

        if text_input.strip():
            display_spinner_and_message("Analyzing sentiment...")
            try:
                # Mock API call since we don't have the actual backend
                result = {
                    "label": "POSITIVE",
                    "score": 0.89,
                    "emotions": {"joy": 0.7, "confidence": 0.2, "neutral": 0.1} if emotion_detection else None
                }
                
                st.subheader("📊 Analysis Results")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Sentiment", result["label"], f"{result['score']:.1%}")
                with col2:
                    st.metric("Confidence", f"{result['score']:.1%}")
                with col3:
                    st.metric("Polarity", "High" if result['score'] > 0.7 else "Moderate")
                
                # Visualization
                fig = go.Figure(go.Bar(
                    x=['Positive', 'Negative'],
                    y=[result['score'], 1-result['score']],
                    marker_color=['green', 'red']
                ))
                fig.update_layout(title="Sentiment Distribution", height=300)
                st.plotly_chart(fig, use_container_width=True)
                
                if emotion_detection and result.get("emotions"):
                    st.subheader("🎨 Emotion Breakdown")
                    emotions_df = pd.DataFrame(list(result["emotions"].items()), columns=['Emotion', 'Score'])
                    fig_emotions = px.bar(emotions_df, x='Emotion', y='Score', color='Score')
                    st.plotly_chart(fig_emotions, use_container_width=True)
                
                log_to_history("Sentiment Analysis", text_input, str(result))
                
            except Exception as e:
                display_error(f"Could not analyze sentiment: {e}")
                log_to_history("Sentiment Analysis", text_input, str(e), False)

def text_summarization_component():
    st.header("📄 Text Summarization")
    st.write("Generate concise summaries with customizable options.")

    col1, col2 = st.columns([3, 1])
    
    with col1:
        text_input = st.text_area("Paste text to summarize:", height=300, key="summarization_text_input")
    
    with col2:
        st.subheader("Summary Options")
        summary_length = st.selectbox("Length:", ["Short", "Medium", "Long"])
        summary_type = st.selectbox("Type:", ["Abstractive", "Extractive"])
        include_keywords = st.checkbox("Include Keywords")
        
    # Document upload
    st.subheader("📁 Document Upload")
    uploaded_doc = st.file_uploader("Upload document", type=['txt', 'pdf', 'docx'])
    
    if st.button("Generate Summary", key="summarization_button"):
        if not text_input.strip() and not uploaded_doc:
            st.warning("Please provide text or upload a document.")
            return

        display_spinner_and_message("Generating summary...")
        try:
            # Mock summary generation
            summary = "This is a comprehensive summary of the provided text. The main points have been extracted and condensed into this readable format while maintaining the core message and important details."
            keywords = ["AI", "technology", "innovation", "future"] if include_keywords else []
            
            st.subheader("📝 Summary")
            st.info(summary)
            
            if keywords:
                st.subheader("🔑 Key Terms")
                for keyword in keywords:
                    st.badge(keyword)
            
            # Summary statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Original Length", f"{len(text_input)} chars")
            with col2:
                st.metric("Summary Length", f"{len(summary)} chars")
            with col3:
                compression_ratio = len(summary) / len(text_input) if text_input else 0
                st.metric("Compression", f"{compression_ratio:.1%}")
            
            # Download option
            st.download_button(
                label="Download Summary",
                data=summary,
                file_name="summary.txt",
                mime="text/plain"
            )
            
            log_to_history("Text Summarization", text_input[:100], summary)
            
        except Exception as e:
            display_error(f"Could not generate summary: {e}")

def text_generation_component():
    st.header("✍️ Creative Text Generation")
    st.write("Generate creative content with AI assistance.")

    col1, col2 = st.columns([2, 1])
    
    with col1:
        text_input = st.text_area("Enter your prompt:", value="Once upon a time,", height=150, key="generation_text_input")
    
    with col2:
        st.subheader("Generation Settings")
        creativity_level = st.slider("Creativity Level", 0.1, 2.0, 1.0, 0.1)
        max_length = st.slider("Max Length", 50, 500, 200)
        writing_style = st.selectbox("Style:", ["Creative", "Technical", "Casual", "Formal"])
        
    # Template prompts
    st.subheader("📋 Quick Templates")
    templates = {
        "Story": "Write a short story about",
        "Email": "Compose a professional email regarding",
        "Blog Post": "Create a blog post about",
        "Product Description": "Write a compelling product description for"
    }
    
    selected_template = st.selectbox("Choose template:", list(templates.keys()))
    if st.button("Use Template"):
        st.session_state.generation_text_input = templates[selected_template]
        st.rerun()

    if st.button("Generate Text", key="generation_button"):
        if not text_input.strip():
            st.warning("Please enter a prompt.")
            return

        display_spinner_and_message("Generating creative text...")
        try:
            # Mock text generation
            generated_text = f"Based on your prompt '{text_input}', here's a creative continuation: The story unfolds in ways that capture the imagination, weaving together elements of mystery, adventure, and human emotion. Each character brings their own unique perspective to the narrative, creating a rich tapestry of experiences that resonate with readers across different backgrounds and interests."
            
            st.subheader("📖 Generated Content")
            st.write(generated_text)
            
            # Content analysis
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Words", len(generated_text.split()))
            with col2:
                st.metric("Characters", len(generated_text))
            with col3:
                st.metric("Readability", "Good")
            
            # Save to favorites
            if st.button("⭐ Save to Favorites"):
                st.session_state.favorites.append({
                    'type': 'Generated Text',
                    'prompt': text_input,
                    'content': generated_text,
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })
                st.success("Saved to favorites!")
            
            log_to_history("Text Generation", text_input, generated_text)
            
        except Exception as e:
            display_error(f"Could not generate text: {e}")

def image_captioning_component():
    st.header("🖼️ Advanced Image Analysis")
    st.write("Analyze images with AI-powered captioning and object detection.")

    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], key="image_upload")
    
    with col2:
        st.subheader("Analysis Options")
        detailed_caption = st.checkbox("Detailed Description")
        object_detection = st.checkbox("Object Detection")
        scene_analysis = st.checkbox("Scene Analysis")
        text_extraction = st.checkbox("Text Extraction (OCR)")

    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
        
        if st.button("Analyze Image", key="captioning_button"):
            display_spinner_and_message("Analyzing image...")
            try:
                # Mock image analysis
                caption = "A beautiful landscape showing mountains in the background with a clear blue sky and green vegetation in the foreground."
                objects = ["mountain", "sky", "tree", "grass"] if object_detection else []
                scene = "Outdoor natural landscape" if scene_analysis else ""
                extracted_text = "No text detected" if text_extraction else ""
                
                st.subheader("🔍 Analysis Results")
                
                # Main caption
                st.write("**Caption:**", caption)
                
                # Additional analysis
                if objects:
                    st.subheader("🎯 Detected Objects")
                    for obj in objects:
                        st.badge(obj)
                
                if scene:
                    st.subheader("🌄 Scene Analysis")
                    st.info(scene)
                
                if text_extraction:
                    st.subheader("📝 Extracted Text")
                    st.code(extracted_text)
                
                # Confidence metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Caption Confidence", "94%")
                with col2:
                    st.metric("Objects Detected", len(objects))
                with col3:
                    st.metric("Processing Time", "2.3s")
                
                log_to_history("Image Analysis", uploaded_file.name, caption)
                
            except Exception as e:
                display_error(f"Could not analyze image: {e}")
    else:
        st.info("👆 Upload an image file to get started with AI-powered analysis.")

def text_to_speech_component():
    st.header("🎤 Text-to-Speech")
    st.write("Convert text to natural speech with voice customization.")

    col1, col2 = st.columns([2, 1])
    
    with col1:
        text_input = st.text_area("Enter text to convert to speech:", height=150, key="tts_text_input")
    
    with col2:
        st.subheader("Voice Settings")
        voice_type = st.selectbox("Voice:", ["Male", "Female", "Child"])
        speed = st.slider("Speed", 0.5, 2.0, 1.0, 0.1)
        pitch = st.slider("Pitch", 0.5, 2.0, 1.0, 0.1)
        language = st.selectbox("Language:", ["English", "Spanish", "French", "German"])

    if st.button("Generate Speech", key="tts_button"):
        if not text_input.strip():
            st.warning("Please enter text for speech generation.")
            return

        display_spinner_and_message("Generating speech...")
        try:
            # Mock audio generation (you would replace this with actual API call)
            st.subheader("🔊 Generated Audio")
            st.success("Speech generated successfully!")
            st.info("In a real implementation, audio would be generated here.")
            
            # Audio analysis
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Duration", "15.2s")
            with col2:
                st.metric("File Size", "240 KB")
            with col3:
                st.metric("Quality", "High")
            
            log_to_history("Text-to-Speech", text_input, "Audio generated")
            
        except Exception as e:
            display_error(f"Could not generate speech: {e}")

def speech_to_text_component():
    st.header("🎙️ Speech-to-Text")
    st.write("Upload audio files for accurate transcription.")

    # File upload instead of recording
    uploaded_audio = st.file_uploader("Upload audio file", type=['wav', 'mp3', 'mp4', 'ogg'], key="stt_upload")
    
    col1, col2 = st.columns([2, 1])
    
    with col2:
        st.subheader("Transcription Options")
        language = st.selectbox("Language:", ["Auto-detect", "English", "Spanish", "French"])
        speaker_detection = st.checkbox("Speaker Detection")
        timestamps = st.checkbox("Include Timestamps")
        punctuation = st.checkbox("Smart Punctuation", value=True)

    if uploaded_audio is not None:
        st.audio(uploaded_audio, format="audio/wav")
        
        if st.button("Transcribe Audio", key="stt_button"):
            display_spinner_and_message("Transcribing audio...")
            try:
                # Mock transcription
                transcribed_text = "Hello, this is a sample transcription of the uploaded audio file. The speech-to-text system has processed the audio and converted it into readable text format."
                
                st.subheader("📝 Transcription Results")
                st.write(transcribed_text)
                
                if timestamps:
                    st.subheader("⏰ Timestamps")
                    st.code("[00:00] Hello, this is a sample\n[00:03] transcription of the uploaded\n[00:06] audio file.")
                
                if speaker_detection:
                    st.subheader("👥 Speaker Detection")
                    st.info("Speaker 1: Main speaker detected")
                
                # Transcription metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Words", len(transcribed_text.split()))
                with col2:
                    st.metric("Confidence", "96%")
                with col3:
                    st.metric("Duration", "12.5s")
                
                # Download transcription
                st.download_button(
                    label="Download Transcription",
                    data=transcribed_text,
                    file_name="transcription.txt",
                    mime="text/plain"
                )
                
                log_to_history("Speech-to-Text", uploaded_audio.name, transcribed_text)
                
            except Exception as e:
                display_error(f"Could not transcribe audio: {e}")
    else:
        st.info("👆 Upload an audio file to get started with transcription.")

def translation_component():
    st.header("🌍 Language Translation")
    st.write("Translate text between multiple languages with high accuracy.")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        text_to_translate = st.text_area("Enter text to translate:", height=150, key="translation_input")
    
    with col2:
        st.subheader("Translation Settings")
        source_lang = st.selectbox("From:", ["Auto-detect", "English", "Spanish", "French", "German", "Chinese", "Japanese"])
        target_lang = st.selectbox("To:", ["English", "Spanish", "French", "German", "Chinese", "Japanese"])
        formal_tone = st.checkbox("Formal tone")
    
    if st.button("Translate", key="translate_button"):
        if not text_to_translate.strip():
            st.warning("Please enter text to translate.")
            return
        
        display_spinner_and_message("Translating text...")
        try:
            # Mock translation
            translated_text = f"[Translated from {source_lang} to {target_lang}]: This is the translated version of your input text."
            
            st.subheader("🔄 Translation Result")
            st.success(translated_text)
            
            # Translation confidence
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Confidence", "98%")
            with col2:
                st.metric("Source Language", source_lang)
            with col3:
                st.metric("Target Language", target_lang)
            
            log_to_history("Translation", text_to_translate, translated_text)
            
        except Exception as e:
            display_error(f"Could not translate text: {e}")

def analytics_dashboard():
    st.header("📊 Analytics Dashboard")
    st.write("Track your AI toolkit usage and insights.")
    
    # Usage statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total API Calls", st.session_state.api_calls_count)
    with col2:
        st.metric("Successful Operations", len([h for h in st.session_state.history if h['success']]))
    with col3:
        st.metric("Services Used", len(set(h['service'] for h in st.session_state.history)))
    with col4:
        st.metric("Favorites", len(st.session_state.favorites))
    
    if st.session_state.history:
        # Usage over time
        df_history = pd.DataFrame(st.session_state.history)
        df_history['timestamp'] = pd.to_datetime(df_history['timestamp'])
        
        # Service usage chart
        service_counts = df_history['service'].value_counts()
        fig_services = px.bar(
            x=service_counts.index, 
            y=service_counts.values,
            title="Service Usage Distribution",
            labels={'x': 'Service', 'y': 'Count'}
        )
        st.plotly_chart(fig_services, use_container_width=True)
        
        # Success rate
        success_rate = df_history['success'].mean() * 100
        st.subheader(f"Success Rate: {success_rate:.1f}%")
        
        # Recent history
        st.subheader("Recent Activity")
        st.dataframe(df_history.tail(10)[['timestamp', 'service', 'success']], use_container_width=True)
    
    else:
        st.info("No usage data available yet. Start using the AI services to see analytics!")

def settings_component():
    st.header("⚙️ Settings & Preferences")
    st.write("Customize your AI toolkit experience.")
    
    # Theme settings
    st.subheader("🎨 Appearance")
    theme = st.selectbox("Theme:", ["Light", "Dark", "Auto"])
    
    # API settings
    st.subheader("🔧 API Configuration")
    api_base = st.text_input("API Base URL:", value=API_BASE)
    timeout = st.slider("Request Timeout (seconds):", 1, 30, 5)
    
    # Default settings
    st.subheader("📋 Default Settings")
    default_voice = st.selectbox("Default TTS Voice:", ["Male", "Female", "Child"])
    default_language = st.selectbox("Default Language:", ["English", "Spanish", "French", "German"])
    auto_save = st.checkbox("Auto-save results", value=True)
    
    # Data management
    st.subheader("💾 Data Management")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Clear History"):
            st.session_state.history = []
            st.success("History cleared!")
    
    with col2:
        if st.button("Clear Favorites"):
            st.session_state.favorites = []
            st.success("Favorites cleared!")
    
    # Export data
    if st.button("Export Usage Data"):
        if st.session_state.history:
            df = pd.DataFrame(st.session_state.history)
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name="ai_toolkit_usage.csv",
                mime="text/csv"
            )
        else:
            st.warning("No data to export.")

# --- Main Application Layout ---

# Sidebar
st.sidebar.title("🤖 AI Toolkit Pro")
st.sidebar.write("Professional AI Services Suite")

# Backend Status
models_loaded, status_error = get_backend_status()
if models_loaded:
    st.sidebar.success("Backend: ● Online")
else:
    if status_error:
        st.sidebar.error(f"Backend: ● {status_error}")
    else:
        st.sidebar.warning("Backend: ● Offline")

# Navigation
st.sidebar.markdown("---")
st.sidebar.subheader("Navigation")

# Main title
st.markdown('<div class="main-header"><h1>🚀 AI Services Toolkit Pro</h1><p>Your Complete AI-Powered Solution</p></div>', unsafe_allow_html=True)

# Enhanced tabs with new features
tab_sentiment, tab_summarization, tab_generation, tab_captioning, tab_tts, tab_stt, tab_translation, tab_analytics, tab_settings = st.tabs([
    "🎭 Sentiment", "📄 Summary", "✍️ Generate", "🖼️ Images", 
    "🎤 TTS", "🎙️ STT", "🌍 Translate", "📊 Analytics", "⚙️ Settings"
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

with tab_tts:
    text_to_speech_component()

with tab_stt:
    speech_to_text_component()

with tab_translation:
    translation_component()

with tab_analytics:
    analytics_dashboard()

with tab_settings:
    settings_component()

# Footer
st.markdown("---")
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("**🔧 Built with FastAPI & Streamlit**")
with col2:
    st.markdown(f"**📊 API Calls: {st.session_state.api_calls_count}**")
with col3:
    st.markdown(f"**⭐ Favorites: {len(st.session_state.favorites)}**")
