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

    col1, col2 = st.columns([3, 1])
    
    with col1:
        text_input = st.text_area(
            "Enter text to analyze sentiment:", 
            height=150, 
            key="sentiment_text_input",
            placeholder="Enter your text here..."
        )
    
    with col2:
        st.subheader("Analysis Options")
        detailed_analysis = st.checkbox("Detailed Analysis", value=False)
        emotion_detection = st.checkbox("Emotion Detection", value=False)
        confidence_threshold = st.slider("Confidence Threshold", 0.5, 1.0, 0.8, 0.05)
        
    st.subheader("📊 Batch Analysis")
    uploaded_file = st.file_uploader(
        "Upload CSV for batch sentiment analysis", 
        type=['csv'], 
        key="sentiment_batch",
        help="Upload a CSV file with a text column for batch analysis"
    )
    
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            st.write("📋 Preview:", df.head())
            if len(df.columns) > 0:
                text_column = st.selectbox("Select text column:", df.columns)
            else:
                st.error("CSV file appears to be empty or invalid.")
                return
        except Exception as e:
            st.error(f"Error reading CSV file: {e}")
            return

    if st.button("🔍 Analyze Sentiment", key="sentiment_button", use_container_width=True):
        if not text_input.strip() and not uploaded_file:
            st.warning("Please enter text or upload a file to analyze.")
            return

        if text_input.strip():
            display_spinner_and_message("Analyzing sentiment...")
            try:
                # Call FastAPI backend for sentiment analysis
                response = requests.post(f"{API_BASE}/sentiment/analyze", json={"text": text_input})
                response.raise_for_status()
                result = response.json()
                
                # Mock additional fields for demonstration if not provided by backend
                result['emotions'] = {"joy": 0.7, "confidence": 0.2, "neutral": 0.1} if emotion_detection else None
                result['detailed'] = {"subjectivity": 0.6, "polarity": 0.8} if detailed_analysis else None

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
    st.write("Generate concise summaries with customizable options.")

    col1, col2 = st.columns([3, 1])
    
    with col1:
        text_input = st.text_area(
            "Paste text to summarize:", 
            height=300, 
            key="summarization_text_input",
            placeholder="Paste your long text here..."
        )
    
    with col2:
        st.subheader("Summary Options")
        summary_length = st.selectbox("Length:", ["Short (1-2 sentences)", "Medium (3-5 sentences)", "Long (6+ sentences)"])
        summary_type = st.selectbox("Type:", ["Abstractive", "Extractive"])
        include_keywords = st.checkbox("Include Keywords", value=True)
        bullet_points = st.checkbox("Bullet Point Format")
        
    st.subheader("📁 Document Upload")
    uploaded_doc = st.file_uploader(
        "Upload document", 
        type=['txt', 'pdf', 'docx'],
        help="Support for TXT, PDF, and DOCX files"
    )
    
    if uploaded_doc:
        st.success(f"📄 Uploaded: {uploaded_doc.name}")
        file_size = len(uploaded_doc.read()) / 1024
        st.info(f"File size: {file_size:.1f} KB")
        uploaded_doc.seek(0)  # Reset file pointer
    
    if st.button("📝 Generate Summary", key="summarization_button", use_container_width=True):
        if not text_input.strip() and not uploaded_doc:
            st.warning("Please provide text or upload a document.")
            return

        display_spinner_and_message("Generating summary...")
        try:
            processed_text = text_input
            if uploaded_doc and not text_input.strip():
                if uploaded_doc.type == "text/plain":
                    processed_text = uploaded_doc.getvalue().decode("utf-8")
                else:
                    st.info("PDF/DOCX processing would require additional libraries (e.g., PyPDF2, python-docx) and backend logic. Using mock text.")
                    processed_text = "Sample text extracted from document for demonstration purposes."
            
            if not processed_text.strip():
                st.warning("No text found to summarize from input or uploaded document.")
                return

            # Call FastAPI backend for summarization
            response = requests.post(f"{API_BASE}/summarization/summarize", json={"text": processed_text})
            response.raise_for_status()
            result = response.json()
            summary = result.get("summary_text", "No summary generated.")
            
            # Mock keywords for demonstration
            keywords = ["innovation", "technology", "advancement", "future", "implications"] if include_keywords else []
            
            st.subheader("📝 Summary")
            if bullet_points:
                bullet_summary = "• " + summary.replace(". ", ".\n• ")
                st.markdown(bullet_summary)
            else:
                st.info(summary)
            
            if include_keywords and keywords:
                st.subheader("🔑 Key Terms")
                cols = st.columns(5)
                for i, keyword in enumerate(keywords):
                    with cols[i % 5]:
                        st.markdown(f"<span style='background-color: #667eea; color: white; padding: 0.3rem 0.6rem; border-radius: 15px; margin: 0.2rem;'>{keyword}</span>", unsafe_allow_html=True)
            
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
                        'keywords': keywords
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

    col1, col2 = st.columns([2, 1])
    
    with col1:
        text_input = st.text_area(
            "Enter your prompt:", 
            value=st.session_state.get('generation_text_input', "Once upon a time,"), 
            height=150, 
            key="generation_text_input",
            placeholder="Start your creative prompt here..."
        )
    
    with col2:
        st.subheader("Generation Settings")
        creativity_level = st.slider("Creativity Level", 0.1, 2.0, 1.0, 0.1)
        max_length = st.slider("Max Length (words)", 50, 500, 200)
        writing_style = st.selectbox("Writing Style:", ["Creative", "Technical", "Casual", "Formal", "Academic"])
        tone = st.selectbox("Tone:", ["Neutral", "Positive", "Professional", "Humorous", "Dramatic"])
        
    st.subheader("📋 Quick Templates")
    templates = {
        "📖 Story": "Write a short story about",
        "📧 Email": "Compose a professional email regarding",
        "📝 Blog Post": "Create a blog post about",
        "🛍️ Product Description": "Write a compelling product description for",
        "📢 Social Media": "Create a social media post about",
        "🎤 Speech": "Write a motivational speech about",
        "📰 News Article": "Write a news article about",
        "🔍 Research Summary": "Summarize research findings on"
    }
    
    col1, col2 = st.columns(2)
    with col1:
        selected_template = st.selectbox("Choose template:", list(templates.keys()), key="gen_template_select")
    with col2:
        if st.button("📝 Use Template", use_container_width=True, key="use_template_btn"):
            st.session_state.generation_text_input = templates[selected_template]
            st.rerun()

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
            
            col1, col2, col3 = st.columns(3)
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
                        'content': generated_text,
                        'settings': {
                            'creativity': creativity_level,
                            'style': writing_style,
                            'tone': tone
                        }
                    })
                    st.success("Saved to favorites!")
            with col3:
                if st.button("🔄 Generate Another", key="regenerate"):
                    st.rerun()
            
            log_to_history("Text Generation", text_input, generated_text)
            
        except requests.exceptions.RequestException as e:
            display_error(f"Could not generate text: {e}")
            log_to_history("Text Generation", text_input, str(e), False)
        except Exception as e:
            display_error(f"An unexpected error occurred: {e}")
            log_to_history("Text Generation", text_input, str(e), False)

def image_captioning_component():
    """Streamlit component for Advanced Image Analysis (Captioning, etc.)."""
    st.header("🖼️ Advanced Image Analysis")
    st.write("Analyze images with AI-powered captioning and object detection.")

    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Choose an image...", 
            type=["jpg", "jpeg", "png", "bmp", "tiff"], 
            key="image_upload",
            help="Supported formats: JPG, JPEG, PNG, BMP, TIFF"
        )
    
    with col2:
        st.subheader("Analysis Options")
        detailed_caption = st.checkbox("Detailed Description", value=True)
        object_detection = st.checkbox("Object Detection", value=True)
        scene_analysis = st.checkbox("Scene Analysis")
        text_extraction = st.checkbox("Text Extraction (OCR)")
        color_analysis = st.checkbox("Color Analysis")
        face_detection = st.checkbox("Face Detection")

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
                
                # Mock additional analysis for demonstration
                objects = ["mountain", "sky", "tree", "grass", "cloud"] if object_detection else []
                scene = "Outdoor natural landscape with mountain scenery" if scene_analysis else ""
                extracted_text = "No text detected in image" if text_extraction else ""
                colors = ["Blue", "Green", "Brown", "White"] if color_analysis else []
                faces = 0 if face_detection else None
                
                st.subheader("🔍 Analysis Results")
                
                st.markdown(f"<div style='background: #e3f2fd; padding: 1rem; border-radius: 10px; border-left: 4px solid #2196f3;'>"
                          f"<h4>📝 Image Caption</h4><p>{caption}</p></div>", unsafe_allow_html=True)
                
                if any([objects, scene, extracted_text, colors, faces is not None]):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if objects:
                            st.subheader("🎯 Detected Objects")
                            for obj in objects:
                                st.markdown(f"<span style='background-color: #4caf50; color: white; padding: 0.2rem 0.5rem; border-radius: 10px; margin: 0.1rem; display: inline-block;'>{obj}</span>", unsafe_allow_html=True)
                        
                        if scene:
                            st.subheader("🌄 Scene Analysis")
                            st.info(scene)
                        
                        if faces is not None:
                            st.subheader("👤 Face Detection")
                            st.metric("Faces Detected", faces)
                    
                    with col2:
                        if colors:
                            st.subheader("🎨 Color Analysis")
                            for color in colors:
                                st.markdown(f"<span style='background-color: {color.lower()}; color: white; padding: 0.2rem 0.5rem; border-radius: 10px; margin: 0.1rem; display: inline-block;'>{color}</span>", unsafe_allow_html=True)
                        
                        if text_extraction:
                            st.subheader("📝 Extracted Text")
                            if extracted_text and extracted_text != "No text detected in image":
                                st.code(extracted_text)
                            else:
                                st.info("No text found in image")
                
                st.subheader("📊 Analysis Confidence")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Caption Confidence", "94%")
                with col2:
                    st.metric("Objects Detected", len(objects))
                with col3:
                    st.metric("Processing Time", "2.3s")
                with col4:
                    st.metric("Image Quality", "High")
                
                col1, col2 = st.columns(2)
                with col1:
                    analysis_report = f"""Image Analysis Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
File: {uploaded_file.name}

Caption: {caption}

Objects Detected: {', '.join(objects) if objects else 'None'}
Scene: {scene if scene else 'Not analyzed'}
Text: {extracted_text if extracted_text else 'None'}
Colors: {', '.join(colors) if colors else 'Not analyzed'}
Faces: {faces if faces is not None else 'Not analyzed'}
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
                            'objects': objects,
                            'scene': scene
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
    st.write("Translate text between multiple languages with high accuracy.")

    col1, col2 = st.columns([2, 1])
    
    with col1:
        text_input = st.text_area(
            "Enter text to translate:", 
            height=200, 
            key="translation_text_input",
            placeholder="Enter text in any language...",
            value=st.session_state.translation_text_input # Use session state for persistence
        )
    
    with col2:
        st.subheader("Translation Settings")
        source_lang = st.selectbox("Source Language:", [
            "Auto-detect", "English", "Spanish", "French", "German", 
            "Italian", "Portuguese", "Russian", "Chinese", "Japanese", 
            "Korean", "Arabic", "Hindi", "Dutch", "Swedish"
        ], key="translation_source", index=["Auto-detect", "English", "Spanish", "French", "German", 
            "Italian", "Portuguese", "Russian", "Chinese", "Japanese", 
            "Korean", "Arabic", "Hindi", "Dutch", "Swedish"].index(st.session_state.translation_source))
        
        target_lang = st.selectbox("Target Language:", [
            "English", "Spanish", "French", "German", "Italian", 
            "Portuguese", "Russian", "Chinese", "Japanese", "Korean", 
            "Arabic", "Hindi", "Dutch", "Swedish"
        ], key="translation_target", index=["English", "Spanish", "French", "German", "Italian", 
            "Portuguese", "Russian", "Chinese", "Japanese", "Korean", 
            "Arabic", "Hindi", "Dutch", "Swedish"].index(st.session_state.translation_target))
        
        formal_tone = st.checkbox("Formal tone", value=False)
        preserve_formatting = st.checkbox("Preserve formatting", value=True)
    
    st.subheader("🔄 Quick Translations")
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🇺🇸 → 🇪🇸 EN to ES", use_container_width=True):
            st.session_state.translation_source = "English"
            st.session_state.translation_target = "Spanish"
            st.session_state.translation_text_input = "Hello, how are you today?"
            st.rerun()
    with col2:
        if st.button("🇪🇸 → 🇺🇸 ES to EN", use_container_width=True):
            st.session_state.translation_source = "Spanish"
            st.session_state.translation_target = "English"
            st.session_state.translation_text_input = "Hola, ¿cómo estás hoy?"
            st.rerun()
    with col3:
        if st.button("🇺🇸 → 🇫🇷 EN to FR", use_container_width=True):
            st.session_state.translation_source = "English"
            st.session_state.translation_target = "French"
            st.session_state.translation_text_input = "Hello, how are you today?"
            st.rerun()
    
    st.subheader("📊 Batch Translation")
    uploaded_file = st.file_uploader(
        "Upload file for batch translation", 
        type=['txt', 'csv'],
        help="Upload TXT or CSV file for batch translation"
    )
    
    if st.button("🔄 Translate", key="translation_button", use_container_width=True):
        if not text_input.strip() and not uploaded_file:
            st.warning("Please enter text or upload a file to translate.")
            return

        display_spinner_and_message("Translating text...")
        try:
            original_text_to_translate = text_input
            if uploaded_file:
                if uploaded_file.type == "text/plain":
                    original_text_to_translate = uploaded_file.getvalue().decode("utf-8")
                elif uploaded_file.type == "text/csv":
                    df_to_translate = pd.read_csv(uploaded_file)
                    # For simplicity, assume first column is text to translate
                    original_text_to_translate = "\n".join(df_to_translate.iloc[:, 0].astype(str).tolist())
                st.info(f"Translating content from {uploaded_file.name}...")

            # Call FastAPI backend for translation
            # Note: The backend currently only supports EN to FR. For other languages,
            # you would need to load different translation models in the backend.
            # For demonstration, we'll use a mock if not EN to FR.
            if source_lang == "English" and target_lang == "French":
                response = requests.post(f"{API_BASE}/translation/translate", json={"text": original_text_to_translate})
                response.raise_for_status()
                translated_text = response.json().get("translated_text", "Translation failed.")
            else:
                # Simple mock translation logic for other language pairs
                mock_translations = {
                    "English": "Hello, how are you today?",
                    "Spanish": "Hola, ¿cómo estás hoy?",
                    "French": "Bonjour, comment allez-vous aujourd'hui?",
                    "German": "Hallo, wie geht es dir heute?",
                    "Italian": "Ciao, come stai oggi?",
                    "Portuguese": "Olá, como você está hoje?",
                    "Russian": "Привет, как дела сегодня?",
                    "Chinese": "你好，你今天好吗？",
                    "Japanese": "こんにちは、今日はいかがですか？",
                    "Korean": "안녕하세요, 오늘 어떻게 지내세요?",
                    "Arabic": "مرحبا، كيف حالك اليوم؟",
                    "Hindi": "नमस्ते, आज आप कैसे हैं?"
                }
                translated_text = mock_translations.get(target_lang, f"Translation to {target_lang} is not supported by the current backend or mocked. Original: {original_text_to_translate}")
            
            st.session_state.translation_history.append({
                'original': original_text_to_translate,
                'translated': translated_text,
                'source_lang': source_lang,
                'target_lang': target_lang,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
            
            st.subheader("🔄 Translation Results")
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"<div style='background: #f8f9fa; padding: 1rem; border-radius: 10px; border-left: 4px solid #dc3545;'>"
                          f"<h5>📝 Original ({source_lang})</h5><p>{original_text_to_translate}</p></div>", unsafe_allow_html=True)
            with col2:
                st.markdown(f"<div style='background: #f8f9fa; padding: 1rem; border-radius: 10px; border-left: 4px solid #28a745;'>"
                          f"<h5>🔄 Translation ({target_lang})</h5><p>{translated_text}</p></div>", unsafe_allow_html=True)
            
            st.subheader("📊 Translation Metrics")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Original Length", f"{len(original_text_to_translate)} chars")
            with col2:
                st.metric("Translated Length", f"{len(translated_text)} chars")
            with col3:
                st.metric("Confidence", "96%") # Mock value
            with col4:
                st.metric("Processing Time", "1.2s") # Mock value
            
            col1, col2, col3 = st.columns(3)
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
                        'source_lang': source_lang,
                        'target_lang': target_lang
                    })
                    st.success("Saved to favorites!")
            with col3:
                if st.button("🔄 Reverse Translation", key="reverse_translation"):
                    st.info("Reverse translation feature would swap source and target languages and populate the text area.")
                    st.session_state.translation_text_input = translated_text
                    st.session_state.translation_source = target_lang
                    st.session_state.translation_target = source_lang
                    st.rerun()
            
            log_to_history("Language Translation", f"{source_lang} → {target_lang}", translated_text)
            
        except requests.exceptions.RequestException as e:
            display_error(f"Could not translate text: {e}")
            log_to_history("Language Translation", text_input[:100], str(e), False)
        except Exception as e:
            display_error(f"An unexpected error occurred: {e}")
            log_to_history("Language Translation", text_input[:100], str(e), False)

def question_answering_component():
    """Streamlit component for Question Answering."""
    st.header("❓ Question Answering")
    st.write("Get intelligent answers to your questions with context-aware AI.")

    col1, col2 = st.columns([2, 1])
    
    with col1:
        context_input = st.text_area(
            "Provide context (optional):", 
            height=150, 
            key="qa_context_input",
            placeholder="Enter relevant context or background information...",
            value=st.session_state.qa_context_input
        )
        
        question_input = st.text_input(
            "Ask your question:", 
            key="qa_question_input",
            placeholder="What would you like to know?",
            value=st.session_state.qa_question_input
        )
    
    with col2:
        st.subheader("Answer Settings")
        answer_length = st.selectbox("Answer Length:", ["Short", "Medium", "Detailed"])
        include_sources = st.checkbox("Include Sources", value=True)
        confidence_display = st.checkbox("Show Confidence", value=True)
        multiple_perspectives = st.checkbox("Multiple Perspectives")
    
    st.subheader("📄 Document Context")
    uploaded_doc = st.file_uploader(
        "Upload document for context", 
        type=['txt', 'pdf', 'docx'],
        help="Upload a document to use as context for answering questions"
    )
    
    st.subheader("📋 Question Categories")
    categories = {
        "🔬 Science": ["How does photosynthesis work?", "What causes climate change?", "How do vaccines work?"],
        "🏛️ History": ["Who was Napoleon Bonaparte?", "What caused World War I?", "When was the Renaissance?"],
        "💻 Technology": ["What is artificial intelligence?", "How does blockchain work?", "What is cloud computing?"],
        "🎭 Literature": ["Who wrote Romeo and Juliet?", "What is the theme of 1984?", "Define literary symbolism"],
        "🧮 Mathematics": ["What is calculus?", "How do you solve quadratic equations?", "What is probability?"]
    }
    
    selected_category = st.selectbox("Choose category:", list(categories.keys()), key="qa_category_select")
    sample_questions = categories[selected_category]
    
    cols = st.columns(3)
    for i, question in enumerate(sample_questions):
        with cols[i % 3]:
            if st.button(f"📝 {question[:20]}...", key=f"sample_q_{i}", use_container_width=True):
                st.session_state.qa_question_input = question
                st.rerun()

    if st.button("🤔 Get Answer", key="qa_button", use_container_width=True):
        if not question_input.strip():
            st.warning("Please enter a question.")
            return

        display_spinner_and_message("Searching for the best answer...")
        try:
            # Mock answer for demonstration (backend integration would go here)
            answer = f"Based on the question '{question_input}', here's a comprehensive answer: This is a complex topic that requires careful consideration of multiple factors. The primary explanation involves understanding the fundamental principles and their practical applications. Key points include the historical context, current understanding, and future implications of this subject matter."
            
            confidence = 0.89
            sources = ["Encyclopedia Britannica", "Academic Journal", "Expert Opinion"] if include_sources else []
            
            st.subheader("💡 Answer")
            st.markdown(f"<div style='background: #e8f5e8; padding: 1.5rem; border-radius: 10px; border-left: 4px solid #4caf50;'>"
                      f"<h4>🤔 Question:</h4><p><em>{question_input}</em></p>"
                      f"<h4>💡 Answer:</h4><p>{answer}</p></div>", unsafe_allow_html=True)
            
            if confidence_display:
                st.subheader("📊 Answer Quality")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Confidence", f"{confidence:.1%}")
                with col2:
                    st.metric("Answer Length", f"{len(answer.split())} words")
                with col3:
                    st.metric("Processing Time", "2.1s")
                with col4:
                    relevance = "High" if confidence > 0.8 else "Medium" if confidence > 0.6 else "Low"
                    st.metric("Relevance", relevance)
            
            if include_sources and sources:
                st.subheader("📚 Sources")
                for i, source in enumerate(sources, 1):
                    st.markdown(f"**{i}.** {source}")
            
            if multiple_perspectives:
                st.subheader("🔄 Alternative Perspectives")
                perspectives = [
                    "From a scientific standpoint, this phenomenon can be explained through...",
                    "Historically, this has been understood as...",
                    "From a practical application perspective..."
                ]
                for i, perspective in enumerate(perspectives, 1):
                    st.markdown(f"**Perspective {i}:** {perspective}")
            
            col1, col2 = st.columns(2)
            with col1:
                qa_report = f"""Question & Answer Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Question: {question_input}
Context: {context_input if context_input else 'None provided'}

Answer: {answer}

Confidence: {confidence:.1%}
Sources: {', '.join(sources) if sources else 'None'}
"""
                st.download_button(
                    label="📥 Download Q&A",
                    data=qa_report,
                    file_name="qa_report.txt",
                    mime="text/plain",
                    use_container_width=True
                )
            with col2:
                if st.button("⭐ Save to Favorites", key="save_qa"):
                    add_to_favorites("Question & Answer", {
                        'question': question_input,
                        'answer': answer,
                        'confidence': confidence
                    })
                    st.success("Saved to favorites!")
            
            log_to_history("Question Answering", question_input, answer)
            
        except Exception as e:
            display_error(f"Unexpected error: {e}")
            log_to_history("Question Answering", question_input, str(e), False)

def chatbot_component():
    """Streamlit component for AI Chatbot."""
    st.header("🤖 AI Chatbot")
    st.write("Have a conversation with our intelligent AI assistant.")

    col1, col2 = st.columns([3, 1])
    with col2:
        st.subheader("Chat Settings")
        personality = st.selectbox("Personality:", ["Professional", "Friendly", "Technical", "Creative", "Humorous"], key="chat_personality")
        response_style = st.selectbox("Response Style:", ["Concise", "Detailed", "Conversational"], key="chat_response_style")
        remember_context = st.checkbox("Remember Context", value=True, key="chat_remember_context")
        
        if st.button("🗑️ Clear Chat", use_container_width=True, key="clear_chat_button"):
            st.session_state.chat_history = []
            st.rerun()
    
    with col1:
        st.subheader("💬 Chat History")
        chat_container = st.container(height=400) # Fixed height for chat history
        
        with chat_container:
            if st.session_state.chat_history:
                for i, message in enumerate(st.session_state.chat_history):
                    if message['role'] == 'user':
                        st.markdown(f"<div style='background: #e3f2fd; padding: 0.8rem; border-radius: 10px; margin: 0.5rem 0; margin-left: 2rem;'>"
                                  f"<strong>🙋 You:</strong> {message['content']}</div>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"<div style='background: #f3e5f5; padding: 0.8rem; border-radius: 10px; margin: 0.5rem 0; margin-right: 2rem;'>"
                                  f"<strong>🤖 AI:</strong> {message['content']}</div>", unsafe_allow_html=True)
            else:
                st.info("👋 Start a conversation! Type your message below.")
    
    col1, col2 = st.columns([4, 1])
    with col1:
        user_input = st.text_input(
            "Type your message:", 
            key="chat_input",
            placeholder="Ask me anything...",
            value=st.session_state.chat_input
        )
    with col2:
        send_button = st.button("📤 Send", key="send_chat", use_container_width=True)
    
    st.subheader("⚡ Quick Prompts")
    quick_prompts = [
        "Tell me a joke", "Explain quantum physics", "Write a poem", 
        "Give me productivity tips", "What's the weather like?", "Help with coding"
    ]
    
    cols = st.columns(3)
    for i, prompt in enumerate(quick_prompts):
        with cols[i % 3]:
            if st.button(f"💬 {prompt}", key=f"quick_{i}", use_container_width=True):
                st.session_state.chat_input = prompt
                st.rerun()

    if send_button and user_input.strip():
        st.session_state.chat_history.append({
            'role': 'user',
            'content': user_input,
            'timestamp': datetime.now().strftime("%H:%M:%S")
        })
        
        display_spinner_and_message("AI is thinking...")
        try:
            # Mock AI response (backend integration would go here)
            ai_responses = {
                "Professional": f"Thank you for your question about '{user_input}'. Based on my analysis, I can provide you with a comprehensive response that addresses your inquiry professionally and thoroughly.",
                "Friendly": f"Hey there! Great question about '{user_input}'! I'd be happy to help you with that. Let me share some insights that might be useful for you.",
                "Technical": f"Analyzing your query '{user_input}', I can provide technical specifications and detailed implementation details relevant to your request.",
                "Creative": f"What an interesting question about '{user_input}'! Let me explore this creatively and provide you with some imaginative perspectives and solutions.",
                "Humorous": f"Ha! You asked about '{user_input}' - that's a great question! Let me give you an answer that's both informative and entertaining."
            }
            
            ai_response = ai_responses.get(personality, f"I understand you're asking about '{user_input}'. Here's my response based on the available information and context.")
            
            st.session_state.chat_history.append({
                'role': 'assistant',
                'content': ai_response,
                'timestamp': datetime.now().strftime("%H:%M:%S")
            })
            
            st.session_state.chat_input = ""
            log_to_history("Chatbot", user_input, ai_response)
            st.rerun()
            
        except Exception as e:
            display_error(f"Unexpected error: {e}")
            log_to_history("Chatbot", user_input, str(e), False)

def speech_to_text_component():
    """Streamlit component for Speech-to-Text."""
    st.header("🎤 Speech to Text")
    st.write("Convert spoken audio into written text.")
    st.warning("This feature uses a self-hosted model, which may require initial download time on the backend.")

    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📁 Audio Upload")
        uploaded_audio = st.file_uploader(
            "Upload audio file", 
            type=['wav', 'mp3', 'flac', 'm4a'],
            key="stt_audio_upload",
            help="Supported formats: WAV, MP3, FLAC, M4A"
        )
        
        if uploaded_audio:
            st.audio(uploaded_audio, format=uploaded_audio.type)
            file_size = len(uploaded_audio.read()) / 1024
            st.info(f"📄 File: {uploaded_audio.name} | Size: {file_size:.1f} KB")
            uploaded_audio.seek(0)
    
    with col2:
        st.subheader("Recognition Settings")
        language = st.selectbox("Language:", [
            "English (US)", "English (UK)", "Spanish", "French", 
            "German", "Italian", "Portuguese", "Chinese", "Japanese"
        ], key="stt_language_select")
        
        enhance_accuracy = st.checkbox("Enhanced Accuracy", value=True)
        punctuation = st.checkbox("Auto Punctuation", value=True)
        speaker_detection = st.checkbox("Speaker Detection")
        timestamps = st.checkbox("Include Timestamps")
        
        st.subheader("🎙️ Live Recording")
        audio_bytes_recorded = st.audio_recorder("Click to record audio", key="stt_audio_recorder")
        
        if audio_bytes_recorded:
            st.audio(audio_bytes_recorded, format="audio/wav")

    source_audio_data = None
    source_filename = "audio_input.wav"
    source_mime = "audio/wav"

    if audio_bytes_recorded:
        source_audio_data = audio_bytes_recorded
        source_filename = "recorded_audio.wav"
        source_mime = "audio/wav"
    elif uploaded_audio:
        source_audio_data = uploaded_audio.getvalue()
        source_filename = uploaded_audio.name
        source_mime = uploaded_audio.type

    if st.button("🔊 Convert to Text", key="stt_convert_button", use_container_width=True):
        if source_audio_data is None:
            st.warning("Please record audio or upload an audio file to transcribe.")
            return

        display_spinner_and_message("Converting speech to text...")
        try:
            # Call FastAPI backend for STT
            files = {"file": (source_filename, source_audio_data, source_mime)}
            response = requests.post(f"{API_BASE}/stt", files=files)
            response.raise_for_status()
            result = response.json()
            transcription = result.get("transcribed_text", "Could not transcribe audio.")
            
            confidence = 0.94 # Mock value
            word_count = len(transcription.split())
            duration = "2:34"  # Mock duration
            
            st.subheader("📝 Transcription Results")
            st.markdown(f"<div style='background: #f0f8ff; padding: 1.5rem; border-radius: 10px; border-left: 4px solid #007bff;'>{transcription}</div>", unsafe_allow_html=True)
            display_success("Audio transcribed successfully!")

            st.subheader("📊 Transcription Statistics")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Confidence", f"{confidence:.1%}")
            with col2:
                st.metric("Word Count", word_count)
            with col3:
                st.metric("Duration", duration)
            with col4:
                st.metric("Processing Time", "12.3s")
            
            if speaker_detection:
                st.subheader("👥 Speaker Analysis")
                speakers = [
                    {"speaker": "Speaker 1", "duration": "1:45", "words": 180},
                    {"speaker": "Speaker 2", "duration": "0:49", "words": 95}
                ]
                
                for speaker in speakers:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric(speaker["speaker"], "Active")
                    with col2:
                        st.metric("Duration", speaker["duration"])
                    with col3:
                        st.metric("Words", speaker["words"])
            
            if timestamps:
                st.subheader("⏰ Timestamps")
                timestamp_text = """[00:00] Hello, this is a sample transcription
[00:15] of the uploaded audio file. The speech
[00:30] recognition system has processed your audio
[00:45] and converted it to text with high accuracy."""
                st.code(timestamp_text)
            
            col1, col2, col3 = st.columns(3)
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
                        'confidence': confidence
                    })
                    st.success("Saved to favorites!")
            with col3:
                if st.button("📋 Copy to Clipboard", key="copy_stt"):
                    st.info("Transcript copied to clipboard!")
            
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
    st.write("Convert text to natural-sounding speech with customizable voices.")

    col1, col2 = st.columns([2, 1])
    
    with col1:
        text_input = st.text_area(
            "Enter text to convert to speech:", 
            height=200, 
            key="tts_text_input",
            value="Hello! This is a sample text for text-to-speech conversion. The AI will generate natural-sounding speech from this text.",
            placeholder="Enter your text here..."
        )
    
    with col2:
        st.subheader("Voice Settings")
        voice_type = st.selectbox("Voice:", ["Female (Neural)", "Male (Neural)", "Female (Standard)", "Male (Standard)"], key="tts_voice_type")
        language = st.selectbox("Language:", ["English (US)", "English (UK)", "Spanish", "French", "German", "Italian"], key="tts_language")
        
        speed = st.slider("Speech Speed", 0.5, 2.0, 1.0, 0.1, key="tts_speed")
        pitch = st.slider("Pitch", 0.5, 2.0, 1.0, 0.1, key="tts_pitch")
        volume = st.slider("Volume", 0.1, 1.0, 0.8, 0.1, key="tts_volume")
        
        st.subheader("Advanced Options")
        add_pauses = st.checkbox("Natural Pauses", value=True)
        emphasize_caps = st.checkbox("Emphasize CAPS", value=False)
        ssml_enabled = st.checkbox("SSML Support", value=False)
    
    st.subheader("📝 Text Preprocessing")
    col1, col2 = st.columns(2)
    with col1:
        word_count = len(text_input.split()) if text_input else 0
        st.metric("Word Count", word_count)
    with col2:
        estimated_duration = word_count / 150 * 60  # Average speaking rate (seconds)
        st.metric("Estimated Duration", f"{estimated_duration:.1f}s")
    
    st.subheader("🎵 Voice Samples")
    sample_voices = ["Emma (Neural)", "Brian (Neural)", "Amy (Standard)", "Justin (Standard)"]
    cols = st.columns(4)
    
    for i, voice in enumerate(sample_voices):
        with cols[i]:
            if st.button(f"🔊 {voice}", key=f"sample_voice_{i}", use_container_width=True):
                st.info(f"Playing sample with {voice}")

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
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Duration", f"{estimated_duration:.1f}s")
            with col2:
                st.metric("File Size", f"{len(audio_bytes) / (1024*1024):.2f} MB") 
            with col3:
                st.metric("Sample Rate", "22 kHz")
            with col4:
                st.metric("Quality", "High")
            
            st.subheader("🌊 Audio Waveform")
            time_points = np.linspace(0, estimated_duration, int(estimated_duration * 100))
            waveform = np.sin(2 * np.pi * 2 * time_points) * np.exp(-time_points/10) + \
                       0.5 * np.sin(2 * np.pi * 5 * time_points) * np.exp(-time_points/5)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=time_points, y=waveform, mode='lines', name='Waveform', line=dict(color='#667eea')))
            fig.update_layout(
                title="Audio Waveform",
                xaxis_title="Time (seconds)",
                yaxis_title="Amplitude",
                height=300,
                margin=dict(l=20, r=20, t=40, b=20)
            )
            st.plotly_chart(fig, use_container_width=True)
            
            col1, col2, col3 = st.columns(3)
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
                        'voice': voice_type,
                        'language': language,
                        'settings': {
                            'speed': speed,
                            'pitch': pitch,
                            'volume': volume
                        }
                    })
                    st.success("Saved to favorites!")
            with col3:
                if st.button("🔄 Regenerate", key="regenerate_tts"):
                    st.info("Regenerating with current settings...")
            
            log_to_history("Text to Speech", text_input[:100], f"Generated speech ({voice_type})")
            
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
tab_sentiment, tab_summarization, tab_generation, tab_captioning, tab_translation, tab_tts, tab_stt, tab_qa, tab_chatbot, tab_history, tab_favorites, tab_settings = st.tabs([
    "Sentiment Analysis", "Text Summarization", "Text Generation",
    "Image Analysis", "Translation", "Text-to-Speech", "Speech-to-Text",
    "Question Answering", "AI Chatbot", "History", "Favorites", "Settings"
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

with tab_qa:
    question_answering_component()

with tab_chatbot:
    chatbot_component()

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

    st.subheader("Text-to-Speech Defaults")
    st.session_state.user_preferences['default_voice'] = st.selectbox(
        "Default Voice:", ["Male", "Female", "Child"],
        index=["Male", "Female", "Child"].index(st.session_state.user_preferences['default_voice']),
        key="settings_default_voice_select"
    )
    st.session_state.user_preferences['default_language'] = st.selectbox(
        "Default TTS Language:", ["English", "French", "Spanish"],
        index=["English", "French", "Spanish"].index(st.session_state.user_preferences['default_language']),
        key="settings_default_language_select"
    )

    st.subheader("API Configuration (Advanced)")
    # No 'global API_BASE' needed here. API_BASE is a module-level variable.
    # When this code runs, it reads the current value of API_BASE.
    # Reassigning it directly updates the module-level variable for subsequent runs within the same script execution.
    new_api_base = st.text_input("Backend API Base URL:", value=API_BASE, key="settings_api_base_input")
    if new_api_base != API_BASE:
        st.warning(f"API Base URL changed from {API_BASE} to {new_api_base}. This change will apply on next rerun.")
        # Directly reassign the module-level variable
        globals()['API_BASE'] = new_api_base
        st.button("Apply API Change and Rerun", key="apply_api_change")

st.markdown("---")
st.markdown("Developed with FastAPI and Streamlit.")
