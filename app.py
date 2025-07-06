import streamlit as st
import requests
import io
import time

# --- Configuration ---
API_BASE = "http://localhost:8000/api"

st.set_page_config(
    page_title="AI Services Toolkit",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Helper Functions ---
@st.cache_data(ttl=3) # Cache status for 3 seconds to avoid excessive calls
def get_backend_status():
    """Checks the backend status."""
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

def display_spinner_and_message(message):
    """Displays a spinner and a message."""
    st.info(f"⏳ {message}")
    st.spinner("Processing...")

def display_error(error_message):
    """Displays an error message."""
    st.error(f"🚨 Error: {error_message}")

# --- AI Service Components (Functions for each tab) ---

def sentiment_analysis_component():
    st.header("Sentiment Analysis")
    st.write("Determine the emotional tone of a piece of text (positive or negative).")

    text_input = st.text_area("Enter text to analyze sentiment:", height=150, key="sentiment_text_input")

    if st.button("Analyze Sentiment", key="sentiment_button"):
        if not text_input.strip():
            st.warning("Please enter some text to analyze.")
            return

        display_spinner_and_message("Analyzing sentiment...")
        try:
            response = requests.post(f"{API_BASE}/sentiment/analyze", json={"text": text_input})
            response.raise_for_status()
            result = response.json()
            label = result.get("label", "N/A")
            score = result.get("score", 0.0)

            st.subheader("Result")
            if label == "POSITIVE":
                st.success(f"Sentiment: **{label}** (Confidence: {score:.2%})")
            else:
                st.error(f"Sentiment: **{label}** (Confidence: {score:.2%})")
        except requests.exceptions.RequestException as e:
            display_error(f"Could not analyze sentiment: {e}")
        except Exception as e:
            display_error(f"An unexpected error occurred: {e}")

def text_summarization_component():
    st.header("Text Summarization")
    st.write("Generate a concise summary of a long article or document.")

    text_input = st.text_area("Paste a long article here to get a summary:", height=300, key="summarization_text_input")

    if st.button("Summarize Text", key="summarization_button"):
        if not text_input.strip():
            st.warning("Please paste some text to summarize.")
            return

        display_spinner_and_message("Summarizing text...")
        try:
            response = requests.post(f"{API_BASE}/summarization/summarize", json={"text": text_input})
            response.raise_for_status()
            result = response.json()
            summary = result.get("summary_text", "No summary generated.")

            st.subheader("Summary")
            st.info(summary)
        except requests.exceptions.RequestException as e:
            display_error(f"Could not summarize text: {e}")
        except Exception as e:
            display_error(f"An unexpected error occurred: {e}")

def text_generation_component():
    st.header("Text Generation")
    st.write("Generate creative text based on a given prompt.")

    text_input = st.text_area("Enter a prompt to generate text:", value="Once upon a time,", height=150, key="generation_text_input")

    if st.button("Generate Text", key="generation_button"):
        if not text_input.strip():
            st.warning("Please enter a prompt to generate text.")
            return

        display_spinner_and_message("Generating text...")
        try:
            response = requests.post(f"{API_BASE}/generation/generate", json={"text": text_input})
            response.raise_for_status()
            result = response.json()
            generated_text = result.get("generated_text", "No text generated.")

            st.subheader("Generated Text")
            st.success(generated_text)
        except requests.exceptions.RequestException as e:
            display_error(f"Could not generate text: {e}")
        except Exception as e:
            display_error(f"An unexpected error occurred: {e}")

def image_captioning_component():
    st.header("Image Captioning")
    st.write("Upload an image to get a descriptive caption.")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], key="image_upload")

    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
        if st.button("Generate Caption", key="captioning_button"):
            display_spinner_and_message("Generating caption...")
            try:
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                response = requests.post(f"{API_BASE}/image/caption", files=files)
                response.raise_for_status()
                result = response.json()
                caption = result.get("generated_text", "No caption generated.")

                st.subheader("Generated Caption")
                st.info(caption)
            except requests.exceptions.RequestException as e:
                display_error(f"Could not generate caption: {e}")
            except Exception as e:
                display_error(f"An unexpected error occurred: {e}")
    else:
        st.info("Upload an image file to get started.")

def text_to_speech_component():
    st.header("Text-to-Speech")
    st.write("Convert written text into natural-sounding speech.")
    st.warning("This feature uses a self-hosted model, which may require initial download time.")

    text_input = st.text_area("Enter text to convert to speech:", height=150, key="tts_text_input")

    if st.button("Generate Speech", key="tts_button"):
        if not text_input.strip():
            st.warning("Please enter some text for speech generation.")
            return

        display_spinner_and_message("Generating speech...")
        try:
            response = requests.post(f"{API_BASE}/tts", json={"text": text_input})
            response.raise_for_status()

            # Streamlit's audio widget can play audio directly from bytes
            audio_bytes = response.content
            st.subheader("Generated Audio")
            st.audio(audio_bytes, format='audio/wav')
            st.success("Speech generated successfully!")
        except requests.exceptions.RequestException as e:
            display_error(f"Could not generate speech: {e}")
        except Exception as e:
            display_error(f"An unexpected error occurred: {e}")

def speech_to_text_component():
    st.header("Speech-to-Text")
    st.write("Transcribe spoken audio into written text.")
    st.warning("This feature uses a self-hosted model, which may require initial download time.")

    audio_bytes = st.audio_recorder("Click to record audio", key="stt_audio_recorder")

    if audio_bytes:
        st.audio(audio_bytes, format="audio/wav") # Play back the recorded audio
        if st.button("Transcribe Audio", key="stt_button"):
            display_spinner_and_message("Transcribing audio...")
            try:
                # Streamlit's audio_recorder returns bytes, which can be sent directly
                # as a file in a multipart/form-data request.
                files = {"file": ("recorded_audio.wav", audio_bytes, "audio/wav")}
                response = requests.post(f"{API_BASE}/stt", files=files)
                response.raise_for_status()
                result = response.json()
                transcribed_text = result.get("transcribed_text", "Could not transcribe audio.")

                st.subheader("Transcribed Text")
                st.info(transcribed_text)
            except requests.exceptions.RequestException as e:
                display_error(f"Could not transcribe audio: {e}")
            except Exception as e:
                display_error(f"An unexpected error occurred: {e}")
    else:
        st.info("Record some audio to transcribe.")


# --- Main Application Layout ---

st.sidebar.title("AI Services Toolkit")
st.sidebar.write("A suite of powerful, self-hosted AI models.")

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


st.title("Professional AI Toolkit Dashboard")

# Use st.tabs for navigation
tab_sentiment, tab_summarization, tab_generation, tab_captioning, tab_tts, tab_stt = st.tabs([
    "Sentiment Analysis", "Text Summarization", "Text Generation",
    "Image Captioning", "Text-to-Speech", "Speech-to-Text"
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

st.markdown("---")
st.markdown("Developed with FastAPI and Streamlit.")
