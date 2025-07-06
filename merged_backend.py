import uvicorn
import logging
from fastapi import FastAPI, APIRouter, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from transformers import pipeline
from PIL import Image
import io
import soundfile as sf # For handling audio files (WAV)
import numpy as np # For numerical operations, especially with audio data

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = FastAPI(
    title="Professional AI Toolkit (Python 3.10+)", # Adjusted Python version for broader compatibility
    description="A suite of high-performance, self-hosted AI models including TTS/STT.",
    version="1.4.1", # Updated version to reflect changes
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allow all origins for broader compatibility with Streamlit
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

models = {}
MODELS_LOADED = False

@app.on_event("startup")
def load_models():
    """
    Loads all AI models at application startup.
    This ensures models are ready for inference and avoids reloading.
    """
    global MODELS_LOADED
    logging.info("Starting to load AI models...")
    try:
        # Existing models
        models["sentiment"] = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
        logging.info("Sentiment Analysis model loaded.")
        models["summarizer"] = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
        logging.info("Text Summarization model loaded.")
        models["translator"] = pipeline("translation_en_to_fr", model="Helsinki-NLP/opus-mt-en-fr")
        logging.info("Language Translation model loaded (EN-FR).")
        models["captioner"] = pipeline("image-to-text", model="nlpconnect/vit-gpt2-image-captioning")
        logging.info("Image Captioning model loaded.")
        models["generator"] = pipeline("text-generation", model="distilgpt2")
        logging.info("Text Generation model loaded.")

        # New: Text-to-Speech model
        # Using 'facebook/mms-tts-eng' for English TTS. This model generates audio.
        # Ensure you have 'soundfile' and 'librosa' installed for audio handling.
        models["tts"] = pipeline("text-to-speech", model="facebook/mms-tts-eng")
        logging.info("Text-to-Speech model loaded.")

        # New: Speech-to-Text (Automatic Speech Recognition) model
        # Using 'openai/whisper-tiny' for STT. This model transcribes audio to text.
        models["stt"] = pipeline("automatic-speech-recognition", model="openai/whisper-tiny")
        logging.info("Speech-to-Text model loaded.")

        # No specific QA model loaded here, as frontend uses mock for now.
        # If a real QA model is needed, it would be added here.
        # No specific Chatbot model loaded here, as frontend uses mock for now.
        # The text-generation model can serve as a base for a simple chatbot.

        MODELS_LOADED = True
        logging.info("All models loaded successfully.")
    except Exception as e:
        logging.error(f"Fatal error during model loading: {e}", exc_info=True)
        # In a production environment, you might want to exit or disable services
        # if critical models fail to load.

class TextIn(BaseModel):
    """Pydantic model for text input."""
    text: str

class TranslationOut(BaseModel):
    """Pydantic model for translation output."""
    original_text: str
    translated_text: str

class TranscriptionOut(BaseModel):
    """Pydantic model for transcription output."""
    transcribed_text: str

class QuestionAnsweringIn(BaseModel):
    """Pydantic model for Question Answering input."""
    question: str
    context: str = "" # Context is optional

class QuestionAnsweringOut(BaseModel):
    """Pydantic model for Question Answering output."""
    answer: str
    confidence: float = 0.0
    sources: list[str] = []

api_router = APIRouter()

@api_router.post("/sentiment/analyze")
def analyze_sentiment(payload: TextIn):
    """
    Analyzes the sentiment of a given text.
    """
    try:
        if not MODELS_LOADED:
            raise HTTPException(status_code=503, detail="Models are not loaded yet. Please wait.")
        result = models["sentiment"](payload.text)
        return result[0]
    except Exception as e:
        logging.error(f"Error in sentiment analysis: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error processing sentiment analysis.")

@api_router.post("/summarization/summarize")
def summarize_text(payload: TextIn):
    """
    Generates a concise summary of a long piece of text.
    """
    try:
        if not MODELS_LOADED:
            raise HTTPException(status_code=503, detail="Models are not loaded yet. Please wait.")
        summary = models["summarizer"](payload.text, max_length=150, min_length=30, do_sample=False)
        return summary[0]
    except Exception as e:
        logging.error(f"Error in text summarization: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error processing summarization.")

@api_router.post("/translation/translate", response_model=TranslationOut)
def translate_text(payload: TextIn):
    """
    Translates text from English to French.
    Note: Current model only supports EN-FR.
    """
    try:
        if not MODELS_LOADED:
            raise HTTPException(status_code=503, detail="Models are not loaded yet. Please wait.")
        translation = models["translator"](payload.text)
        return TranslationOut(original_text=payload.text, translated_text=translation[0]['translation_text'])
    except Exception as e:
        logging.error(f"Error in language translation: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error processing translation.")

@api_router.post("/image/caption")
async def caption_image(file: UploadFile = File(...)):
    """
    Generates a descriptive caption for an uploaded image.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")
    try:
        if not MODELS_LOADED:
            raise HTTPException(status_code=503, detail="Models are not loaded yet. Please wait.")
        image = Image.open(io.BytesIO(await file.read()))
        caption = models["captioner"](image)
        return caption[0]
    except Exception as e:
        logging.error(f"Error in image captioning: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error processing image captioning.")

@api_router.post("/generation/generate")
def generate_text(payload: TextIn):
    """
    Generates text based on a given prompt.
    """
    try:
        if not MODELS_LOADED:
            raise HTTPException(status_code=503, detail="Models are not loaded yet. Please wait.")
        if not payload.text.strip():
            raise HTTPException(status_code=400, detail="Prompt cannot be empty.")
        generated = models["generator"](payload.text, max_length=100, num_return_sequences=1)
        return {"generated_text": generated[0]["generated_text"]}
    except Exception as e:
        logging.error(f"Error in text generation: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error processing text generation.")

@api_router.post("/tts")
async def text_to_speech(payload: TextIn):
    """
    Converts text to speech and returns an audio file.
    """
    try:
        if not MODELS_LOADED:
            raise HTTPException(status_code=503, detail="Models are not loaded yet. Please wait.")
        if not payload.text.strip():
            raise HTTPException(status_code=400, detail="Text for speech cannot be empty.")

        tts_output = models["tts"](payload.text)
        audio_array = tts_output["audio"]
        sampling_rate = tts_output["sampling_rate"]

        buffer = io.BytesIO()
        sf.write(buffer, audio_array.astype(np.float32), sampling_rate, format='wav')
        buffer.seek(0) # Rewind the buffer to the beginning

        return StreamingResponse(buffer, media_type="audio/wav", headers={"Content-Disposition": "attachment; filename=speech.wav"})
    except Exception as e:
        logging.error(f"Error in Text-to-Speech: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error processing Text-to-Speech.")

@api_router.post("/stt", response_model=TranscriptionOut)
async def speech_to_text(file: UploadFile = File(...)):
    """
    Transcribes an audio file to text.
    """
    if not file.content_type.startswith("audio/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an audio file.")
    try:
        if not MODELS_LOADED:
            raise HTTPException(status_code=503, detail="Models are not loaded yet. Please wait.")
        audio_bytes = await file.read()
        buffer = io.BytesIO(audio_bytes)

        # Load audio data using soundfile
        audio_data, current_sampling_rate = sf.read(buffer)

        # If audio_data is stereo, convert to mono by averaging channels
        if audio_data.ndim > 1:
            audio_data = np.mean(audio_data, axis=1)

        stt_input = {"sampling_rate": current_sampling_rate, "raw": audio_data}
        transcription = models["stt"](stt_input)
        
        return TranscriptionOut(transcribed_text=transcription["text"])
    except Exception as e:
        logging.error(f"Error in Speech-to-Text: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error processing Speech-to-Text.")

@api_router.post("/qa/answer", response_model=QuestionAnsweringOut)
def answer_question(payload: QuestionAnsweringIn):
    """
    Provides an answer to a question based on an optional context.
    (Currently uses mock data as no specific QA model is loaded)
    """
    try:
        # In a real scenario, you would use a QA model here, e.g.:
        # if not MODELS_LOADED:
        #     raise HTTPException(status_code=503, detail="Models are not loaded yet. Please wait.")
        # qa_model_output = models["qa_model"](question=payload.question, context=payload.context)
        # answer = qa_model_output['answer']
        # confidence = qa_model_output['score']
        # sources = ["Mock Source 1", "Mock Source 2"] # Replace with actual source extraction

        # Mock response for demonstration
        if payload.context:
            answer_text = f"Based on the provided context, the answer to '{payload.question}' is a mock response demonstrating context awareness."
        else:
            answer_text = f"For the question '{payload.question}', here is a general mock answer."

        return QuestionAnsweringOut(
            answer=answer_text,
            confidence=0.85, # Mock confidence
            sources=["Mock Source: Wikipedia", "Mock Source: Research Paper"] # Mock sources
        )
    except Exception as e:
        logging.error(f"Error in Question Answering: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Error processing Question Answering.")


app.include_router(api_router, prefix="/api")

@app.get("/")
def root():
    """
    Root endpoint for the API.
    """
    return {"message": "Welcome to the Professional AI Toolkit API. Visit /docs for details."}

if __name__ == "__main__":
    print("Starting Professional AI Toolkit Backend Server...")
    print("API documentation will be available at http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)
