import os
import streamlit as st
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from gtts import gTTS
import tempfile
import PyPDF2
import base64
import sounddevice as sd
import numpy as np
import wave
import pydub
from pydub import AudioSegment
import speech_recognition as sr

# Load the API key from 'secret.txt'
with open('secret.txt') as f:
    api_key = f.read().strip()

# Ensure the API key is set in the environment
os.environ["GROQ_API_KEY"] = api_key

# Initialize ChatGroq with API key
llm = ChatGroq(
    model="llama3-8b-8192",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    api_key=api_key  # Pass the API key directly to the client
)

# Supported languages
LANGUAGES = {
    "English": "en",
    "French": "fr",
    "Spanish": "es",
    "German": "de",
    "Italian": "it",
    "Portuguese": "pt",
    "Russian": "ru",
    "Japanese": "ja",
    "Chinese": "zh",
    "Korean": "ko"
}

# Initialize session state for storing recorded text
if 'recorded_text' not in st.session_state:
    st.session_state.recorded_text = None

# Helper function to load images and convert to base64
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return None

# Function to clean text
def clean_text(text):
    return ' '.join(text.split())

# Record audio function using sounddevice
def record_audio(duration=5, samplerate=16000):
    st.write("Recording...")
    # Record audio
    audio_data = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='int16')
    sd.wait()  # Wait until recording is finished
    
    # Save the audio data to a .wav file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as wf:
        wav_file = wave.open(wf, 'wb')
        wav_file.setnchannels(1)  # Mono channel
        wav_file.setsampwidth(2)  # 16-bit samples
        wav_file.setframerate(samplerate)
        wav_file.writeframes(audio_data.tobytes())
        audio_path = wf.name

    return audio_path

# Function to transcribe audio to text
def transcribe_audio(audio_path):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio = recognizer.record(source)
    try:
        return recognizer.recognize_google(audio)
    except sr.UnknownValueError:
        st.error("Could not understand audio")
    except sr.RequestError as e:
        st.error(f"Error with transcription service: {e}")
    return None

# Function to convert text to speech
def text_to_speech(text, language):
    try:
        tts = gTTS(text=text, lang=LANGUAGES[language])
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as fp:
            tts.save(fp.name)
            return fp.name
    except Exception as e:
        st.error(f"Error generating speech: {e}")
        return None

# Translation chain function
def translation_chain(source_lang, target_lang):
    prompt_template = PromptTemplate(
        input_variables=["text"],
        template=f"""
         You are a helpful translator. Translate the following text from {source_lang} to {target_lang}.
         Do not add any explanations, clarifications, or extra comments. Provide only the direct translation of the text.    
        Text to translate:
        {{text}}
        """,
    )
    return LLMChain(llm=llm, prompt=prompt_template)

# Function to get user input text
def get_input_text():
    st.title("Text Translation with Voice")

    input_method = st.radio("Choose input method:", ["Upload File", "Type Text", "Record Voice"])

    if input_method == "Upload File":
        file_type = st.radio("Select file type:", ["Text File (.txt)", "PDF File (.pdf)"])
        if file_type == "Text File (.txt)":
            uploaded_file = st.file_uploader("Choose a text file", type="txt")
            if uploaded_file:
                return uploaded_file.read().decode("utf-8")
        else:  # PDF File
            uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
            if uploaded_file:
                text = extract_text_from_pdf(uploaded_file)
                if text:
                    cleaned_text = clean_text(text)
                    st.text_area("You can edit the text before translation:", cleaned_text, height=200)
                    return cleaned_text

    elif input_method == "Type Text":
        return st.text_area("Enter text:", height=200)

    elif input_method == "Record Voice":
        duration = st.slider("Recording Duration (seconds)", 1, 10, 5)
        if st.button("Record"):
            audio_path = record_audio(duration)
            st.audio(audio_path, format='audio/wav')
            st.write("Transcribing audio...")
            transcribed_text = transcribe_audio(audio_path)
            if transcribed_text:
                st.session_state.recorded_text = transcribed_text
                st.write("Transcribed Text:")
                st.text_area("Edit the transcribed text before translation:", transcribed_text, height=200)

    # Return recorded or uploaded text
    return st.session_state.recorded_text

# Main app logic
def main():
    input_text = get_input_text()

    if input_text:
        st.write("Select languages for translation:")
        col1, col2 = st.columns(2)
        with col1:
            source_language = st.selectbox("From:", list(LANGUAGES.keys()))
        with col2:
            target_language = st.selectbox("To:", [lang for lang in LANGUAGES.keys() if lang != source_language])

        if st.button("Translate"):
            with st.spinner("Translating..."):
                chain = translation_chain(source_language, target_language)
                translated_text = chain.run({"text": input_text})

            st.subheader("Translation:")
            st.text_area("Translated Text:", translated_text, height=200)

            with st.spinner("Generating audio..."):
                audio_file = text_to_speech(translated_text, target_language)
                if audio_file:
                    st.audio(audio_file, format='audio/mp3')

                # Download options
                col1, col2 = st.columns(2)
                with col1:
                    st.download_button(
                        label="Download Translation (Text)",
                        data=translated_text,
                        file_name=f"translation_{LANGUAGES[source_language]}_{LANGUAGES[target_language]}.txt",
                        mime="text/plain"
                    )
                with col2:
                    if audio_file:
                        with open(audio_file, 'rb') as f:
                            st.download_button(
                                label="Download Translation (Audio)",
                                data=f,
                                file_name=f"translation_{LANGUAGES[source_language]}_{LANGUAGES[target_language]}.mp3",
                                mime="audio/mp3"
                            )

if __name__ == "__main__":
    main()
