import os
import streamlit as st
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from gtts import gTTS
import speech_recognition as sr
import tempfile
import PyPDF2
import io

# Access your secret key
with open('secret.txt') as f:
    key = f.read().strip()

# Check for GROQ_API_KEY and set it if not present
if "GROQ_API_KEY" not in os.environ:
    os.environ["GROQ_API_KEY"] = key

# Initialize session state for storing recorded text
if 'recorded_text' not in st.session_state:
    st.session_state.recorded_text = None

# Initialize ChatGroq model with parameters
llm = ChatGroq(
    model="llama3-8b-8192",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
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

# Function to convert speech to text
def speech_to_text():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("🎤 Listening... Speak now!")
        try:
            audio = r.listen(source, timeout=5)
            st.info("Processing speech...")
            text = r.recognize_google(audio)
            st.session_state.recorded_text = text
            st.success("Voice recorded successfully!")
            return text
        except sr.WaitTimeoutError:
            st.error("No speech detected within timeout period")
            return None
        except sr.UnknownValueError:
            st.error("Could not understand audio")
            return None
        except sr.RequestError as e:
            st.error(f"Could not request results; {e}")
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

# Task 1: Get input text (file upload, typing, or voice)
def clean_text(text):
    # Remove extra spaces and newlines
    cleaned_text = ' '.join(text.split())
    return cleaned_text

def get_input_text():
    st.title("Text Translation with Voice")
    
    input_method = st.radio("Choose input method:", ["Upload File", "Type Text", "Voice Input"])
    
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
                    st.subheader("Extracted text from PDF:")
                    st.text_area("You can edit the text before translation:", cleaned_text, height=200, key="pdf_text")
                    return st.session_state.pdf_text
    
    elif input_method == "Type Text":
        return st.text_area("Enter text:", height=200)
    
    else:  # Voice Input
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Start Recording"):
                text = speech_to_text()
                if text:
                    st.write("Transcribed text:")
                    st.write(text)
                    return text
        with col2:
            if st.session_state.recorded_text:
                if st.button("Clear Recording"):
                    st.session_state.recorded_text = None
                    st.rerun()
        
        # Display previously recorded text
        if st.session_state.recorded_text:
            st.write("Currently recorded text:")
            st.write(st.session_state.recorded_text)
            return st.session_state.recorded_text
    
    return None

# Translation chain function
def translation_chain(source_lang, target_lang):
    prompt_template = PromptTemplate(
        input_variables=["text"],
        template=f"""
        You are a helpful translator. Translate the following text from {source_lang} to {target_lang}.
        Rules:
        1. Maintain the original meaning and tone
        2. Keep any special formatting and punctuation
        3. If there are any culturally specific terms, provide appropriate translations
        4. For names, keep them in their original form unless there's a widely accepted translation
        
        Provide only the translation, no explanations.
        Text to translate:
        {{text}}
        """,
    )
    return LLMChain(llm=llm, prompt=prompt_template)

# Main app logic
def main():
    # Get input text
    input_text = get_input_text()
    
    # Proceed with translation if we have input text
    if input_text or st.session_state.recorded_text:
        st.write("Select languages for translation:")
        col1, col2 = st.columns(2)
        with col1:
            source_language = st.selectbox("From:", list(LANGUAGES.keys()))
        with col2:
            target_language = st.selectbox("To:", [lang for lang in LANGUAGES.keys() if lang != source_language])

        # Use either input_text or recorded_text
        text_to_translate = input_text if input_text else st.session_state.recorded_text

        if st.button("Translate"):
            if text_to_translate and text_to_translate.strip():
                with st.spinner("Translating..."):
                    chain = translation_chain(source_language, target_language)
                    translated_text = chain.run({"text": text_to_translate})

                st.subheader("Translation:")
                st.text_area("Translated Text:", translated_text, height=200)

                # Generate audio for translated text
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
            else:
                st.warning("Please provide text to translate.")

    st.markdown("---")
    st.markdown("""<div style='text-align: center; color: #666;'>Powered by LangChain, Groq LLM, and Voice Technologies</div>""", unsafe_allow_html=True)

if __name__ == "__main__":
    main()