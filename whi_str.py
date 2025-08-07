import streamlit as st
import whisper
import tempfile
import os

# 🆕 Add this import
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate

# Page setup
st.set_page_config(page_title="Whisper STT", page_icon="🎤", layout="centered")

# CSS Styling
st.markdown("""
    <style>
    body {
        background-color: #fff0f5;
    }
    .title {
        color: #ff69b4;
        font-size: 48px;
        text-align: center;
        font-weight: bold;
    }
    .subtitle {
        color: #FFD700;
        font-size: 24px;
        text-align: center;
        margin-bottom: 20px;
    }
    .uploadbox {
    background-color: #fff;
    border-radius: 20px;
    padding: 20px;
    box-shadow: 0px 0px 10px #ffbfd9;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('''
    <div style="text-align:center;">
        <span style="font-size:48px; font-weight:bold; color:black;">Whisper Speech-to-Text</span>
    </div>
''', unsafe_allow_html=True)
st.markdown('<div class="subtitle" style="color:#E1AD01;">Speech2Spec 🎙️</div>', unsafe_allow_html=True)

st.markdown(
    "Upload an **audio file** and get transcription using OpenAI's Whisper model. Supports `.mp3`, `.wav`, `.m4a`")

# Microphone SVG
st.markdown('''
    <div style="text-align:center;">
        <svg width="150" height="150" viewBox="0 0 24 24" fill="#ff69b4" xmlns="http://www.w3.org/2000/svg">
            <path d="M12 14a3 3 0 0 0 3-3V5a3 3 0 1 0-6 0v6a3 3 0 0 0 3 3Zm5-3a5 5 0 1 1-10 0H5a7 7 0 0 0 14 0h-2Zm-5 9a9 9 0 0 0 9-9h-2a7 7 0 1 1-14 0H3a9 9 0 0 0 9 9Zm-1-2h2v2h-2v-2Z"/>
        </svg>
    </div>
''', unsafe_allow_html=True)

# Upload audio
audio_file = st.file_uploader("🎧 Upload audio file here", type=["mp3", "wav", "m4a"])

# Load Whisper model
@st.cache_resource
def load_model():
    return whisper.load_model("base")

model = load_model()

# Process file
if audio_file:
    st.audio(audio_file, format='audio/mp3')

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
        tmp.write(audio_file.read())
        tmp_path = tmp.name

    with st.spinner("✨ Transcribing... please wait..."):
        # Auto language detection and transcription
        result = model.transcribe(tmp_path, task="transcribe")
        os.remove(tmp_path)
        st.success("✅ Transcription complete!")

        # Detected Language
        lang_detected = result.get("language", "unknown").lower()
        st.markdown("### 🌐 Detected Language")
        st.markdown(f"""
        <div style="background-color:#e8f0fe; padding:10px; border-radius:10px; border: 1px solid #1a73e8;">
            <strong>{lang_detected.capitalize()}</strong>
        </div>
        """, unsafe_allow_html=True)

        # Transcription
        original_text = result['text']

        # 🆕 Transliterate if Marathi or Hindi
        if lang_detected in ["marathi", "hindi"]:
            converted_text = transliterate(original_text, sanscript.ITRANS, sanscript.DEVANAGARI)
        else:
            converted_text = original_text

        st.markdown("### 📝 Transcription")
        st.markdown(f"""
        <div style="background-color:#fff0f5; padding:15px; border-radius:10px; border: 1px solid #ff66b2;">
            {converted_text}
        </div>
        """, unsafe_allow_html=True)
