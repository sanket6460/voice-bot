import streamlit as st
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import torch
import soundfile as sf
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
import requests

# Set API Keys
openai_api_key = "sk_ecf5f58485d2ddf72b3964521080968225f8b98822c722d5"

# Device configuration
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Initialize Indic Parler TTS Model
tts_model = ParlerTTSForConditionalGeneration.from_pretrained("ai4bharat/indic-parler-tts").to(device)
tokenizer = AutoTokenizer.from_pretrained("ai4bharat/indic-parler-tts")
description_tokenizer = AutoTokenizer.from_pretrained(tts_model.config.text_encoder._name_or_path)

# LangChain Chat Model Initialization
llm = ChatOpenAI(temperature=0, model="gpt-4", openai_api_key=openai_api_key)

# Audio recording and saving functions
def record_audio(duration=5, samplerate=44100):
    st.write("Recording...")
    audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='int16')
    sd.wait()
    st.write("Recording complete!")
    return audio, samplerate

def save_audio(audio, samplerate, filename="output.wav"):
    wav.write(filename, samplerate, audio)

# Transcription using OpenAI Whisper
def transcribe_audio(filename):
    url = "https://api.openai.com/v1/audio/transcriptions"
    headers = {"Authorization": f"Bearer {openai_api_key}"}
    files = {"file": open(filename, "rb"), "model": (None, "whisper-1")}
    response = requests.post(url, headers=headers, files=files)
    if response.status_code == 200:
        return response.json().get("text", "")
    else:
        st.error("Error in transcription.")
        st.error(response.text)
        return ""

# TTS with ai4bharat/indic-parler-tts
def text_to_speech(prompt, description):
    description_input_ids = description_tokenizer(description, return_tensors="pt").to(device)
    prompt_input_ids = tokenizer(prompt, return_tensors="pt").to(device)

    generation = tts_model.generate(
        input_ids=description_input_ids.input_ids,
        attention_mask=description_input_ids.attention_mask,
        prompt_input_ids=prompt_input_ids.input_ids,
        prompt_attention_mask=prompt_input_ids.attention_mask
    )

    audio_arr = generation.cpu().numpy().squeeze()
    return audio_arr, tts_model.config.sampling_rate

def play_audio(audio_data, samplerate):
    sd.play(audio_data, samplerate=samplerate)
    sd.wait()

# Streamlit App
st.title("LangChain-Powered Voice Assistant with Indic Parler-TTS")

# Recording Section
duration = st.slider("Select recording duration (seconds):", min_value=1, max_value=10, value=5)
if st.button("Record"):
    audio, samplerate = record_audio(duration)
    save_audio(audio, samplerate)

    # Transcription
    st.write("Transcribing...")
    text = transcribe_audio("output.wav")
    if text:
        st.write(f"Transcribed Text: {text}")

        # ChatGPT Response via LangChain
        st.write("Getting ChatGPT response using LangChain...")
        messages = [
            SystemMessage(content="You are a helpful sales assistant."),
            HumanMessage(content=text)
        ]
        response = llm.invoke(messages)
        response_text = response.content
        st.write(f"ChatGPT Response: {response_text}")

        # Text-to-Speech
        st.write("Converting response to speech...")
        description = "A female speaker with a British accent delivers a slightly expressive and animated speech with a moderate speed and pitch. The recording is of very high quality, with the speaker's voice sounding clear and very close up."
        tts_audio, samplerate = text_to_speech(response_text, description)
        if tts_audio is not None:
            # Save the audio
            sf.write("indic_tts_out.wav", tts_audio, samplerate)

            # Play the response
            st.write("Playing the response...")
            play_audio(tts_audio, samplerate)
