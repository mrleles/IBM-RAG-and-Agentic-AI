'''
# installing required libraries in my_env
pip install transformers==4.35.2 \
torch==2.1.1 \
gradio==5.9.0 \
langchain==0.3.12 \
langchain-community==0.3.12 \
langchain_ibm==0.3.5 \
ibm-watsonx-ai==1.1.16 \
pydantic==2.10.3

sudo apt update
sudo apt install ffmpeg -y
'''

import torch
from transformers import pipeline
import gradio as gr

def transcript_audio(audio_file):
    pipe = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-tiny.en",
        chunk_length_s=30,
    )

    result = pipe(audio_file, batch_size=8)["text"]
    return result

audio_input = gr.Audio(sources="upload", type="filepath")
output_text = gr.Textbox()

iface = gr.Interface(fn=transcript_audio,
inputs=audio_input, outputs=output_text,
title="Audio Transcription App",
description="Upload the audio file")

iface.launch(server_name="0.0.0.0", server_port= 5000)