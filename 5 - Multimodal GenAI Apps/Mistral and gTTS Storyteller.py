'''
%pip install gTTS==2.5.4 | tail -n 1
%pip install ibm-watsonx-ai==1.1.20 | tail -n 1
'''

from ibm_watsonx_ai import Credentials
import os

credentials = Credentials(
    url="https://us-south.ml.cloud.ibm.com",
)

project_id = "skills-network"

from ibm_watsonx_ai import APIClient

client = APIClient(credentials)
client.foundation_models.TextModels

model_id = 'mistralai/mistral-medium-2505'

from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams

params = {
    GenParams.DECODING_METHOD: "greedy",
    GenParams.MAX_NEW_TOKENS: 5000, # 1000 in the course
}

model = ModelInference(
    model_id=model_id,
    credentials=credentials,
    project_id=project_id,
    params=params,
)

def generate_story(topic):
    prompt = f"""Write an engaging and educational story about {topic} for beginners.
    Use simple and clear language to explain basic concepts.
    Include interesting facts and keep it friendly and encouraging.
    The story should be around 200-300 words and end with a brief summary of what we learned.
    Make it perfect for someone just starting to learn about this topic."""

    response = model.generate_text(prompt=prompt)
    return response

topic = "the lifespan of trees"
story = generate_story(topic)

from gtts import gTTS
from IPython.display import Audio
import io

tts = gTTS(story)
audio_bytes = io.BytesIO()
tts.write_to_fp(audio_bytes)
audio_bytes.seek(0)
Audio(audio_bytes.read(), autoplay=False)
tts.save("generated_story.aac") # optional