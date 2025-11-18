'''
%pip install openai==1.64.0 | tail -n 1
'''
from openai import OpenAI
from IPython import display

client = OpenAI()

response = client.images.generate(
    model="dall-e-2",
    prompt="A siamese cat on the beach",
    size="1024x1024",
    # quality="standard",
    n=1,
)

url = response.data[0].url
display.Image(url=url, width=512)

response2 = client.images.generate(
    model="dall-e-3",
    prompt="A siamese cat on the beach",
    size="1024x1792",
    quality="standard",
    n=1,
)