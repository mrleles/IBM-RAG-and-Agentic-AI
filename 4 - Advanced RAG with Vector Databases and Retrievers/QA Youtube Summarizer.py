# Import necessary libraries for the YouTube bot
import gradio as gr
import re  #For extracting video id 
from youtube_transcript_api import YouTubeTranscriptApi  # For extracting transcripts from YouTube videos
from langchain.text_splitter import RecursiveCharacterTextSplitter  # For splitting text into manageable segments
from ibm_watsonx_ai.foundation_models.utils.enums import ModelTypes  # For specifying model types
from ibm_watsonx_ai import APIClient, Credentials  # For API client and credentials management
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams  # For managing model parameters
from ibm_watsonx_ai.foundation_models.utils.enums import DecodingMethods  # For defining decoding methods
from langchain_ibm import WatsonxLLM, WatsonxEmbeddings  # For interacting with IBM's LLM and embeddings
from ibm_watsonx_ai.foundation_models.utils import get_embedding_model_specs  # For retrieving model specifications
from ibm_watsonx_ai.foundation_models.utils.enums import EmbeddingTypes  # For specifying types of embeddings
from langchain_community.vectorstores import FAISS  # For efficient vector storage and similarity search
from langchain.chains import LLMChain  # For creating chains of operations with LLMs
from langchain.prompts import PromptTemplate  # For defining prompt templates

def get_video_id(url):
    pattern = r'https:\/\/www\.youtube\.com\/watch\?v=([a-zA-Z0-9_-]{11})'
    match = re.search(pattern, url)
    return match.group(1) if match else None

url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
video_id = get_video_id(url)
print(video_id)

def get_transcript(url):
    video_id = get_video_id(url)

    ytt_api = YouTubeTranscriptApi()

    transcripts = ytt_api.list(video_id)

    transcript = ""
    for t in transcripts:
        if t.language_code =='en':
            if t.is_generated:
                if len(transcript) == 0:
                    transcript = t.fetch()
            else:
                transcript = t.fetch()
                break
    return transcript if transcript else None

def process(transcript):
    txt = ""
    for i in transcript:
        try:
            txt += f"Text: {i.text} Start: {i.start}\n"
        except KeyError:
            pass
    return txt

def chunk_transcript(processed_transcript, chunk_size=200, chunk_overlap=20):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    chunks = text_splitter.split_text(processed_transcript)
    return chunks

def setup_credentials():
    # Define the model ID for the WatsonX model being used
    model_id = "meta-llama/llama-3-3-70b-instruct"
    
    # Set up the credentials by specifying the URL for IBM Watson services
    credentials = Credentials(url="https://us-south.ml.cloud.ibm.com")
    
    # Create an API client using the credentials
    client = APIClient(credentials)
    
    # Define the project ID associated with the WatsonX platform
    project_id = "skills-network"
    
    # Return the model ID, credentials, client, and project ID for later use
    return model_id, credentials, client, project_id

def define_parameters():
    return {
        GenParams.DECODING_METHOD: DecodingMethods.GREEDY,
        GenParams.MAX_NEW_TOKENS: 900,
    }

def initialize_watsonx_llm(model_id, credentials, project_id, parameters):
    return WatsonxLLM(
        model_id=model_id,
        url=credentials.get("url"),
        project_id=project_id,
        params=parameters
    )

def setup_embedding_model(credentials, project_id):
    return WatsonxEmbeddings(
        model_id='ibm/slate-30m-english-rtrvr-v2',
        url=credentials["url"],
        project_id=project_id
    )

def create_faiss_index(chunks, embedding_model):
    return FAISS.from_texts(chunks, embedding_model)

def perform_similarity_search(faiss_index, query, k=3):
    results = faiss_index.similarity_search(query, k=k)
    return results

def create_summary_prompt():
    template = """
    <|begin_of_text|><|start_header_id|>system<|end_header_id|>
    You are an AI assistant tasked with summarizing YouTube video transcripts. Provide concise, informative summaries that capture the main points of the video content.

    Instructions:
    1. Summarize the transcript in a single concise paragraph.
    2. Ignore any timestamps in your summary.
    3. Focus on the spoken content (Text) of the video.

    Note: In the transcript, "Text" refers to the spoken words in the video, and "start" indicates the timestamp when that part begins in the video.<|eot_id|><|start_header_id|>user<|end_header_id|>
    Please summarize the following YouTube video transcript:

    {transcript}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
    """

    prompt = PromptTemplate(
        input_variables=["transcript"],
        template=template
    )

    return prompt

def create_summary_chain(llm, prompt, verbose=True):
    return LLMChain(llm=llm, prompt=prompt, verbose=verbose)

def retrieve(query, faiss_index, k=7):
    relevant_context = faiss_index.similarity_search(query, k=k)
    return relevant_context

from langchain import PromptTemplate

def create_qa_prompt_template():
    qa_template = """
    You are an expert assistant providing detailed answers based on the following video content.
    Relevant Video Context: {context}
    Based on the above context, please answer the following question:
    Question: {question}
    """

    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template=qa_template
    )

    return prompt_template

def create_qa_chain(llm, prompt_template, verbose=True):
    return LLMChain(llm=llm, prompt=prompt_template, verbose=verbose)

def generate_answer(question, faiss_index, qa_chain, k=7):
    relevant_context = retrieve(question, faiss_index, k=k)
    answer = qa_chain.predict(context=relevant_context, question=question)

processed_transcript = ""

def summarize_video(video_url):
    global fetched_transcript, processed_transcript

    if video_url:
        fetched_transcript = get_transcript(video_url)
        processed_transcript = process(fetched_transcript)
    else:
        return "Please provide a valid YouTube URL."
    
    if processed_transcript:
        model_id, credentials, client, project_id = setup_credentials()

        llm = initialize_watsonx_llm(model_id, credentials, project_id, define_parameters())

        summary_prompt = create_summary_prompt()
        summary_chain = create_summary_chain(llm, summary_prompt)

        summary = summary_chain.run({"transcript": processed_transcript})
        return summary
    else:
        return "No transcript available. Please fetch the transcript first."
    
def answer_question(video_url, user_question):
    global fetched_transcript, processed_transcript

    if not processed_transcript:
        if video_url:
            fetched_transcript = get_transcript(video_url)
            processed_transcript = process(fetched_transcript)
        else:
            return "Please provide a valid YouTube URL."
        
    if processed_transcript and user_question:
        chunks = chunk_transcript(processed_transcript)
        model_id, credentials, client, project_id = setup_credentials()
        llm = initialize_watsonx_llm(model_id, credentials, project_id, define_parameters())
        embedding_model = setup_embedding_model(credentials, project_id)
        faiss_index = create_faiss_index(chunks, embedding_model)
        qa_prompt = create_qa_prompt_template()
        qa_chain = create_qa_chain(llm, qa_prompt)

        answer = generate_answer(user_question, faiss_index, qa_chain)
        return answer
    else:
        return "Please provide a valid question and ensure the transcript has been fetched."
    
with gr.Blocks() as interface:
    video_url = gr.Textbox(label="Youtube video url", placeholder="Enter the youtube video url")
    summary_output = gr.Textbox(label="Video summary", lines=5)
    question_input = gr.Textbox(label="Ask a question about the video", placeholder="Ask your question")
    answer_output = gr.Textbox(label="Answer to your question", lines=5)

    summarize_btn = gr.Button("Summarize video")
    question_btn = gr.Button("Ask a question")

    transcript_status = gr.Textbox(label="Transcript Status", interactive=False)

    summarize_btn.click(summarize_video, inputs=video_url, outputs=summary_output)
    question_btn.click(answer_question, inputs=[video_url, question_input], outputs=answer_output)

interface.launch(server_name="0.0.0.0", server_port=7860)