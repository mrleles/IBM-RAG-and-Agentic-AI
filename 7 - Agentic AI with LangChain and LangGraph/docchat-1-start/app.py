import gradio as gr
import hashlib
from typing import List, Dict
import os

from document_processor.file_handler import DocumentProcessor
from retriever.builder import RetrieverBuilder
from agents.workflow import AgentWorkflow
from config import constants, settings
from utils.logging import logger

# 1) Define some example data 
#    (i.e., question + paths to documents relevant to that question).
EXAMPLES = {
    "Google 2024 Environmental Report": {
        "question": "Retrieve the data center PUE efficiency values in Singapore 2nd facility in 2019 and 2022. Also retrieve regional average CFE in Asia pacific in 2023",
        "file_paths": ["examples/google-2024-environmental-report.pdf"]  
    },
    "DeepSeek-R1 Technical Report": {
        "question": "Summarize DeepSeek-R1 model's performance evaluation on all coding tasks against OpenAI o1-mini model",
        "file_paths": ["examples/DeepSeek Technical Report.pdf"]
    }
}

def main():
    processor = DocumentProcessor()
    retriever_builder = RetrieverBuilder()
    workflow = AgentWorkflow()

    # Define custom CSS for styling
    css = """
    .title {
        font-size: 1.5em !important;
        text-align: center !important;
        color: #FFD700;
    }

    .subtitle {
        font-size: 1em !important;
        text-align: center !important;
        color: #FFD700;
    }

    .text {
        text-align: center;
    }
    """

    js = """
    function createGradioAnimation() {
        var container = document.createElement('div');
        container.id = 'gradio=animation';
        container.style.fontSize = '2am';
        container.style.fontWeight = 'bold';
        container.style.textAlign = 'center';
        container.style.marginBottom = '20px';
        container.style.color = '#eba93f';

        var text = 'Welcome to DocChat!';

        for (var i =0; i < text.length; i++) {
            (function(i){
                setTimeout(function(){
                    var letter = document.createElement('span');
                    letter.style.opacity = '0';
                    letter.style.trasition = 'opacity 0.1s';
                    letter.innerText = text[i];

                    container.appendChild(letter);

                    setTimeout(function(){
                        letter.style.opacity = '0.9';
                    }, 50);
                }, i * 250);
            })(i);
        }

        var gradioContainer = document.querySelector('.gradio-container');
        gradioContainer.insertBefore(container, gradioContainer.firstChild);

        return 'Animation created';
    }
    """

    with gr.Blocks(theme.gr.themes.Citrus(), title="DocChat", css=css, js=js) as demo:
        gr.Markdown("## DocChat: powered by Docling and LangGraph", elem_classes="subtile")
        gr.Markdown("# How it works", elem_classes="title")
        gr.Markdown("Upload your document(s), enter your query then press Submit", elem_clasees="text")
        gr.Markdown("Or you can select one of the examples from the drop-down menu, select Load Example then Submit", elem_classes="text")
        gr.Markdown("**Note:** DocChat only accepts documents in these formats: '.pdf', '.docx', '.txt', '.md'", elem_classes="text")

        # 2) Maintain the session state for retrieving doc changes
        session_state = gr.State({
            "file_hashes": frozenset(),
            "retriever": None
        })

        # 3) Layout
        with gr.Row():
            with gr.Column():
                # Section for Examples
                gr.Markdown("### Example")
                example_dropdown = gr.Dropdown(
                    label="Select an Example",
                    choices=list(EXAMPLES.keys()),
                    value=None, # initially unselected
                )
                load_example_btn = gr.Button("Load Example")

                # Standard input components
                files = gr.Files(label="Upload Documents", file_types=constants.ALLOWED_TYPES)
                question = gr.Textbox(label="Question", lines=3)

                submit_btn = gr.Button("Submit")

            with gr.Column():
                answer_output = gr.Textbox(label=" Answer", interactive=False)
                verification_output = gr.Textbox(label="Verification Report")

        # 4) Helper function to load example into the UI

if __name__ == "__main__":
    main()