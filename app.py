import streamlit as st
import torch
from auto_gptq import AutoGPTQForCausalLM
from langchain import HuggingFacePipeline
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.embeddings import HuggingFaceInstructEmbeddings
from transformers import AutoTokenizer, TextStreamer, pipeline
import os

# Initialize Streamlit UI
st.title("PDF Chatbot")
question = st.text_input("Enter your question here:")
pdf_file = st.file_uploader("Upload PDF file", type=["pdf"])

# Check for user input and execute the model
if st.button("Ask") and pdf_file:
    # Check if CUDA is available
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

    # Data loading
    pdf_directory = "pdfs"
    if not os.path.exists(pdf_directory):
        os.makedirs(pdf_directory)
    pdf_path = os.path.join(pdf_directory, "document.pdf")
    with open(pdf_path, 'wb') as f:
        f.write(pdf_file.read())
    loader = PyPDFDirectoryLoader(pdf_path)
    docs = loader.load()
    embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-large", model_kwargs={"device": DEVICE})

    # Model loading
    model_name_or_path = "TheBloke/Llama-2-13B-chat-GPTQ"
    model_basename = "model"
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
    model = AutoGPTQForCausalLM.from_pretrained(model_name_or_path, model_basename=model_basename,
                                               use_safetensors=True, trust_remote_code=True,
                                               inject_fused_attention=False, device=DEVICE, quantize_config=None)

    # Pipeline setup
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    text_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=1024, temperature=0,
                             top_p=0.95, repetition_penalty=1.15, streamer=streamer)
    llm = HuggingFacePipeline(pipeline=text_pipeline, model_kwargs={"temperature": 0})

    # Generate response
    response = llm(question)
    st.write("Answer:", response)
