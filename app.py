import streamlit as st
import torch
from auto_gptq import AutoGPTQForCausalLM
from langchain import HuggingFacePipeline
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.embeddings import HuggingFaceInstructEmbeddings
from transformers import AutoTokenizer, TextStreamer, pipeline

# Initialize Streamlit UI
st.title("PDF Chatbot")
question = st.text_input("Enter your question here:")

# Check for user input and execute the model
if st.button("Ask"):
    # Check if CUDA is available
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

    # Data loading
    pdf_directory = st.file_uploader("Upload PDF", type=["pdf"])
    if pdf_directory is None:
        st.error("Please upload a PDF file.")
        st.stop()

    with pdf_directory:
        # Check if the PDF file exists
        try:
            loader = PyPDFDirectoryLoader(pdf_directory)
            docs = loader.load()
        except Exception as e:
            st.error("An error occurred while loading the PDF file.")
            st.stop()

    embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-large", model_kwargs={"device": DEVICE})

    # Model loading
    model_name_or_path = "TheBloke/Llama-2-13B-chat-GPTQ"
    model_basename = "model"
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
    model = AutoGPTQForCausalLM.from_pretrained(model_name_or_path, revision="gptq-4bit-128g-actorder_True",
                                               model_basename=model_basename, use_safetensors=True,
                                               trust_remote_code=True, inject_fused_attention=False, device=DEVICE,
                                               quantize_config=None)

    # Pipeline setup
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    text_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=1024, temperature=0,
                             top_p=0.95, repetition_penalty=1.15, streamer=streamer)
    llm = HuggingFacePipeline(pipeline=text_pipeline, model_kwargs={"temperature": 0})

    # Generate response
    response = llm(question)
    st.write("Answer:", response)
