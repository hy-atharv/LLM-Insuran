import streamlit as st
import torch
from auto_gptq import AutoGPTQForCausalLM
from langchain import HuggingFacePipeline, PromptTemplate
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from pdf2image import convert_from_path
from transformers import AutoTokenizer, TextStreamer, pipeline
import os
import gdown

# Check if CUDA is available
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# Download the PDF file using gdown
pdf_url = "https://drive.google.com/uc?id=1DnG_6LoXjn57oGGP5jfLvTxCRoRy87qz"
pdf_path = "pdfs/Insurance.pdf"
os.makedirs("pdfs", exist_ok=True)
gdown.download(pdf_url, pdf_path, quiet=False)

# Initialize Streamlit UI
st.title("PDF Chatbot")
question = st.text_input("Enter your question here:")

# Check for user input and execute the model
if st.button("Ask"):
    # Data loading
    pdf_directory = "pdfs"
    loader = PyPDFDirectoryLoader(pdf_directory)
    docs = loader.load()

    embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-large", model_kwargs={"device": DEVICE})

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=64)
    texts = text_splitter.split_documents(docs)

    # Model loading
    model_name_or_path = "TheBloke/Llama-2-13B-chat-GPTQ"
    model_basename = "model"
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
    model = AutoGPTQForCausalLM.from_quantized(
        model_name_or_path,
        revision="gptq-4bit-128g-actorder_True",
        model_basename=model_basename,
        use_safetensors=True,
        trust_remote_code=True,
        inject_fused_attention=False,
        device=DEVICE,
        quantize_config=None,
    )

    # Pipeline setup
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    text_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=1024, temperature=0,
                             top_p=0.95, repetition_penalty=1.15, streamer=streamer)
    llm = HuggingFacePipeline(pipeline=text_pipeline, model_kwargs={"temperature": 0})

    # Generate response
    response = llm(question)
    st.write("Answer:", response)
