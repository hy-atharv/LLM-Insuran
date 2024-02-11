import streamlit as st
import torch
from auto_gptq import AutoGPTQForCausalLM
from langchain import HuggingFacePipeline, PromptTemplate
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer, TextStreamer, pipeline

# Data loading
pdf_directory = "pdfs"
loader = PyPDFDirectoryLoader(pdf_directory)
docs = loader.load()
embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-large")

# Model loading
model_name_or_path = "TheBloke/Llama-2-13B-chat-GPTQ"
model_basename = "model"
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
model = AutoGPTQForCausalLM.from_pretrained(model_name_or_path, revision="gptq-4bit-128g-actorder_True",
                                           model_basename=model_basename)

# Pipeline setup
streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
text_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=1024, temperature=0,
                         top_p=0.95, repetition_penalty=1.15, streamer=streamer)
llm = HuggingFacePipeline(pipeline=text_pipeline, model_kwargs={"temperature": 0})

# Streamlit UI
st.title("PDF Chatbot")
question = st.text_input("Enter your question here:")

if st.button("Ask"):
    result = llm(question)
    st.write("Answer:", result)
