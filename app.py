import streamlit as st
from tempfile import NamedTemporaryFile
from langchain.document_loaders import PyPDFLoader
from langchain_community.llms import CTransformers
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai
from langchain.chains.question_answering import load_qa_chain
GOOGLE_API_KEY = 'GET_YOUR_GEMINI_API_KEY_&_PASTE_HERE'

genai.configure(api_key=GOOGLE_API_KEY)
model = ChatGoogleGenerativeAI(model="gemini-pro",
                             temperature=0.3, google_api_key=GOOGLE_API_KEY)


def main():
    st.header("MindZen")
    st.title("Insurance Policy Extractor📄")

    uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

    if uploaded_file is not None:
        # Save the uploaded file to a temporary location
        with NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(uploaded_file.read())
            temp_file_path = temp_file.name

        loader = PyPDFLoader(temp_file_path)
        pages = loader.load()
        context = "\n".join(str(p.page_content) for p in pages[:30])
        print(context)

        st.write(f"Number of pages: {len(pages)}")
        prog_bar = st.progress(0, text="Reading PDF and Generating JSON...")

        #llm = CTransformers(model="llama-2-13b-chat.ggmlv3.q4_1.bin", model_type="llama", config={'max_new_tokens': 512, 'context_length': 2048, 'temperature': 0.01})

        template = """Extract these values as precisely as possible from the provided context:\n\nContext:\n{context}\n\nValues to be extracted: 
        
        - Insured Name/Policyholder
        - Company/Insurer
        - Issuing Office Address
        - Policy Vehicle(Example: Scooter, Car etc.)
        - Policy Number
        - Policy Date/Period of Insurance
        - Policy Expiry 
        - Date/Period of Insurance
        - Insured's Declared Value
        - Gross OD ₹
        - Gross TP ₹
        - Vehicle No/Registration No
        - Premium ₹
        - Receipt Number
        - Receipt Date  
         
        'Label': 'Value' 
        In JSON format
        
        Answer:
        """
        prompt_template = PromptTemplate(input_variables=["context"], template=template)
        #chain = LLMChain(model=model, prompt=prompt_template)
        stuff_chain = load_qa_chain(model, chain_type="stuff", prompt=prompt_template)

        prog_bar.progress(50, text="Reading PDF and Generating JSON...")

        stuff_answer = stuff_chain(
            {"input_documents": pages[1:]}, return_only_outputs=True
        )

        prog_bar.progress(100, text="JSON Generated")

        st.sidebar.header("Extracted entities:")

        st.sidebar.write(stuff_answer['output_text'])


if __name__ == "__main__":
    main()
