import streamlit as st
from pagepilot_gptq import process_question  # Import your function from pagepilot_gptq.py

def main():
    # Set page title and description
    st.title("Simple AI Interface")
    st.markdown("This is a simple AI interface powered by Streamlit and Google Colab.")

    # Get user input
    question = st.text_input("Enter your question:")

    # Process user input
    if st.button("Submit"):
        if question:
            # Process the question using your Colab code
            response = process_question(question)
            # Display the response
            st.write("Response:", response)
        else:
            st.error("Please enter a question.")

if __name__ == "__main__":
    main()
