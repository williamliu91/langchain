import os
import streamlit as st
from langchain_groq import ChatGroq
import base64

# Function to load the image and convert it to base64
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Path to the locally stored QR code image
qr_code_path = "qrcode.png"  # Ensure the image is in your app directory

# Convert image to base64
qr_code_base64 = get_base64_of_bin_file(qr_code_path)

# Custom CSS to position the QR code close to the top-right corner under the "Deploy" area
st.markdown(
    f"""
    <style>
    .qr-code {{
        position: fixed;  /* Keeps the QR code fixed in the viewport */
        top: 10px;       /* Sets the distance from the top of the viewport */
        right: 10px;     /* Sets the distance from the right of the viewport */
        width: 200px;    /* Adjusts the width of the QR code */
        z-index: 100;    /* Ensures the QR code stays above other elements */
    }}
    </style>
    <img src="data:image/png;base64,{qr_code_base64}" class="qr-code">
    """,
    unsafe_allow_html=True
)


# Access your secret file
with open('secret.txt') as f:
    key = f.read().strip()

# Check for GROQ_API_KEY and prompt if not found
if "GROQ_API_KEY" not in os.environ:
    os.environ["GROQ_API_KEY"] = key


# Initialize ChatGroq model with parameters
llm = ChatGroq(
    model="llama3-8b-8192",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

# Define a generic function to handle different tasks
def perform_task(task_type, user_input):
    if task_type == "translate_to_french":
        system_msg = "You are a helpful assistant that translates English to French. Translate the user sentence."
    elif task_type == "summarize":
        system_msg = "You are a helpful assistant that summarizes the text provided by the user."
    elif task_type == "sentiment_analysis":
        system_msg = "You are a helpful assistant that determines the sentiment of the user's sentence."
    else:
        raise ValueError("Unsupported task type.")

    # Prepare the message list based on the task
    messages = [
        ("system", system_msg),
        ("human", user_input),
    ]

    # Invoke the model and get response
    ai_msg = llm.invoke(messages)
    return ai_msg.content

# Streamlit UI
st.title("LangChain Text Processing App")

# Select task type
task_type = st.selectbox("Select Task Type:", 
                          ["translate_to_french", "summarize", "sentiment_analysis"])

# Input area for long text
user_input = st.text_area("Enter your text here:", height=300)

# Button to submit the input
if st.button("Process Text"):
    if user_input.strip():
        result = perform_task(task_type, user_input)
        st.subheader("Result:")
        st.write(result)
    else:
        st.warning("Please enter some text to process.")
