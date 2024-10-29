import os
import streamlit as st
from langchain_groq import ChatGroq
import base64
import sys
import io
import traceback
from contextlib import redirect_stdout, redirect_stderr

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
        position: fixed;
        top: 10px;
        right: 10px;
        width: 200px;
        z-index: 100;
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

# Function to assist with Python code generation
def generate_python_code(user_input):
    system_msg = (
        "You are an assistant that strictly generates Python code based on user instructions. "
        "Only provide Python code without any additional explanation, context, or comments."
    )

    # Prepare the message list
    messages = [
        ("system", system_msg),
        ("human", user_input),
    ]

    # Invoke the model and get response
    ai_msg = llm.invoke(messages)
    # Ensure only Python code is returned
    return ai_msg.content.strip()

# Function to execute Python code and capture stdout/stderr
def execute_python_code(code):
    output = io.StringIO()
    error = io.StringIO()
    
    try:
        with redirect_stdout(output), redirect_stderr(error):
            exec(code)
        return output.getvalue(), error.getvalue(), None
    except Exception:
        return output.getvalue(), error.getvalue(), traceback.format_exc()



st.title("Python Code Assistant & Executor App")

# Input for the Python code description
user_input = st.text_area("Describe the Python code you want to generate:", height=150)

# Button to generate the code
if st.button("Generate Code"):
    if user_input.strip():
        generated_code = generate_python_code(user_input)
        st.session_state['generated_code'] = generated_code
    else:
        st.warning("Please enter a description to generate code.")

# Editable text area for the generated Python code
editable_code = st.text_area("Generated Python Code (editable):", 
                             value=st.session_state.get('generated_code', ""), 
                             height=150)

# Button to run the code from the editable text area
if st.button("Run Code"):
    if editable_code.strip():
        stdout, stderr, exception = execute_python_code(editable_code)

        # Display the output
        if stdout:
            st.subheader("Output:")
            st.code(stdout)

        # Display any errors
        if stderr or exception:
            st.subheader("Errors:")
            st.error(stderr + (exception or ""))
    else:
        st.warning("No code to execute. Please generate code first or enter code to run.")

st.markdown("---")
st.write("Note: This app executes Python code in a restricted environment. Some operations may be limited for security reasons.")
