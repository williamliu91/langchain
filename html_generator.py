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

# Define a generic function to handle different tasks
def perform_task(task_type, user_input):
    if task_type == "write_html":
        system_msg = "You are a helpful assistant that writes HTML code based on the user's description. The output should be HTML only."
    else:
        raise ValueError("Unsupported task type.")
    
    # Prepare the message list based on the task
    messages = [
        ("system", system_msg),
        ("human", user_input),
    ]
    
    # Invoke the model and get the response
    ai_msg = llm.invoke(messages)
    return ai_msg.content

# Streamlit UI
st.title("LangChain Text Processing App")

# Move input controls to sidebar
with st.sidebar:
    st.header("Input Controls")
    task_type = st.selectbox("Select Task Type:", ["write_html"])
    user_input = st.text_area("Enter your text here:", height=300)
    process_button = st.button("Process Text")

# Main page content
if process_button:
    if user_input.strip():
        result = perform_task(task_type, user_input)
        
        # Display the HTML code
        st.subheader("Generated HTML Code:")
        st.code(result, language="html")
        
        # Add download button for the HTML
        st.download_button(
            label="Download HTML",
            data=result,
            file_name="output.html",
            mime="text/html"
        )
        
        # Display the rendered HTML
        st.subheader("Preview:")
        st.components.v1.html(result, height=600, scrolling=True)
    else:
        st.warning("Please enter some text in the sidebar to process.")
else:
    st.info("👈 Enter your text in the sidebar and click 'Process Text' to generate HTML.")