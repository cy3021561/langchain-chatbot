import requests
import streamlit as st

# Importing necessary libraries:
# - requests: A Python library used for making HTTP requests
# - streamlit: A Python library used for building interactive, web-based applications

# Function to call OpenAI model API for generating an essay
def get_openai_response(input_text):
    # Sending a POST request to the OpenAI model API
    response = requests.post("http://localhost:8000/essay/invoke",
                             json={"input":{"topic":input_text}})
    
    # Returning the generated essay content from the API response
    return response.json()["output"]["content"]

# Function to call Llama3 model API for generating a poem
def get_llama3_response(input_text):
    # Sending a POST request to the Llama3 model API
    response = requests.post("http://localhost:8000/poem/invoke",
                             json={"input":{"topic":input_text}})
    
    # Returning the generated poem from the API response
    return response.json()["output"]

# Creating a Streamlit web application with a title
st.title("Langchain Demo with OPENAI and LLAMA3 models")

# Adding two text input fields for users to enter topics for the essay and poem
input_text_1 = st.text_input("Enter a topic for the essay")
input_text_2 = st.text_input("Enter a topic for the poem")

# Displaying the generated essay when a topic is entered
if input_text_1:
    st.write(get_openai_response(input_text_1))

# Displaying the generated poem when a topic is entered
if input_text_2:
    st.write(get_llama3_response(input_text_2))