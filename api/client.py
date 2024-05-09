import requests
import streamlit as st

# Call openai model api for generating an eassy
def get_openai_response(input_text):
    response = requests.post("http://localhost:8000/essay/invoke",
                             json={"input":{"topic":input_text}})
    
    return response.json()["output"]["content"]

# Call llama3 model api for generating an eassy
def get_llama3_response(input_text):
    response = requests.post("http://localhost:8000/poem/invoke",
                             json={"input":{"topic":input_text}})
    
    return response.json()["output"]

st.title("Langchain Demo with OPENAI and LLAMA3 models")
input_text_1 = st.text_input("Enter a topic for the essay")
input_text_2 = st.text_input("Enter a topic for the poem")

if input_text_1:
    st.write(get_openai_response(input_text_1))

if input_text_2:
    st.write(get_llama3_response(input_text_2))
