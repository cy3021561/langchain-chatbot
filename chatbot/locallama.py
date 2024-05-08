from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
import streamlit as st
import os
from dotenv import load_dotenv

# Loading environment variables
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
## LangSmith tracking
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

# Define prompt template in form of list
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Please response to the user queries."),
        ("user", "Question:{question}")
    ]
)

# Streamlit framework
st.title("Langchain Demo With Ollama API")
input_text=st.text_input("Search the topic you want")

# Ollama llama3 LLM
llm = Ollama(model="llama3")
output_parser = StrOutputParser()
chain = prompt | llm | output_parser

if input_text:
    st.write(chain.invoke({'question': input_text}))