import streamlit as st
import os
import time
from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from dotenv import load_dotenv

load_dotenv()

## load the GROQ And OpenAI API KEY 
os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
groq_api_key = os.getenv('GROQ_API_KEY')

st.title("ChatGroq with Lllama3 Demo")

llm = ChatGroq(groq_api_key=groq_api_key,
               model_name="Llama3-8b-8192")

prompt = ChatPromptTemplate.from_template(
"""
Answer the questions based on the provided context only,
please provide the most accurate response for the question.
<context>
{context}
</context>
Question: {input}
"""
)



def vector_embedding():
    """
    Vector Embedding of the Documents
    """
    if "vectors" not in st.session_state:
        # Use st.session_state to store and access session-specific data across multiple user interactions within the same session. 
        # It provides a way to maintain state between different function calls or user interactions without having to use global variables or external storage.
        st.session_state.embeddings = OpenAIEmbeddings()
        st.session_state.loader = PyPDFDirectoryLoader("./us_census")
        st.session_state.docs = st.session_state.loader.load() ## Document Loading
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.final_docs = st.session_state.text_splitter.split_documents(st.session_state.docs) ## Splitted to Chunk
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_docs, st.session_state.embeddings) ## Use OPENAI embedding

prompt1 = st.text_input("Enter Your Question Of the Documents")

if st.button("Documents Embedding"):
    vector_embedding()
    st.write("Vector Store DB is ready")

if prompt1:
    start = time.process_time()
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever()
    retriever_chain = create_retrieval_chain(retriever, document_chain)
    response = retriever_chain.invoke({"input": prompt1})
    print("Response time :", time.process_time() - start)
    st.write(response['answer'])

    # With a streamlit expander
    with st.expander("Document Similarity Search"):
        # Find the relevant chunks
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("--------------------------------")