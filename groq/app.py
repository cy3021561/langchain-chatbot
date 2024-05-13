import streamlit as st
import os
import time
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
load_dotenv()

## Load the Groq API KEY
groq_api_key = os.environ["GROQ_API_KEY"]

# Initialize session state variables.
if "vectors" not in st.session_state:
    # Initialize the embeddings, loader, and documents if they don't exist.
    st.session_state.embeddings = OllamaEmbeddings() or st.session_state.embeddings
    st.session_state.loader = WebBaseLoader("https://docs.smith.langchain.com/") or st.session_state.loader
    st.session_state.docs = st.session_state.loader.load() or st.session_state.docs

    # Initialize the text splitter and split the documents into chunks.
    st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs)

    # Create a vector store from the final documents and embeddings.
    st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings) or st.session_state.vectors

# Set the title of the Streamlit app.
st.title("ChatGroq Demo")  

# Create an instance of the ChatGroq model with the Groq API key.
llm = ChatGroq(groq_api_key=groq_api_key, model_name="mixtral-8x7b-32768")

# Create a chat prompt template with a context and input.
prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question.
    <context>
    {context}
    </context>
    Questions:{input}
    """
)

# Create a document chain using the LLM and prompt.
document_chain = create_stuff_documents_chain(llm, prompt)

# Create a retriever from the vector store.
retriever = st.session_state.vectors.as_retriever()

# Create a retrieval chain using the retriever and document chain.
retrieval_chain = create_retrieval_chain(retriever, document_chain)

# Create a text input field for the user to enter their question.
input = st.text_input("Enter your question here.")

if input:
    # Measure the execution time of the retrieval chain.
    start = time.process_time()
    response = retrieval_chain.invoke({"input": input})
    print("Response time: ", time.process_time() - start)

    # Display the answer and context from the response.
    st.write(response["answer"])

    # Create an expander to display the document similarity search results.
    with st.expander("Document Similarity Search"):
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("-" * 20)