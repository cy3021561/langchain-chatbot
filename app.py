import streamlit as st
import os
import time
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama
from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.document_loaders import WebBaseLoader
from langchain.tools.retriever import create_retriever_tool
from langchain_community.utilities import ArxivAPIWrapper
from langchain_community.tools import ArxivQueryRun
from langchain.agents import create_openai_tools_agent, create_tool_calling_agent
from langchain.agents import AgentExecutor
from langchain import hub
import tempfile
from dotenv import load_dotenv

load_dotenv()

## load the GROQ And OpenAI API KEY 
os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")
os.environ['LANGCHAIN_PROJECT'] = os.getenv("LANGCHAIN_PROJECT")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
groq_api_key = os.getenv('GROQ_API_KEY')

## Create tools
# Wiki tool
wiki_api_wrapper = WikipediaAPIWrapper(top_k_results=3)
wiki_tool = WikipediaQueryRun(api_wrapper=wiki_api_wrapper)

# Web page tool
loader = WebBaseLoader("https://aman.ai/primers/ai/RAG/")
docs = loader.load()
documents = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(docs)
web_vectordb = FAISS.from_documents(documents, OpenAIEmbeddings())
web_retriever = web_vectordb.as_retriever()
web_page_tool = create_retriever_tool(web_retriever, "RAG", "Search for information about RAG. For any questions about Retrieval Augmented Generation, you must use this tool!")

# Arxiv tool
arxiv_wrapper = ArxivAPIWrapper(top_k_results=3)
arxiv_tool = ArxivQueryRun(api_wrapper=arxiv_wrapper)

llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)
#llm = ChatOllama(model="llama3")

# Update the session state with the new prompt template
st.session_state.prompt = hub.pull("hwchase17/openai-functions-agent")

st.title("Q&A Chatbot With LangChain API")
st.subheader("Agent's tool list -- Wiki, Arxiv, WebPage, PDF")
st.session_state.tools = [wiki_tool, arxiv_tool, web_page_tool]
st.session_state.agent = create_openai_tools_agent(llm, st.session_state.tools, st.session_state.prompt)
st.session_state.agent_executor = AgentExecutor(agent=st.session_state.agent, tools=st.session_state.tools, verbose=True)

# Add customized tool
def input_pdf_vector_embedding(temp_file_path):
    """
    Vector Embedding of the Documents
    """
    st.session_state.custom_loader = PyPDFLoader(temp_file_path)
    st.session_state.custom_docs = st.session_state.custom_loader.load()
    st.session_state.chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(st.session_state.custom_docs)
    st.session_state.custom_vectordb = FAISS.from_documents(st.session_state.chunks, OpenAIEmbeddings())
    st.session_state.custom_retriever = st.session_state.custom_vectordb.as_retriever()
    st.session_state.custom_tool = create_retriever_tool(st.session_state.custom_retriever, "uploaded_file", "Search for information in the uploaded file. \
                                          For any questions about the uploaded file, you must use this tool!")
    st.session_state.tools = [wiki_tool, arxiv_tool, web_page_tool, st.session_state.custom_tool]
    st.session_state.agent = create_openai_tools_agent(llm, st.session_state.tools, st.session_state.prompt)
    st.session_state.agent_executor = AgentExecutor(agent=st.session_state.agent, tools=st.session_state.tools, verbose=True)
    return

def clear_text():
    st.session_state.text_input = ""

# Add user input section
prompt1 = st.text_input("Enter Your Question Here", key='text_input')

# Add a file uploader widget
uploaded_file = st.file_uploader("Upload Your PDF File", type=["pdf"], on_change=clear_text)

if uploaded_file is not None:
    # Save the uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(uploaded_file.read())
        temp_file_path = temp_file.name

    # Process the uploaded PDF
    input_pdf_vector_embedding(temp_file_path)
    uploaded_file = None  # This line doesn't actually clear the uploader, it's for logic clarity
    st.write("Customized VectorDB is ready.")

if prompt1:
    start = time.process_time()
    response = st.session_state.agent_executor.invoke({"input": prompt1})
    print(response)
    print("Response time :", time.process_time() - start)
    st.write(response["output"])