from fastapi import FastAPI  # Importing FastAPI for building the API server
from langchain.prompts import ChatPromptTemplate  # Importing ChatPromptTemplate for creating chat prompts
from langchain.chat_models import ChatOpenAI  # Importing ChatOpenAI for using OpenAI chat models
from langserve import add_routes  # Importing add_routes for adding routes to the API
import uvicorn  # Importing uvicorn for running the API server
import os  # Importing os for environment variable management
from langchain_community.llms import Ollama  # Importing Ollama for using LLaMA models
from dotenv import load_dotenv  # Importing load_dotenv for loading environment variables from a.env file

# Loading environment variables from a.env file
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")  # Setting OpenAI API key
os.environ["LANCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")  # Setting LangChain API key
## LangSmith tracking
os.environ["LANGCHAIN_TRACING_V2"] = "true"  # Enabling LangChain tracing v2


app = FastAPI(  # Creating a FastAPI application
    title="Langchain Server",  # Setting the title of the API
    version="1.0",  # Setting the version of the API
    description="A simple API server"  # Setting the description of the API
)


add_routes(  # Adding routes to the API for OpenAI chat models
    app,
    ChatOpenAI(),
    path="/openai"
)


model_1 = ChatOpenAI()  # Creating an instance of ChatOpenAI
model_2 = Ollama(model="llama3")  # Creating an instance of Ollama with the LLaMA 3 model


prompt_1 = ChatPromptTemplate.from_template("Write me an essay about {topic} with 100 words.")  # Creating a chat prompt for essays
prompt_2 = ChatPromptTemplate.from_template("Write me an poem about {topic} with 100 words.")  # Creating a chat prompt for poems


add_routes(  # Adding routes to the API for essay generation
    app,
    prompt_1 | model_1,
    path="/essay"
)


add_routes(  # Adding routes to the API for poem generation
    app,
    prompt_2 | model_2,
    path="/poem"
)


if __name__ == "__main__":  # Checking if the script is run directly
    uvicorn.run(app, host="localhost", port=8000)  # Running the API server using uvicorn