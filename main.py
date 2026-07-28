import os
from dotenv import load_dotenv
import streamlit as st
from langchain_core.messages import AIMessage,HumanMessage
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()
available_models={
    "Ling 3 Flash": "inclusionai/ling-3.0-flash:free",
    "Google Gemma 4-26b-a4b": "google/gemma-4-26b-a4b-it:free",
    "Nvidia Nemotron 3 Ultra": "nvidia/nemotron-3-ultra-550b-a55b:free",
    "Nvidia Nano Omni": "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free",
    "Nemotron 3 Super": "nvidia/nemotron-3-super-120b-a12b:free",
    "Cohere: North Mini Code": "cohere/north-mini-code:free",
    "PoolSide Laguna S2.1": "poolside/laguna-s-2.1:free",
    "PoolSide Laguna XS2.1": "poolside/laguna-xs-2.1:free",
    "OpenAI: gpt-oss-20b": "openai/gpt-oss-20b:free",
    "Auto Free Router": "openrouter/free",
}

embedding=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
@st.cache_resource
def load_vectorstore():
    assets_directory="./assets"
    persist_directory='./chromadb'

    if os.path.exists(persist_directory) and os.listdir(persist_directory):
        vectorstore=Chroma(persist_directory=persist_directory,embedding_function=embedding)
        return vectorstore
    if os.path.exists(assets_directory) and os.listdir(assets_directory):
        loader=PyPDFDirectoryLoader(assets_directory)
        document=loader.load()

        splitter=RecursiveCharacterTextSplitter(
            chunk_size=600,
            chunk_overlap=200
        )
        split_text=splitter.split_documents(document)

        vectorstore=Chroma.from_documents(
            documents=split_text,
            persist_directory=persist_directory,
            embedding=embedding
            )
        return vectorstore
    return None
vectorstore=load_vectorstore()

