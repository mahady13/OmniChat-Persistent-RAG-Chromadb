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

st.set_page_config(page_title="OmniChat AI with persistent RAG",page_icon="💬",layout="centered")

st.title("OmniChat AI with persistent RAG")

with st.sidebar:
    st.title("Model Configuration")
    selected_model=st.selectbox(label="Select a model",options=available_models.keys(),placeholder=available_models['Ling 3 Flash'])
    model_id=available_models[selected_model]
    if st.button("Clear Conversation"):
        st.session_state.chat_history = [
            AIMessage(
                content="Hello! I am an AI assistant powered by OpenRouter & LangChain. How can I assist you today?")
        ]
        st.rerun()
    if vectorstore is not None:
        st.success("🎯 Chroma DB: Connected & Active (Assets Loaded)")
    else:
        st.warning("⚠️ No PDFs found in 'assets/' folder. Running in normal chat mode.")

    st.markdown("---")
    st.header("Developer Information")
    st.markdown("""
            **Mohiuddin Mahady**  
            *BSc in CSE*  
            Mymensingh Engineering College  
            *(Affiliated with Dhaka University)*
            """)
    col3, col4 = st.columns([1, 1])
    with col3:
        st.link_button("LinkedIn", "https://www.linkedin.com/in/mohiuddin-mahady/", use_container_width=True)
    with col4:
        st.link_button("Github", 'https://www.github.com/mahady13', use_container_width=True)

@st.cache_resource
def get_llm(model_id):
    return ChatOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
        model=model_id,
        max_tokens=1000,
        temperature=0.3,
        default_headers={
            "HTTP-Referer":"https://localhost:8501/",
            "X-Title":"OmniChatAI Persistent RAG"
        }
    )


def get_response(user_query, chat_history, model_id, vectorstore):
    context = ""
    if vectorstore is not None:
        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
        relevant_docs = retriever.invoke(user_query)
        context = "\n\n".join([doc.page_content for doc in relevant_docs])

    template = """
    You are OmniChat AI, an intelligent, modern, and adaptive AI assistant developed by Mohiuddin Mahady.

        Guidelines:
        1. Prioritize using the Document Context below to answer the user's question accurately.
        2. If you don't know or context doesn't contain the answer, state it clearly. Do not make up facts.
        3. Keep your tone professional, helpful, and respect the language style of the user.

        Document Context:
        {context}

        Conversation History:
        {chat_history}

        User Question:
        {user_query}

    """
    prompt = ChatPromptTemplate.from_template(template)
    llm = get_llm(model_id)
    chain = prompt | llm | StrOutputParser()
    output = chain.stream({
        "context": context,
        "chat_history": chat_history,
        "user_query": user_query
    })

    return output
if "chat_history" not in st.session_state:
    st.session_state.chat_history=[
        AIMessage(content="Hello! I am an AI assistant powered by OpenRouter & LangChain. How can I assist you today?")
    ]
for message in st.session_state.chat_history:
    if isinstance(message,AIMessage):
        with st.chat_message("assistant"):
            st.write(message.content)

    elif isinstance(message,HumanMessage):
        with st.chat_message("user"):
            st.write(message.content)
user_query=st.chat_input("Type Your Message Here")

if user_query:
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    with st.chat_message("user"):
        st.markdown(user_query)

    try:
        response = st.write_stream(get_response(user_query, st.session_state.chat_history, model_id, vectorstore))
        st.session_state.chat_history.append(AIMessage(content=response))
    except Exception as e:
        st.error("⚠️ Selected free model is temporarily rate-limited or busy. Please switch to another model from the sidebar!")


