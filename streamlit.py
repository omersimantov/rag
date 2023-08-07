import os
import time
import pinecone
import streamlit as st
from pinecone import Index
from dotenv import load_dotenv
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.vectorstores import Pinecone
from langchain.embeddings import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory

# Credentials
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")

# LLM
llm = OpenAI(streaming=True, temperature=0)

# Vector store
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)
index = Index(PINECONE_INDEX_NAME)

# Retriever
embedding_function = OpenAIEmbeddings().embed_query
vectorstore = Pinecone(index, embedding_function, "text")
qa = RetrievalQA.from_chain_type(
    llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(),
    memory=ConversationBufferMemory(memory_key="chat_history"),
    verbose=True,
)

# Streamlit app
TITLE = os.getenv("COMPANY_NAME") + " GPT"
st.set_page_config(page_title=TITLE)
st.title(TITLE)

if "conversation" not in st.session_state:
    st.session_state.conversation = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = None

if "messages" not in st.session_state:
    st.session_state["messages"] = []

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    with st.chat_message("assistant"):
        response = qa.run(prompt)
        stream = ""
        message_placeholder = st.empty()
        for chunk in response.split():
            stream += chunk + " "
            time.sleep(0.05)
            message_placeholder.markdown(stream + "▌")
        message_placeholder.markdown(stream)
    st.session_state.messages.append({"role": "assistant", "content": stream})
