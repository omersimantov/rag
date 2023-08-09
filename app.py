import os
import time
import pinecone
from PIL import Image
import streamlit as st
from dotenv import load_dotenv
from langchain.vectorstores import Pinecone
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

# Load environment variables
load_dotenv()
TITLE = os.getenv("COMPANY_NAME") + " GPT"
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")

# Create Streamlit app
favicon = Image.open("favicon.png")
st.set_page_config(page_title=TITLE, page_icon=favicon)
st.title(TITLE)
if "conversation" not in st.session_state:
    st.session_state.conversation = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = None
if "messages" not in st.session_state:
    st.session_state["messages"] = []
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])
st.markdown(
    # Hides title links
    """
        <style>
        .css-15zrgzn {display: none}
        .css-eczf16 {display: none}
        .css-jn99sy {display: none}
        </style>
        """,
    unsafe_allow_html=True,
)

# Configure LLM
llm = ChatOpenAI(model="gpt-4", temperature=0, streaming=True)

# Initialize vector store
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)
index = pinecone.Index(PINECONE_INDEX_NAME)
embedding_function = OpenAIEmbeddings().embed_query
vectorstore = Pinecone(index, embedding_function, "text")

# Configure retrieval chain
chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(),
    memory=ConversationBufferMemory(memory_key="chat_history"),
    verbose=True,
)

# Handle prompt
prompt = st.chat_input()
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    with st.chat_message("assistant"):
        response = chain.run(prompt)
        stream = ""
        message_placeholder = st.empty()
        for chunk in response.split():
            stream += chunk + " "
            time.sleep(0.05)
            message_placeholder.markdown(stream + "▌")
        message_placeholder.markdown(stream)
    st.session_state.messages.append({"role": "assistant", "content": stream})
