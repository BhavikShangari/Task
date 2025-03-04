import streamlit as st
import os
import time
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
# from langchain_community.document_loaders import WebBaseLoader
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
# from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyMuPDFLoader
from langchain_google_genai import ChatGoogleGenerativeAI
import torch
# os.environ["GOOGLE_API_KEY"] = "AIzaSyDuXBRuUZLyIUcM78NtUye9uY5BJKUrWow"
os.environ["GOOGLE_API_KEY"] = "AIzaSyBclgFWk-puE618EnrNx5N2jjSoq6xDRYk"
from langchain_google_genai import GoogleGenerativeAIEmbeddings


embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

with st.sidebar:
    st.title("📄 Retrieved Documents")
    if "retrieved_docs" not in st.session_state:
        st.session_state["retrieved_docs"] = []

st.markdown("<h1 style='text-align: center;'>What’s the vibe today?</h1>", unsafe_allow_html=True)

if "qa_chain" not in st.session_state:

    documents = []

    for file in os.listdir("Data"):
        if file.endswith(".pdf"):
            loader = PyMuPDFLoader(file_path=f"Data/{file}")
            doc = loader.load()
            documents.extend(doc)

    # text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    # splits = text_splitter.split_documents(documents)
    vector_store = FAISS.from_documents(documents, embeddings)

    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"score_threshold": 0.5}
    )

    contextualize_q_system_prompt = """Given a chat history and the latest user question \
    which might reference context in the chat history, formulate a standalone question \
    which can be understood without the chat history. Do NOT answer the question, \
    just reformulate it if needed and otherwise return it as is."""
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    qa_system_prompt = """You are an assistant for JioPay specialized in question-answering tasks. \
    If anyone asks you anything not related to JioPay just say you cannot comment anything else related to jioPay. \
    Use the following pieces of retrieved context to answer the question. \
    If you don't know the answer, just say that you don't know. \
    keep the answer concise and reply in Neatly formattedm manner.\

    {context}"""
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)


    store = {}


    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in store:
            store[session_id] = ChatMessageHistory()
        return store[session_id]


    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )
    
    st.session_state['qa_chain'] = conversational_rag_chain
    st.session_state['retriever'] = retriever
    st.session_state['chat_history'] = []
    st.session_state['gemini_chat_history'] = []


for message in st.session_state["chat_history"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is up?"):
    st.session_state["chat_history"].append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    streaming_response = st.session_state['qa_chain'].stream(
        {"input": prompt},
        config={"configurable": {"session_id": "abc123"}},
    )
    
    full_response = ''
    with st.chat_message("assistant"):
        placeholder = st.empty()
        for i, out in enumerate(streaming_response):
            if i == 0:
                st.session_state['gemini_chat_history'].extend(out['chat_history'])
            if i ==1:
                context = out['context']
            elif i > 1: 
                for word in out['answer'].split():
                    placeholder.markdown(full_response +' ' + word+ ' 🖊️')
                    full_response += ' '+word
                    time.sleep(0.005)
        placeholder.markdown(full_response)
        st.session_state["chat_history"].append({"role": "assistant", "content": full_response})
    with st.sidebar:
        for idx, doc in enumerate(context, start=1):
            with st.expander(f"Document {idx} ({doc.metadata['source'].split('/')[1]})"):
                print(doc, doc.page_content)
                st.write(doc.page_content.strip())