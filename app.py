import os
import streamlit as st
import pickle
import time
from langchain_community.llms import Ollama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import SKLearnVectorStore
import tempfile

st.title("Website Research ...")
st.sidebar.title("URLs to articels")


main_placefolder = st.empty()

ans = st.empty()


urls = []

for i in range (1):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)
    main_placefolder.write(url)
processURL_clicked = st.sidebar.button("Process URL")

vector_store = None

for url in urls:
    st.sidebar.write(url)

llm = Ollama(model="mistral")

embeddings = OllamaEmbeddings(model="mistral")
from pathlib import Path
persist_path = Path("./vectore_store.pkl")
if processURL_clicked:
    with st.status("Loading Data ...", expanded=False):
        #load data
        try:
            loader = WebBaseLoader(urls)
            data = loader.load()
        except: 
            st.write("error in loading URL")

        #split data
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = 1000,
            chunk_overlap=200
        )

    docs = text_splitter.split_documents(data)

    #save embeddings
    persist_path = os.path.join(tempfile.gettempdir(), "union.parquet")
    with st.status("Making embeddings ...", expanded=False):
        st.image("https://duckduckgo.com/?q=images&iax=images&ia=images&iai=https://wallpapercave.com/wp/wp2599594.jpg&t=ffab&atb=v419-1")
        vector_store = SKLearnVectorStore.from_documents(
           documents=docs,
           embedding=embeddings,
           persist_path=persist_path,  # persist_path and serializer are optional
           serializer="parquet",
       )
        vector_store.persist()


main_placefolder.text("done")
time.sleep(1)


# button = st.empty()
from langchain import hub

# Loads the latest version
# prompt = hub.pull("rlm/rag-prompt", api_url="https://api.hub.langchain.com")


from langchain.chains import RetrievalQA
    

question:str = st.text_input(label="Input Question")
    
ask = st.button("ask")
if ask:
    try:
        vector_store = SKLearnVectorStore(embedding=embeddings,
                persist_path=persist_path,  # persist_path and serializer are optional
                serializer="parquet", )
    except:
        st.error("vector_store not found")
    qa_chain = RetrievalQA.from_chain_type(
        llm, retriever=vector_store.as_retriever(), 
        #chain_type_kwargs={"prompt": prompt}
    )
    with main_placefolder.status(f'thinking {question}'):
        result = qa_chain({"query": question})  
# with st.status("thinking ...", expanded=False):
    st.text(result["result"])

#result = qa_chain({"query": question})
#ans.write(result['result'])