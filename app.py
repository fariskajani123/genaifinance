import os
from pathlib import Path
from dotenv import load_dotenv, find_dotenv
import streamlit as st

from langchain_community.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# --- env / config
load_dotenv(find_dotenv(), override=True)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY not found in your .env file")
    st.stop()

INDEX_DIR = Path("faiss_index")

@st.cache_resource(show_spinner=False)
def get_embeddings():
    return OpenAIEmbeddings(model="text-embedding-3-small", api_key=OPENAI_API_KEY)

@st.cache_resource(show_spinner=False)
def load_or_build_index(urls):
    embeddings = get_embeddings()
    if INDEX_DIR.exists():
        db = FAISS.load_local(str(INDEX_DIR), embeddings, allow_dangerous_deserialization=True)
        return db
    # Load pages
    loader = UnstructuredURLLoader(urls=urls)
    docs = loader.load()
    # Chunk
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(docs)
    # Build & save
    db = FAISS.from_documents(chunks, embeddings)
    db.save_local(str(INDEX_DIR))
    return db

def make_qa(retriever):
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=OPENAI_API_KEY)
    prompt = PromptTemplate.from_template(
        "Use the context to answer the question concisely. If unsure, say you don't know.\n"
        "Context:\n{context}\n\nQuestion: {question}"
    )
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True,
    )

# --- UI
st.set_page_config(page_title="Finance Q&A (RAG)", page_icon="📈")
st.title("📈 Finance Q&A (RAG)")
st.caption("Paste URLs → Build index → Ask questions. Index is saved to disk for reuse.")

default_urls = """https://www.cnbc.com/2024/01/02/stock-market-today.html
https://www.reuters.com/markets/us/"""
urls = [u.strip() for u in st.text_area("URLs (one per line)", value=default_urls, height=100).splitlines() if u.strip()]

col1, col2 = st.columns([1,1])
with col1:
    if st.button("Build / Load Index"):
        with st.spinner("Building / loading index…"):
            st.session_state.db = load_or_build_index(urls)
            st.session_state.retriever = st.session_state.db.as_retriever(search_kwargs={"k": 4})
        st.success("Index ready ✅")

with col2:
    if st.button("Reset Index"):
        if INDEX_DIR.exists():
            import shutil; shutil.rmtree(INDEX_DIR)
        st.session_state.pop("db", None)
        st.session_state.pop("retriever", None)
        st.toast("Index cleared")

q = st.text_input("Ask a question")
if q:
    if "retriever" not in st.session_state:
        st.warning("Click **Build / Load Index** first.")
        st.stop()
    qa = make_qa(st.session_state.retriever)
    with st.spinner("Thinking…"):
        res = qa.invoke({"query": q})
    st.subheader("Answer")
    st.write(res["result"])
    with st.expander("Sources"):
        for d in res["source_documents"]:
            st.write("-", d.metadata.get("source"))
    if st.checkbox("Show retrieved chunks"):
        for i, d in enumerate(res["source_documents"], 1):
            st.markdown(f"**Chunk {i}** – {d.metadata.get('source')}")
            st.write(d.page_content[:1000])
