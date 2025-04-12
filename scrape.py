import os
import streamlit as st
import pickle
import time
from dotenv import load_dotenv

# LangChain & GenAI
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI

# Load .env variables
load_dotenv()

# === Initialize Streamlit UI ===
st.set_page_config(page_title="URL Insight Bot", layout="wide")
st.title("🔗 URL Insight Bot")
st.sidebar.title("📥 Enter Web Page URLs")

# === URL Input Section ===
urls = []
for i in range(3):
    urls.append(st.sidebar.text_input(f"URL {i+1}"))

process_button = st.sidebar.button("🔄 Process URLs")
main_placeholder = st.empty()

# === Check if vectorstore exists ===
faiss_exists = os.path.exists("faiss.pkl")
if faiss_exists:
    st.sidebar.success("🔎 Vectorstore loaded. You can ask questions!")
else:
    st.sidebar.warning("⚠️ No vectorstore found. Please process URLs first.")

# === Cache heavy resources ===
@st.cache_resource(show_spinner=False)
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

@st.cache_resource(show_spinner=False)
def get_llm():
    return ChatGoogleGenerativeAI(
        model="models/gemini-1.5-pro",
        temperature=0.5,
        max_output_tokens=500,
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )

@st.cache_resource(show_spinner=False)
def load_faiss():
    with open("faiss.pkl", "rb") as f:
        return pickle.load(f)

# === Retry logic ===
def safe_chain_call(chain, query, retries=3, cooldown=60):
    for attempt in range(retries):
        try:
            return chain.invoke({"question": query})
        except Exception as e:
            if "429" in str(e):
                st.warning(f"[{attempt+1}/{retries}] Quota exceeded, retrying in {cooldown} seconds...")
                time.sleep(cooldown)
            else:
                st.error(f"Unexpected error: {e}")
                return {"answer": "Error occurred", "sources": ""}
    return {"answer": "Quota exhausted. Try again later.", "sources": ""}

# === Process URLs ===
if process_button:
    valid_urls = [url for url in urls if url.strip()]
    if not valid_urls:
        st.warning("⚠️ Please enter at least one valid URL.")
        st.stop()

    loader = UnstructuredURLLoader(urls=valid_urls)
    main_placeholder.text("🔄 Loading data from URLs...")

    try:
        data = loader.load()
        if not data:
            st.error("❌ No content could be loaded from the provided URLs.")
            st.stop()
    except Exception as e:
        st.error(f"❌ Failed to load URLs: {str(e)}")
        st.stop()

    st.success(f"✅ Loaded {len(data)} documents.")

    for i, d in enumerate(data):
        st.markdown(f"**Preview {i+1}:** {d.metadata.get('source', '')}")
        st.write(d.page_content[:300] + "..." if d.page_content else "⚠️ Empty content")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    main_placeholder.text("🧩 Splitting data into chunks...")
    docs = text_splitter.split_documents(data)

    if not docs:
        st.error("❌ No content to embed. The webpage may be empty or unreadable.")
        st.stop()

    embeddings = get_embeddings()
    main_placeholder.text("🔗 Creating FAISS vector store...")
    faiss = FAISS.from_documents(docs, embeddings)

    with open("faiss.pkl", "wb") as f:
        pickle.dump(faiss, f)

    main_placeholder.success("✅ Processing complete. Now you can ask questions below.")

# === Question Input Section ===
st.divider()
st.subheader("🧠 Ask a question based on the links provided")

query = st.text_input("Type your question here:")
get_answer_button = st.button("Get Answer")

@st.cache_data(show_spinner=False)
def get_cached_response(question):
    faiss = load_faiss()
    retriever = faiss.as_retriever(search_kwargs={"k": 3})
    llm = get_llm()
    chain = RetrievalQAWithSourcesChain.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff"
    )
    return safe_chain_call(chain, question)

# === Get Answer Logic ===
if get_answer_button and query:
    if os.path.exists("faiss.pkl"):
        main_placeholder.text("🔍 Searching for answer...")
        answer = get_cached_response(query)

        st.header("Answer:")
        st.subheader(answer["answer"])

        sources = answer.get("sources", "")
        if sources:
            st.header("Sources:")
            for source in sources.split("\n"):
                if source.strip():
                    st.write(source)
    else:
        st.error("No FAISS index found. Please process URLs first.")
