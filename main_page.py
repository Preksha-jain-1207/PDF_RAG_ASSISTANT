import os
import tempfile
import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="PDF-Brain AI | RAG Studio",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

load_dotenv()

# --- BULLETPROOF DARK THEME & TEXT CONTRAST CSS ---
st.markdown("""
<style>
    /* Global Background & Base Text */
    .stApp {
        background-color: #090d16 !important;
        color: #f1f5f9 !important;
        font-family: 'Inter', sans-serif;
    }
    
    /* Hide default streamlit header elements */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    
    /* Glassmorphism Sidebar */
    [data-testid="stSidebar"] {
        background-color: #0d1322 !important;
        border-right: 1px solid rgba(255, 255, 255, 0.08);
    }
    
    [data-testid="stSidebar"] p, 
    [data-testid="stSidebar"] span, 
    [data-testid="stSidebar"] label {
        color: #cbd5e1 !important;
    }

    /* --- CHAT MESSAGE RESPONSE TEXT CONTRAST FIX --- */
    [data-testid="stChatMessage"] {
        color: #f1f5f9 !important;
    }
    
    [data-testid="stChatMessage"] p, 
    [data-testid="stChatMessage"] span, 
    [data-testid="stChatMessage"] li,
    [data-testid="stMarkdownContainer"] p {
        color: #f1f5f9 !important;
        font-size: 1.05rem !important;
        line-height: 1.6 !important;
    }

    /* --- CHAT INPUT DARK MODE FIX --- */
    [data-testid="stChatInput"] {
        background-color: #1e293b !important;
        border: 1px solid #38bdf8 !important;
        border-radius: 14px !important;
    }
    
    [data-testid="stChatInput"] > div {
        background-color: #1e293b !important;
        border-radius: 14px !important;
    }

    [data-testid="stChatInput"] textarea {
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
        background-color: transparent !important;
        font-size: 1rem !important;
    }
    
    [data-testid="stChatInput"] textarea::placeholder {
        color: #94a3b8 !important;
        -webkit-text-fill-color: #94a3b8 !important;
        opacity: 1 !important;
    }

    /* --- FILE UPLOADER FIX --- */
    [data-testid="stFileUploader"] {
        background: #1e293b !important;
        border: 2px dashed #38bdf8 !important;
        border-radius: 14px;
        padding: 10px;
    }
    
    [data-testid="stFileUploader"] *, 
    [data-testid="stFileUploaderDropzone"] * {
        color: #e2e8f0 !important;
    }
    
    [data-testid="stFileUploader"] section {
        background-color: #0f172a !important;
    }

    /* Neon Action Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #0ea5e9 0%, #6366f1 100%) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.6rem 1.2rem;
        font-weight: 600;
        box-shadow: 0 4px 15px rgba(14, 165, 233, 0.3);
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(99, 102, 241, 0.5);
    }
    
    /* Headers */
    h1, h2, h3 {
        letter-spacing: -0.5px;
        color: #f8fafc !important;
    }
    
    .gradient-title {
        background: linear-gradient(90deg, #38bdf8 0%, #818cf8 50%, #c084fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 2.4rem;
    }
</style>
""", unsafe_allow_html=True)

# --- CACHED / INITIALIZED RESOURCES ---
@st.cache_resource
def get_embedding_model():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

embedding_model = get_embedding_model()
DB_DIR = "chroma_db"

# --- SIDEBAR: DATABASE & SYSTEM CONFIGURATION ---
with st.sidebar:
    st.markdown("### ⚡ Control Matrix")
    st.markdown("Upload your custom PDF document to build a persistent vector search index.")
    
    db_exists = os.path.exists(DB_DIR)
    chunk_count = 0
    
    if db_exists:
        try:
            temp_store = Chroma(persist_directory=DB_DIR, embedding_function=embedding_model)
            chunk_count = temp_store._collection.count()
            st.success(f"🟢 DB Active: `{chunk_count}` Chunks loaded")
        except Exception:
            st.warning("⚠️ DB Found, but count unavailable.")
    else:
        st.error("🔴 Database not found!")

    st.markdown("---")
    st.markdown("### 📄 Upload PDF Document")
    
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])
    
    if st.button("🚀 Ingest & Build Matrix"):
        if uploaded_file is None:
            st.error("Please upload a PDF file first!")
        else:
            with st.spinner("⏳ Saving file, tokenizing chunks & embedding vectors..."):
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_file_path = tmp_file.name

                    loader = PyPDFLoader(tmp_file_path)
                    docs = loader.load()
                    
                    splitter = RecursiveCharacterTextSplitter(
                        separators=["\n\n", "\n", " ", ""],
                        chunk_size=250,
                        chunk_overlap=50
                    )
                    chunks = splitter.split_documents(docs)
                    
                    vector_store = Chroma.from_documents(
                        documents=chunks,
                        embedding=embedding_model,
                        persist_directory=DB_DIR
                    )
                    
                    os.unlink(tmp_file_path)
                    
                    st.success(f"Successfully indexed {len(chunks)} chunks from `{uploaded_file.name}`!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error building database: {e}")

    st.markdown("---")
    st.markdown("### 🤖 Model Specs")
    st.info("**LLM:** gemini-1.5-flash\n\n**Embeddings:** all-MiniLM-L6-v2\n\n**Retriever:** ChromaDB (k=5)")

# --- MAIN INTERFACE ---
st.markdown('<p class="gradient-title">🧠 PDF-Brain RAG Studio</p>', unsafe_allow_html=True)
st.markdown("Query deep inside your uploaded documentation with high precision context visualization.")

if not db_exists or chunk_count == 0:
    st.warning("⚠️ Your Vector Database is empty or missing. Please upload a PDF file in the sidebar and click **'Ingest & Build Matrix'** to begin.")
    st.stop()

vector_store = Chroma(persist_directory=DB_DIR, embedding_function=embedding_model)
retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}
)

api_key = os.getenv("GOOGLE_API_KEY")
model = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash",
    google_api_key=api_key
)

prompt_template = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a helpful AI assistant.

Use ONLY the provided context to answer the question.

If the answer is not present in the context,
say: "I could not find the answer in the document." """
    ),
    (
        "human",
        """Context:
{context}

Question:
{question}
"""
    )
])

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "debug_chunks" in message and message["debug_chunks"]:
            with st.expander("🔍 [DEBUG] Retrieved Chunks from Vector Store"):
                for idx, chunk_text in enumerate(message["debug_chunks"]):
                    st.markdown(f"**Chunk {idx+1}**: {chunk_text}")

if query := st.chat_input("Ask a question related to your uploaded PDF..."):
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.status("🔍 Retrieving context chunks & generating answer...", expanded=False) as status:
            docs = retriever.invoke(query)
            debug_texts = [doc.page_content for doc in docs]
            context = "\n\n".join(debug_texts)
            
            final_prompt = prompt_template.invoke({
                "context": context,
                "question": query
            })
            
            try:
                response = model.invoke(final_prompt)
                answer_text = response.content
                if isinstance(answer_text, list):
                    answer_text = "".join([item.get("text", "") for item in answer_text if isinstance(item, dict)])
            except Exception as e:
                if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                    answer_text = "⚠️ **API Limit Reached:** Your Gemini API quota has been exceeded. Please check your plan or try again later."
                else:
                    answer_text = f"An error occurred: {e}"
            
            status.update(label="Response generated successfully!", state="complete", expanded=False)

        st.markdown(answer_text)
        
        with st.expander("🔍 [DEBUG] Retrieved Chunks"):
            for idx, c_text in enumerate(debug_texts):
                st.markdown(f"**Chunk {idx+1}**: `{c_text[:300]}...`")

    st.session_state.messages.append({
        "role": "assistant", 
        "content": answer_text,
        "debug_chunks": [c[:300] for c in debug_texts]
    })