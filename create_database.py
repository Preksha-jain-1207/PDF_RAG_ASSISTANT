# we have to only run our main.pyb file
# so whenever we run main.py it will recreate the database causing a lot of time
# so the plan is that
# we will create our database once and use it for the rest of our operations
# instead of creating it again and again for every operation

#HERE WE
# load our database
# create chunks -> split the big info into chunks
# create embedding for those chunks
# store it in database (chromaDB)

# to apan itni processess baar baar run krne se bach jaege

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv

load_dotenv()

data = PyPDFLoader("SE_UNIT 4.pdf")
docs = data.load()

print(f"Total pages loaded from PDF: {len(docs)}")

splitter = RecursiveCharacterTextSplitter(
    separators="",
    chunk_size = 250,
    chunk_overlap = 50
)

chunks = splitter.split_documents(docs)

from langchain_huggingface import HuggingFaceEmbeddings
Embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

vector_store = Chroma.from_documents(
      documents=chunks,
      embedding=Embedding_model,
      persist_directory="chroma_db"
)
