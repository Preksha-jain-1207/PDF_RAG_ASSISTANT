import os
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()

from langchain_huggingface import HuggingFaceEmbeddings
Embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

vector_store = Chroma(
      persist_directory="chroma_db",
      embedding_function=Embedding_model      
)

print(f"Total chunks in database: {vector_store._collection.count()}")

retriever = vector_store.as_retriever(
    search_type = "similarity",
    search_kwagrs = {
         "k" :5
    }
)

model = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

prompt = ChatPromptTemplate.from_messages(
    [
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
    ]
)

print("RAG sysrtem created")
print("press 0 to exit thre conversation")

while True:
    query = input("Ask a question related to the PDF: ")
    if query == "0":
        break

    docs = retriever.invoke(query)

    print("\n[DEBUG] Retrieved Chunks:")
    for i, doc in enumerate(docs):
        print(f"--- Chunk {i+1} ---")
        print(doc.page_content[:300])

    context = "\n\n".join([doc.page_content for doc in docs])

    # Yahan doosra input() HATA diya hai, ab upar wala 'query' hi use hoga
    final_prompt = prompt.invoke(
        {
            "context" : context,
            "question" : query
        }
    )

    response = model.invoke(final_prompt)

    answer_text = response.content
    if isinstance(answer_text, list):
        answer_text = "".join([item.get("text", "") for item in answer_text if isinstance(item, dict)])

    print("\n")
    print(f"AI : {answer_text}")
