from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from config import EMBEDDING_MODEL_NAME, FAISS_DIR

def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        encode_kwargs={"normalize_embeddings": True}
    )

def build_faiss_index(chunks, embeddings):
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore

def save_faiss_index(vectorstore):
    FAISS_DIR.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(FAISS_DIR))

def load_faiss_index(embeddings):
    return FAISS.load_local(
        str(FAISS_DIR),
        embeddings,
        allow_dangerous_deserialization=True
    )