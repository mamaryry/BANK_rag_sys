from rank_bm25 import BM25Okapi
from config import TOP_K

def similarity_search(vectorstore, query, k=TOP_K, product_type=None):
    if product_type:
        docs = [
            doc for doc in vectorstore.docstore._dict.values()
            if doc.metadata.get("product_type") == product_type
        ]
        return docs[:k]
    return vectorstore.similarity_search(query, k=k)

def mmr_search(vectorstore, query, k=TOP_K):
    return vectorstore.max_marginal_relevance_search(query, k=k)

def build_bm25(chunks):
    tokenized_corpus = [doc.page_content.lower().split() for doc in chunks]
    return BM25Okapi(tokenized_corpus)

def bm25_search(bm25, chunks, query, k=TOP_K):
    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)
    ranked = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in ranked[:k]]

def hybrid_search(vectorstore, bm25, chunks, query, k=TOP_K):
    dense_docs = vectorstore.similarity_search(query, k=k)
    sparse_docs = bm25_search(bm25, chunks, query, k=k)

    merged = []
    seen = set()

    for doc in dense_docs + sparse_docs:
        key = (doc.page_content[:100], doc.metadata.get("source_file"))
        if key not in seen:
            merged.append(doc)
            seen.add(key)

    return merged[:k]

def detect_product_type(query: str):
    q = query.lower()

    if "ипот" in q:
        return "mortgage"
    if "вклад" in q or "депозит" in q or "накопитель" in q:
        return "deposit"
    if "кредит" in q or "автокредит" in q or "образовательный" in q:
        return "credit"
    if "документ" in q or "заемщик" in q or "требован" in q:
        return "requirements"
    if "что делать" in q or "как узнать" in q or "если" in q:
        return "faq"
    return None