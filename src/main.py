import json
from data_loader import load_documents
from chunking import chunk_with_character, chunk_with_recursive, chunk_with_nltk
from embeddings_store import get_embeddings, build_faiss_index, save_faiss_index
from retriever import build_bm25, similarity_search, mmr_search, hybrid_search
from evaluator import load_test_questions, evaluate_retrieval
from llm_utils import load_local_llm
from rag_chain import generate_answer
from config import RESULTS_DIR

def main():
    print("Загрузка документов...")
    docs = load_documents()

    print("Чанкинг...")
    char_chunks = chunk_with_character(docs)
    rec_chunks = chunk_with_recursive(docs)
    nltk_chunks = chunk_with_nltk(docs)

    print(f"Character chunks: {len(char_chunks)}")
    print(f"Recursive chunks: {len(rec_chunks)}")
    print(f"NLTK chunks: {len(nltk_chunks)}")

    best_chunks = rec_chunks

    print("Эмбеддинги и FAISS...")
    embeddings = get_embeddings()
    vectorstore = build_faiss_index(best_chunks, embeddings)
    save_faiss_index(vectorstore)

    print("BM25...")
    bm25 = build_bm25(best_chunks)

    print("Оценка retrieval...")
    test_questions = load_test_questions()

    similarity_metrics = evaluate_retrieval(
        test_questions,
        lambda q: similarity_search(vectorstore, q)
    )

    mmr_metrics = evaluate_retrieval(
        test_questions,
        lambda q: mmr_search(vectorstore, q)
    )

    hybrid_metrics = evaluate_retrieval(
        test_questions,
        lambda q: hybrid_search(vectorstore, bm25, best_chunks, q)
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "retrieval_metrics.json", "w", encoding="utf-8") as f:
        json.dump({
            "similarity": similarity_metrics,
            "mmr": mmr_metrics,
            "hybrid": hybrid_metrics
        }, f, ensure_ascii=False, indent=2)

    print("Загрузка LLM...")
    generator = load_local_llm()

    user_query = "Какая ставка по семейной ипотеке?"
    docs_found = hybrid_search(vectorstore, bm25, best_chunks, user_query)
    answer = generate_answer(generator, user_query, docs_found)

    print("\nОтвет системы:\n")
    print(answer)

if __name__ == "__main__":
    main()