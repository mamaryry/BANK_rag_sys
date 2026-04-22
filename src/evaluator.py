import json
from pathlib import Path
from config import TESTS_DIR

def load_test_questions():
    file_path = TESTS_DIR / "test_questions.json"
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def hit_rate_at_k(results, gold_source, k=3):
    top_k_sources = [doc.metadata.get("source_file") for doc in results[:k]]
    return int(gold_source in top_k_sources)

def reciprocal_rank(results, gold_source):
    for idx, doc in enumerate(results, start=1):
        if doc.metadata.get("source_file") == gold_source:
            return 1 / idx
    return 0.0

def evaluate_retrieval(test_questions, search_fn):
    hit_scores = []
    rr_scores = []

    for item in test_questions:
        query = item["question"]
        gold_source = item["gold_source"]

        results = search_fn(query)

        hit_scores.append(hit_rate_at_k(results, gold_source, k=3))
        rr_scores.append(reciprocal_rank(results, gold_source))

    return {
        "HitRate@3": sum(hit_scores) / len(hit_scores),
        "MRR": sum(rr_scores) / len(rr_scores)
    }