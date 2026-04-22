from retriever import detect_product_type

SYSTEM_PROMPT = """
Ты банковский AI-консультант.
Отвечай только на основе предоставленного контекста.
Если ответа в контексте нет, честно скажи, что информация не найдена.
Не выдумывай условия, ставки и требования.
В конце укажи источники.
"""

def format_context(docs):
    blocks = []
    for i, doc in enumerate(docs, start=1):
        block = (
            f"[Источник {i}] {doc.metadata.get('source_file')}\n"
            f"{doc.page_content}\n"
        )
        blocks.append(block)
    return "\n\n".join(blocks)

def generate_answer(generator, query, docs):
    context = format_context(docs)

    prompt = f"""
{SYSTEM_PROMPT}

Контекст:
{context}

Вопрос:
{query}

Ответ:
"""

    result = generator(prompt)[0]["generated_text"]
    return result

def build_sources_list(docs):
    unique_sources = []
    seen = set()

    for doc in docs:
        src = doc.metadata.get("source_file")
        if src not in seen:
            unique_sources.append(src)
            seen.add(src)

    return unique_sources