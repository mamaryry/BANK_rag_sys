from pathlib import Path
from langchain_core.documents import Document
from config import DATA_DIR

PRODUCT_MAP = {
    "01_credit.txt": "credit",
    "02_deposit.txt": "deposit",
    "03_mortgage.txt": "mortgage",
    "04_requirements.txt": "requirements",
    "05_faq.txt": "faq",
}

def load_documents():
    documents = []

    for file_path in DATA_DIR.glob("*.txt"):
        text = file_path.read_text(encoding="utf-8").strip()

        doc = Document(
            page_content=text,
            metadata={
                "source_file": file_path.name,
                "product_type": PRODUCT_MAP.get(file_path.name, "unknown"),
                "doc_id": file_path.stem
            }
        )
        documents.append(doc)

    return documents