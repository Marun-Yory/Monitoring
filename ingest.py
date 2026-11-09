
import json
from pathlib import Path

from minsearch import Index

BASE_DIR = Path(__file__).resolve().parent


def load_currency_data(file_path=None):
    documents = []

    if file_path is None:
        file_path = BASE_DIR / "documents" / "currencies.json"

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    documents = data["currencies"]

    for i, doc in enumerate(documents):
        doc["doc_id"] = doc.get("code", str(i))
        doc["countries_text"] = ", ".join(doc["countries"])

    return documents


def build_index(documents):
    index = Index(
        text_fields=[
            "code",
            "name",
            "symbol",
            "countries_text",
        ],
        keyword_fields=[
            "code",
        ],
    )

    index.fit(documents)

    return index


if __name__ == "__main__":
    documents = load_currency_data()
    index = build_index(documents)

    results = index.search(
        "What currency is used in Germany?",
        num_results=3,
        boost_dict={
            "code": 3.0,
            "name": 2.0,
            "symbol": 2.0,
            "countries_text": 2.0,
        },
    )

    for result in results:
        print(result)
