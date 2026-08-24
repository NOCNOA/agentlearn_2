import re
import requests
import feedparser


def normalize_arxiv_id(arxiv_id: str) -> str:
    arxiv_id = arxiv_id.strip()

    if "/abs/" in arxiv_id:
        arxiv_id = arxiv_id.split("/abs/")[-1]

    if "/pdf/" in arxiv_id:
        arxiv_id = arxiv_id.split("/pdf/")[-1]

    arxiv_id = arxiv_id.removesuffix(".pdf")

    # 2608.12345v2 → 2608.12345
    arxiv_id = re.sub(r"v\d+$", "", arxiv_id)

    return arxiv_id


def get_arxiv_paper(arxiv_id: str):
    arxiv_id = normalize_arxiv_id(arxiv_id)

    response = requests.get(
        "https://export.arxiv.org/api/query",
        params={
            "id_list": arxiv_id,
            "max_results": 1,
        },
        timeout=20,
    )

    response.raise_for_status()

    feed = feedparser.parse(response.text)

    if not feed.entries:
        return {
            "error": "Paper not found",
            "arxiv_id": arxiv_id,
        }

    entry = feed.entries[0]

    return {
        "arxiv_id": arxiv_id,
        "title": entry.title.strip(),
        "authors": [
            author.name
            for author in entry.authors
        ],
        "summary": " ".join(entry.summary.split()),
        "published": entry.published,
        "updated": getattr(entry, "updated", None),
        "categories": [
            tag.term
            for tag in getattr(entry, "tags", [])
        ],
        "url": entry.link,
    }
    
if __name__ == "__main__":
    paper = get_arxiv_paper("2608.20788")
    print(paper)
    