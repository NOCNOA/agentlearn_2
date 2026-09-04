import feedparser
import json
import re

from tools.arxiv_client import arxiv_get


def build_arxiv_query(query: str) -> str:
    """Convert plain keywords into an explicit arXiv AND query.

    ``all:large disparity stereo`` does not apply ``all:`` independently to
    every word.  The explicit form below is both unambiguous and narrower:
    ``all:large AND all:disparity AND all:stereo``.
    """
    terms = re.findall(r"[A-Za-z0-9][A-Za-z0-9_.+\-]*", query)

    if not terms:
        raise ValueError("The arXiv query must contain at least one search term")

    return " AND ".join(f"all:{term}" for term in terms)


def search_arxiv(query: str, max_results: int = 5):

    params = {
        "search_query": build_arxiv_query(query),
        "start": 0,
        "max_results": max_results,
        "sortBy": "relevance",
        "sortOrder": "descending",
    }

    response = arxiv_get(
        params=params,
        timeout=30,
    )

    feed = feedparser.parse(response.text)

    papers = []

    for entry in feed.entries:

        papers.append(
            {
                "arxiv_id": entry.id.split("/abs/")[-1],
                "title": entry.title.strip(),
                "authors": [
                    author.name
                    for author in entry.authors
                ],
                "published": entry.published,
                "url": entry.link,
            }
)
    
    result = {
        "query": query,
        "count": len(papers),
        "papers": papers,
    }
    #转为json格式
    return json.dumps(
        result,
        ensure_ascii=False,
    )

if __name__ == "__main__":

    result = search_arxiv(
        "stereo matching",
        max_results=5
    )

    papers = json.loads(result)["papers"]
    
    for paper in papers:
        print("=" * 80)
        print(paper["title"])
        print(paper["published"])
        print(paper["url"])
