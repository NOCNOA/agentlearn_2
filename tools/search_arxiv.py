import feedparser
import json
import re

from tools.arxiv_client import arxiv_get


def build_arxiv_query(
    query: str,
    start_year: int | None = None,
    end_year: int | None = None,
) -> str:
    """Convert plain keywords into an explicit arXiv AND query.

    ``all:large disparity stereo`` does not apply ``all:`` independently to
    every word.  The explicit form below is both unambiguous and narrower:
    ``all:large AND all:disparity AND all:stereo``.
    """
    terms = re.findall(r"[A-Za-z0-9][A-Za-z0-9_.+\-]*", query)

    if not terms:
        raise ValueError("The arXiv query must contain at least one search term")

    keyword_query = " AND ".join(f"all:{term}" for term in terms)

    if start_year is None and end_year is None:
        return keyword_query

    if start_year is None or end_year is None:
        raise ValueError("start_year and end_year must be provided together")

    if start_year > end_year:
        raise ValueError("start_year cannot be later than end_year")

    date_query = (
        f"submittedDate:[{start_year}01010000 TO {end_year}12312359]"
    )
    return f"({keyword_query}) AND {date_query}"


def search_arxiv(
    query: str,
    max_results: int = 5,
    start_year: int | None = None,
    end_year: int | None = None,
):

    params = {
        "search_query": build_arxiv_query(
            query,
            start_year=start_year,
            end_year=end_year,
        ),
        "start": 0,
        "max_results": max_results,
        "sortBy": "submittedDate" if start_year is not None else "relevance",
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
        "year_range": (
            {"start_year": start_year, "end_year": end_year}
            if start_year is not None
            else None
        ),
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
