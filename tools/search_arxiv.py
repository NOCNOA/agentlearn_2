import requests
import feedparser
import json

def search_arxiv(query: str, max_results: int = 5):

    url = "https://export.arxiv.org/api/query"

    params = {
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": max_results,
        "sortBy": "relevance",
        "sortOrder": "descending",
    }

    response = requests.get(
        url,
        params=params,
        timeout=20,
    )

    response.raise_for_status()

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