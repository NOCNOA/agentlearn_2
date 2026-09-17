from tools.search_arxiv import search_arxiv
from tools.get_arxiv_paper import get_arxiv_paper
from tools.get_arxiv_pdf_text import get_arxiv_pdf_text

TOOL_REGISTRY = {
    "search_arxiv": search_arxiv,
    "get_arxiv_paper": get_arxiv_paper,
    "get_arxiv_pdf_text": get_arxiv_pdf_text,
}
