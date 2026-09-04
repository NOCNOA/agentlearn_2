from io import BytesIO

from pypdf import PdfReader

from tools.arxiv_client import arxiv_download_pdf
from tools.get_arxiv_paper import normalize_arxiv_id


DEFAULT_MAX_PAGES = 8
MAX_PAGES = 12
MAX_TEXT_CHARACTERS = 24_000


def get_arxiv_pdf_text(
    arxiv_id: str,
    max_pages: int = DEFAULT_MAX_PAGES,
) -> dict:
    """Download a bounded excerpt of an arXiv PDF for evidence extraction."""
    arxiv_id = normalize_arxiv_id(arxiv_id)

    if not isinstance(max_pages, int) or not 1 <= max_pages <= MAX_PAGES:
        raise ValueError(
            f"max_pages must be an integer from 1 to {MAX_PAGES}."
        )

    response = arxiv_download_pdf(arxiv_id)
    content_type = response.headers.get("Content-Type", "").lower()

    if "pdf" not in content_type and not response.content.startswith(b"%PDF"):
        raise RuntimeError("arXiv did not return a PDF document.")

    try:
        reader = PdfReader(BytesIO(response.content))
    except Exception as exc:
        raise RuntimeError(f"Could not parse arXiv PDF: {exc}") from exc

    pages_to_read = min(max_pages, len(reader.pages))
    extracted_pages = []
    truncated = False
    remaining_characters = MAX_TEXT_CHARACTERS

    for page_number in range(pages_to_read):
        page_text = reader.pages[page_number].extract_text() or ""
        page_text = " ".join(page_text.split())

        if not page_text:
            continue

        if len(page_text) > remaining_characters:
            page_text = page_text[:remaining_characters]
            truncated = True

        extracted_pages.append(
            {
                "page": page_number + 1,
                "text": page_text,
            }
        )
        remaining_characters -= len(page_text)

        if remaining_characters == 0:
            truncated = True
            break

    return {
        "arxiv_id": arxiv_id,
        "pdf_url": f"https://arxiv.org/pdf/{arxiv_id}",
        "page_count": len(reader.pages),
        "pages_read": pages_to_read,
        "text_truncated": truncated or pages_to_read < len(reader.pages),
        "extracted_pages": extracted_pages,
    }
