import threading
import time

import requests


ARXIV_API_URL = "https://export.arxiv.org/api/query"
MIN_REQUEST_INTERVAL_SECONDS = 3.1
RETRY_BACKOFF_SECONDS = 3.1
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}

_request_lock = threading.Lock()
_last_request_started_at = 0.0
_session = requests.Session()
_session.headers.update(
    {
        "User-Agent": "agentlearn-arxiv-client/0.1",
    }
)


def _retry_after_seconds(response: requests.Response) -> float:
    value = response.headers.get("Retry-After")
    if value is None:
        return 0.0

    try:
        return max(0.0, float(value))
    except ValueError:
        return 0.0


def arxiv_get(
    params: dict,
    timeout: int = 30,
    max_attempts: int = 3,
) -> requests.Response:
    """Call the legacy arXiv API politely and retry transient failures."""
    global _last_request_started_at

    last_error = None

    for attempt in range(1, max_attempts + 1):
        response = None

        # arXiv asks legacy API clients to use one connection and wait at
        # least three seconds between requests.
        with _request_lock:
            elapsed = time.monotonic() - _last_request_started_at
            wait_seconds = MIN_REQUEST_INTERVAL_SECONDS - elapsed

            if wait_seconds > 0:
                time.sleep(wait_seconds)

            _last_request_started_at = time.monotonic()

            try:
                response = _session.get(
                    ARXIV_API_URL,
                    params=params,
                    timeout=timeout,
                )
            except (
                requests.exceptions.Timeout,
                requests.exceptions.ConnectionError,
            ) as exc:
                last_error = exc

        if response is not None:
            if response.status_code not in RETRYABLE_STATUS_CODES:
                response.raise_for_status()
                return response

            last_error = requests.HTTPError(
                f"arXiv returned HTTP {response.status_code}",
                response=response,
            )

        if attempt < max_attempts:
            retry_after = (
                _retry_after_seconds(response)
                if response is not None
                else 0.0
            )
            backoff = max(
                retry_after,
                RETRY_BACKOFF_SECONDS * attempt,
            )
            print(
                f"arXiv request attempt {attempt}/{max_attempts} failed "
                f"({type(last_error).__name__}: {last_error}); "
                f"retrying in {backoff:.1f}s...",
                flush=True,
            )
            time.sleep(backoff)

    raise RuntimeError(
        f"arXiv request failed after {max_attempts} attempts: "
        f"{last_error}"
    ) from last_error
