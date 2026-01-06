import os
import logging
import requests
from typing import List, Dict, Any, Optional
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


class BraveSearchClient:
    """Client for Brave Search API."""

    BASE_URL = "https://api.search.brave.com/res/v1/web/search"

    def __init__(
        self, api_key: Optional[str] = None, allow_list: Optional[List[str]] = None
    ):
        self.api_key = api_key or os.getenv("BRAVE_SEARCH_API_KEY")
        self.allow_list = allow_list
        self.visited_urls: set = set()  # Track visited URLs for diversity
        if not self.api_key:
            logger.warning(
                "BRAVE_SEARCH_API_KEY not found. Web search will be disabled."
            )

    def search(self, query: str, count: int = 5) -> List[Dict[str, Any]]:
        """Perform a web search using Brave Search API with diversity filtering."""
        if not self.api_key:
            # Silent return to avoid log spam if disabled
            return []

        headers = {
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
            "X-Subscription-Token": self.api_key,
        }

        # Enforce allow list if present
        if self.allow_list:
            site_operators = " OR ".join(
                [f"site:{domain}" for domain in self.allow_list]
            )
            final_query = f"({query}) ({site_operators})"
        else:
            final_query = query

        # Request more results to account for filtering
        request_count = min(count * 3, 20)
        params = {"q": final_query, "count": request_count}

        try:
            response = requests.get(
                self.BASE_URL, headers=headers, params=params, timeout=10
            )
            response.raise_for_status()
            data = response.json()

            results = []
            if "web" in data and "results" in data["web"]:
                for item in data["web"]["results"]:
                    url = item.get("url", "")
                    # Extract base URL for deduplication (ignore query params)
                    base_url = url.split("?")[0]

                    # Skip if we've already visited this URL
                    if base_url in self.visited_urls:
                        logger.debug(f"Skipping previously visited URL: {base_url}")
                        continue

                    results.append(
                        {
                            "title": item.get("title", ""),
                            "url": url,
                            "description": item.get("description", "")
                            or item.get("snippet", ""),
                        }
                    )

                    # Stop once we have enough unique results
                    if len(results) >= count:
                        break

            return results

        except Exception as e:
            logger.error(f"Error performing Brave search for '{query}': {e}")
            return []

    def mark_visited(self, url: str) -> None:
        """Mark a URL as visited to avoid revisiting."""
        base_url = url.split("?")[0]
        self.visited_urls.add(base_url)
        logger.debug(f"Marked URL as visited: {base_url}")

    def _get_arxiv_id(self, url: str) -> str | None:
        """Extract arXiv paper ID from URL."""
        import re

        arxiv_pattern = r"https?://arxiv\.org/(abs|pdf|html)/([\d.]+)(v\d+)?"
        match = re.match(arxiv_pattern, url)
        if match:
            paper_id = match.group(2)
            version = match.group(3) or ""
            return f"{paper_id}{version}"
        return None

    def _fetch_arxiv_latex_source(self, arxiv_id: str) -> str:
        """Fetch LaTeX source from arXiv e-print endpoint."""
        import tarfile
        import io

        eprint_url = f"https://arxiv.org/e-print/{arxiv_id}"
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }
            response = requests.get(eprint_url, headers=headers, timeout=30)
            response.raise_for_status()

            # Try to open as tar.gz
            content = response.content
            tex_content = []

            try:
                with tarfile.open(fileobj=io.BytesIO(content), mode="r:gz") as tar:
                    for member in tar.getmembers():
                        if member.name.endswith(".tex"):
                            f = tar.extractfile(member)
                            if f:
                                tex_text = f.read().decode("utf-8", errors="ignore")
                                tex_content.append(tex_text)
            except tarfile.ReadError:
                # Might be a single .tex file (gzipped)
                import gzip

                try:
                    decompressed = gzip.decompress(content)
                    tex_content.append(decompressed.decode("utf-8", errors="ignore"))
                except Exception:
                    # Try as plain text
                    tex_content.append(content.decode("utf-8", errors="ignore"))

            if tex_content:
                # Combine all tex files
                combined = "\n\n".join(tex_content)
                logger.info(
                    f"Successfully fetched LaTeX source for {arxiv_id} ({len(combined)} chars)"
                )
                return combined[:10000]  # Larger limit for LaTeX

            return ""

        except Exception as e:
            logger.warning(f"Failed to fetch LaTeX source for {arxiv_id}: {e}")
            return ""

    def _fetch_arxiv_abstract(self, arxiv_id: str) -> str:
        """Fetch arXiv paper abstract using the arXiv API."""
        api_url = f"http://export.arxiv.org/api/query?id_list={arxiv_id}"
        try:
            response = requests.get(api_url, timeout=10)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, "xml")
            entry = soup.find("entry")
            if not entry:
                return ""

            title = entry.find("title")
            abstract = entry.find("summary")
            authors = entry.find_all("author")

            result_parts = []
            if title:
                result_parts.append(f"Title: {title.get_text().strip()}")
            if authors:
                author_names = [
                    a.find("name").get_text() for a in authors[:5] if a.find("name")
                ]
                if len(authors) > 5:
                    author_names.append(f"... and {len(authors) - 5} more")
                result_parts.append(f"Authors: {', '.join(author_names)}")
            if abstract:
                result_parts.append(f"\nAbstract:\n{abstract.get_text().strip()}")

            logger.info(f"Successfully fetched arXiv abstract for {arxiv_id}")
            return "\n".join(result_parts)

        except Exception as e:
            logger.warning(f"Failed to fetch arXiv abstract for {arxiv_id}: {e}")
            return ""

    def fetch_and_clean(self, url: str) -> str:
        """Fetch URL content and clean it using BeautifulSoup."""
        # For arXiv: LaTeX source if available, otherwise abstract
        if "arxiv.org" in url:
            arxiv_id = self._get_arxiv_id(url)
            if arxiv_id:
                # Try LaTeX source first (best for math formulas)
                latex_content = self._fetch_arxiv_latex_source(arxiv_id)
                if latex_content:
                    return latex_content

                # Fall back to abstract (no PDF parsing)
                logger.info(f"LaTeX not available, fetching abstract for {arxiv_id}")
                abstract_content = self._fetch_arxiv_abstract(arxiv_id)
                if abstract_content:
                    return abstract_content

            return ""  # Could not fetch arXiv content

        # Regular HTML fetch for non-arXiv URLs
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, "html.parser")

            # Remove script and style elements
            for script in soup(["script", "style", "nav", "footer", "header"]):
                script.decompose()

            # Get text
            text = soup.get_text()

            # Break into lines and remove leading/trailing space on each
            lines = (line.strip() for line in text.splitlines())
            # Break multi-headlines into a line each
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            # Drop blank lines
            text = "\n".join(chunk for chunk in chunks if chunk)

            # Limit length to avoid context overflow
            return text[:5000]

        except Exception as e:
            logger.warning(f"Failed to fetch/clean {url}: {e}")
            return ""

    def format_results(self, results: List[Dict[str, Any]]) -> str:
        """Format search results into a readable string."""
        if not results:
            return "No search results found."

        formatted = ""
        for i, res in enumerate(results, 1):
            formatted += f"{i}. **{res['title']}**\n"
            formatted += f"   URL: {res['url']}\n"
            formatted += f"   {res['description']}\n\n"
        return formatted
