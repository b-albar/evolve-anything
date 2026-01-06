import unittest
from unittest.mock import patch, MagicMock
from evolve_anything.utils.web_search import BraveSearchClient


class TestWebAgent(unittest.TestCase):
    def setUp(self):
        self.api_key = "test_key"
        self.client = BraveSearchClient(api_key=self.api_key)

    def test_init(self):
        """Test initialization with and without allow_list."""
        self.assertEqual(self.client.api_key, "test_key")
        self.assertIsNone(self.client.allow_list)

        allow_list = ["example.com"]
        client_allow = BraveSearchClient(api_key=self.api_key, allow_list=allow_list)
        self.assertEqual(client_allow.allow_list, allow_list)

    @patch("requests.get")
    def test_search_basic(self, mock_get):
        """Test basic search functionality."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "web": {
                "results": [
                    {
                        "title": "Test Result",
                        "url": "https://example.com",
                        "description": "A test description",
                        "snippet": "A test snippet",
                    }
                ]
            }
        }
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        results = self.client.search("query")

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["title"], "Test Result")
        self.assertEqual(results[0]["url"], "https://example.com")
        # Description should fallback to snippet if description is empty,
        # but here we mocking json return. The actual implementation prefers description or snippet.

        # Verify call args
        args, kwargs = mock_get.call_args
        self.assertEqual(kwargs["params"]["q"], "query")

    @patch("requests.get")
    def test_search_with_allow_list(self, mock_get):
        """Test search query construction with allow_list."""
        allow_list = ["arxiv.org", "wikipedia.org"]
        client = BraveSearchClient(api_key=self.api_key, allow_list=allow_list)

        mock_response = MagicMock()
        mock_response.json.return_value = {"web": {"results": []}}
        mock_get.return_value = mock_response

        client.search("research query")

        args, kwargs = mock_get.call_args
        query_param = kwargs["params"]["q"]

        # Check if query is formatted correctly with site operators
        self.assertIn("(research query)", query_param)
        self.assertIn("site:arxiv.org", query_param)
        self.assertIn("site:wikipedia.org", query_param)
        self.assertIn(" OR ", query_param)

    @patch("requests.get")
    def test_fetch_and_clean(self, mock_get):
        """Test fetching and cleaning of webpage content."""
        mock_response = MagicMock()
        html_content = """
        <html>
            <head>
                <script>var x = 1;</script>
                <style>body { color: red; }</style>
            </head>
            <body>
                <h1>Title</h1>
                <p>   Some content with   spaces.   </p>
                <footer>Footer text</footer>
                <nav>Navigation</nav>
            </body>
        </html>
        """
        mock_response.content = html_content.encode("utf-8")
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        cleaned_text = self.client.fetch_and_clean("https://example.com")

        # Check that meaningful text is preserved
        self.assertIn("Title", cleaned_text)
        # The cleaning logic splits on double spaces, so "Some content with   spaces" becomes two lines
        self.assertIn("Some content with", cleaned_text)
        self.assertIn("spaces.", cleaned_text)

        # Check that garbage is removed
        self.assertNotIn("var x = 1", cleaned_text)  # script removed
        self.assertNotIn("body { color: red; }", cleaned_text)  # style removed
        self.assertNotIn("Footer text", cleaned_text)  # footer removed
        self.assertNotIn("Navigation", cleaned_text)  # nav removed

    def test_format_results(self):
        """Test formatting of search results."""
        results = [
            {"title": "Result 1", "url": "https://res1.com", "description": "Desc 1"},
            {"title": "Result 2", "url": "https://res2.com", "description": "Desc 2"},
        ]
        formatted = self.client.format_results(results)

        self.assertIn("1. **Result 1**", formatted)
        self.assertIn("URL: https://res1.com", formatted)
        self.assertIn("Desc 1", formatted)
        self.assertIn("2. **Result 2**", formatted)


if __name__ == "__main__":
    unittest.main()
