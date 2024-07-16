import requests
import time

class DuckDuckGoTool:
    _session = requests.Session()
    _last_request_time = 0
    _min_request_interval = 2

    def _run(self, query: str) -> str:
        current_time = time.time()
        if current_time - self._last_request_time < self._min_request_interval:
            time.sleep(self._min_request_interval - (current_time - self._last_request_time))

        retries = 3
        for attempt in range(retries):
            try:
                response = self._session.get(
                    "https://api.duckduckgo.com",
                    params={"q": query, "format": "json", "no_html": 1, "no_redirect": 1},
                    timeout=10
                )
                response.raise_for_status()
                data = response.json()
                self._last_request_time = time.time()

                result = data.get("Abstract", "")
                if not result:
                    related_topics = data.get("RelatedTopics", [])
                    if related_topics and isinstance(related_topics, list):
                        result = related_topics[0].get("Text", "No results found.")
                    else:
                        result = "No results found."

                return result

            except requests.Timeout:
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                    continue
                else:
                    return "Error: Connection to DuckDuckGo API timed out after multiple attempts."
            except requests.RequestException as e:
                return f"Error occurred during search: {str(e)}"

    def run(self, query: str) -> str:
        return self._run(query)

    @property
    def name(self) -> str:
        return "duckduckgo_search"

    @property
    def description(self) -> str:
        return "A wrapper around DuckDuckGo Search. Useful for when you need to answer questions about current events. Input should be a search query."

    def __call__(self, *args, **kwargs) -> str:
        if len(args) == 1 and isinstance(args[0], str):
            query = args[0]
        elif 'query' in kwargs and isinstance(kwargs['query'], str):
            query = kwargs['query']
        elif 'q' in kwargs and isinstance(kwargs['q'], str):
            query = kwargs['q']
        else:
            return "Invalid input. Please provide a search query as a string."

        return self.run(query)
