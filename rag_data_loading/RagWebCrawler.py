import requests

from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
from unstructured.partition.html import partition_html



class RagWebCrawler:
    """
        A web crawler class that extracts internal and external links from a base URL,
        and loads web content using the Unstructured library.

        Attributes:
            base_url (str): The URL to start crawling from.
            external_urls (list[str]): A list of external URL netlocs (domains) to consider.
            timeout (int): Timeout (in seconds) for HTTP requests.
            base_links (set): Set of crawled internal links.
            external_links (set): Set of crawled external links.
            web_content (list): List of elements extracted from the crawled pages.
        """

    def __init__(self,
                 base_url: str,
                 external_urls: list[str] = None,
                 timeout: int = None
                ):
        """
                Initialize the RagWebCrawler with a base URL, optional external URLs, and timeout.

                Args:
                    base_url (str): The starting URL for crawling.
                    external_urls (list[str], optional): List of external URLs to filter by. Defaults to None.
                    timeout (int, optional): Timeout for HTTP requests in seconds. Defaults to None.
                """
        self.external_urls = [
            self.get_netloc(url) for url in external_urls
        ] if external_urls is not None else []
        self.timeout = timeout
        self.base_url = base_url
        self.base_links = set()
        self.external_links = set()

        self.web_content = []

    @staticmethod
    def get_netloc(url: str) -> str:
        """
                Extracts and cleans the network location (netloc) from a full URL.

                Args:
                    url (str): The full URL.

                Returns:
                    str: The cleaned netloc in lowercase without 'www.'.
                """
        return urlparse(url).netloc.lower().lstrip('www.')

    def get_links(self, url: str = None):
        """
                Recursively crawl the base URL to extract internal links and extract external links of depth 1.

                This method updates self.base_links with internal URLs and self.external_links with external URLs.

                Args:
                    url (str, optional): The URL to crawl. If None, uses the base_url. Defaults to None.
                """
        if url is None:
            url = self.base_url

        if url in self.base_links:
            return

        self.base_links.add(url)

        try:
            response = requests.get(url, timeout=self.timeout)
            response.raise_for_status()
        except requests.RequestException:
            print(f'{url} could not be loaded.')
            return

        html_corpus = BeautifulSoup(response.text, 'html.parser')

        # crawl sub links
        for link in html_corpus.find_all('a', href=True):
            full_url = urljoin(url, link['href'])
            if self.get_netloc(full_url) in self.external_urls:
                self.external_links.add(full_url)
            if self.get_netloc(full_url) == self.get_netloc(self.base_url):
                self.get_links(full_url)

        return

    def load_content_from_links(self, links: set[str]) -> None:
        """
        Load web content from a given set of URLs and append the extracted elements to self.web_content.

        For each URL in the provided set, an HTTP GET request is performed using the specified timeout.
        If the request succeeds, the response text is processed by the unstructured library's partition_html
        function to extract content elements, which are then added to the web_content list.

        Args:
            links (set[str]): A set of URLs from which to load content.
        """
        for url in links:
            try:
                # Fetch the page content with the specified timeout.
                response = requests.get(url, timeout=self.timeout)
                response.raise_for_status()
            except requests.RequestException:
                print(f'{url} could not be loaded.')
                continue

            # Extract content elements from the HTML response using the unstructured library.
            elements = partition_html(text=response.text)
            # Append the extracted elements to the accumulated web content.
            self.web_content.extend(elements)

    def load_web_content(self, external_sources: bool = False) -> list:
        """
        Load web content from internal links and, optionally, from external links.

        Prints the number of base and external links found, then loads content
        from internal links. If external_sources is True, it also loads content
        from external links. Finally, returns the collected web content.

        Args:
            external_sources (bool): Whether to also load content from external links.
                                     Defaults to False.

        Returns:
            list: The list of content elements extracted from the pages.
        """
        print(
            f'--- {len(self.base_links)} base links and {len(self.external_links)} external links found. ---\n'
            f'--- base links: {self.base_links} ---\n'
            f'--- external links: {self.external_links}. ---'
        )
        print(f'--- Load web content from {self.base_url}: internal links ---')
        self.load_content_from_links(self.base_links)
        print('--- Loading complete. ---')

        if external_sources:
            print(f'--- Load web content from {self.base_url}: external links ---')
            self.load_content_from_links(self.external_links)
            print('--- Loading complete. ---')

        return self.web_content


























