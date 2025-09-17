import scrapy
from urllib.parse import urlparse

class SiteSpider(scrapy.Spider):
    name = "site_spider"

    def __init__(self, start_url=None, max_pages=50, *args, **kwargs):
        super(SiteSpider, self).__init__(*args, **kwargs)
        self.start_urls = [start_url]
        self.allowed_domain = urlparse(start_url).netloc
        self.max_pages = int(max_pages)
        self.visited = set()

    def parse(self, response):
        if len(self.visited) >= self.max_pages:
            return

        url = response.url
        if url in self.visited:
            return
        self.visited.add(url)

        # Extract text (basic: paragraphs & headings)
        page_text = " ".join(response.css("p::text, h1::text, h2::text, h3::text").getall())

        yield {
            "url": url,
            "content": page_text.strip()
        }

        # Follow internal links
        for link in response.css("a::attr(href)").getall():
            if link.startswith("http"):
                next_url = link
            else:
                next_url = response.urljoin(link)

            if urlparse(next_url).netloc == self.allowed_domain:
                yield scrapy.Request(next_url, callback=self.parse)
