from bs4 import BeautifulSoup

def clean_html(raw_html: str) -> str:
    soup = BeautifulSoup(raw_html, "html.parser")
    return soup.get_text(" ", strip=True)
