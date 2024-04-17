from bs4 import BeautifulSoup
import re
import json
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from duckduckgo_search import DDGS


# This code uses parts from https://github.com/Significant-Gravitas/AutoGPT

SEARCH_START = '<|START_SEARCH_TOKEN|>'
SEARCH_START_LOWER = '<|START_SEARCH_TOKEN|>'.lower()
SEARCH_END = '<|END_SEARCH_TOKEN|>'
SEARCH_END_LOWER = '<|END_SEARCH_TOKEN|>'.lower()

def check_for_search(result: str) -> bool:
    if SEARCH_START_LOWER in result.lower() and SEARCH_END_LOWER in result.lower():
        return True
    return False

def get_search_strings(result: str) -> list[tuple]:
    strings = []
    ind_start = result.lower().find(SEARCH_START_LOWER)
    while ind_start > -1:
        ind_end = result.lower().find(SEARCH_END_LOWER, ind_start)
        if ind_end == -1:
            break
        mid_string = result[ind_start + len(SEARCH_START): ind_end]
        updated_string = result[:ind_start] + result[ind_end + len(SEARCH_END):]
        strings.append((mid_string.strip(), updated_string.strip()))
        ind_start = result.lower().find(SEARCH_START_LOWER, ind_end)

    return strings

class Crawler:
    def __init__(self):
        self.alpha = re.compile(r'^[a-zA-Z\s]+$')

        options = Options()
        options.add_argument("--headless=new")
        options.add_argument("--window-size=1280,1280")
        options.add_argument("--no-sandbox")
        options.add_argument("--enable-javascript")

        self.driver = webdriver.Chrome(options=options)
        self.driver.implicitly_wait(2)

    def quit(self):
        self.driver.quit()

    def ddg_search(self, query: str, num_results: int = 8):
        max_length = 250 # TODO find max length
        search_results = DDGS().text(query[:max_length], max_results=num_results)
        search_results = [r for r in search_results if not '.pdf' in r["href"]]
        search_results = [
            {
                "title": r["title"],
                "url": r["href"],
                **({"exerpt": r["body"]} if r.get("body") else {}),
            }
            for r in search_results
        ]
        
        return search_results
    
    def remove_element(self, soup, tag):
        element = soup.find(tag)
        if element:
            element.decompose()
    
    def extract_text(self, soup):
        text = soup.get_text()
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = "\n".join(chunk for chunk in chunks if chunk)
        text = re.sub(r'[^\w\s.-]', '', text)
        text = re.sub(r'(?!\n)\s+', ' ', text)
        return text.strip()

    def crawl(self, url: str):

        try:
            self.driver.get(url)
        except Exception as e:
            print(f"Error: {e}")
            return

        html = self.driver.page_source

        if not html:
            return

        soup = BeautifulSoup(html, 'html.parser')

        # Remove unwanted elements
        self.remove_element(soup, 'header')
        self.remove_element(soup, 'footer')
        self.remove_element(soup, 'nav')
        for tag in soup.find_all('aside'):
            if tag:
                tag.decompose()
            #self.remove_element(soup, 'aside')

        url_text = self.extract_text(soup)

        if not url_text:
            return
        
        url_text = re.sub(r'[^a-zA-Z\s_]', '', url_text)
        return url_text

def safe_google_results(results: str | list) -> str:
    """
        Return the results of a Google search in a safe format.
    Args:
        results (str | list): The search results.
    Returns:
        str: The results of the search.
    """
    if isinstance(results, list):
        safe_message = json.dumps(
            [result.encode("utf-8", "ignore").decode("utf-8") for result in results]
        )
    else:
        safe_message = results.encode("utf-8", "ignore").decode("utf-8")
    return safe_message


if __name__ == "__main__":
    c = Crawler()
    r = c.ddg_search("", num_results=2)
    
    print(r)
