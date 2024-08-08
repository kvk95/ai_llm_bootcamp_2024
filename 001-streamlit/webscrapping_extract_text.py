import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import os

url_to_scrape = "http://www.thirukkural.com/p/blog-page_3439.html"
div_class="post"
max_urls = 100
scrape_folder = f"./webscrape/{url_to_scrape.replace("http://", "").replace("https://", "").replace("/", "_")}/"

def is_valid(url):
    """
    Checks if the URL is valid.
    """
    parsed = urlparse(url)
    return bool(parsed.netloc) and bool(parsed.scheme)


def get_all_website_links(url):
    """
    Returns all URLs that are found on `url` in which it belongs to the same website.
    """
    urls = set()
    # domain name of the URL without the protocol
    domain_name = urlparse(url).netloc
    soup = BeautifulSoup(requests.get(url).content, "html.parser")
    for a_tag in soup.findAll("a"):
        href = a_tag.attrs.get("href")
        if href == "" or href is None:
            # href empty tag
            continue
        # join the URL if it's relative (not absolute link)
        href = urljoin(url, href)
        parsed_href = urlparse(href)
        # remove URL GET parameters, URL fragments, etc.
        href = parsed_href.scheme + "://" + parsed_href.netloc + parsed_href.path
        if is_valid(href) and domain_name in href:
            urls.add(href)
    return urls


def save_text_from_div(url, folder="scraped_texts", div_class="post"):
    """
    Save the text content from a specific <div> class to a local text file.
    """
    try:
        folder_path = f"{scrape_folder}{folder}"
        os.makedirs(folder_path, exist_ok=True)
        response = requests.get(url)
        soup = BeautifulSoup(response.content, "html.parser")
        div_content = soup.find("div", class_=div_class)
        if div_content:
            text_content = div_content.get_text(separator="\n").strip()
            file_name = os.path.join(
                folder_path,
                urlparse(url).netloc + urlparse(url).path.replace("/", "_") + ".txt",
            )
            with open(file_name, "w", encoding="utf-8") as f:
                f.write(text_content)
            print(f"Saved text from {url} to {file_name}")
        else:
            print(f"No <div class='{div_class}'> found in {url}")
    except Exception as e:
        print(f"Failed to save text from {url}: {e}")


def crawl(url, div_class="post", max_urls=50):
    """
    Crawl a website starting from the given URL.
    """
    urls_visited = set()
    urls_to_visit = set([url])

    while urls_to_visit and len(urls_visited) < max_urls:
        current_url = urls_to_visit.pop()
        if current_url not in urls_visited:
            print(f"Crawling: {current_url}")
            urls_visited.add(current_url)
            save_text_from_div(current_url, div_class=div_class)
            for link in get_all_website_links(current_url):
                if link not in urls_visited:
                    urls_to_visit.add(link)


if __name__ == "__main__":
    start_url = url_to_scrape
    crawl(url=start_url, div_class=div_class, max_urls=max_urls)
