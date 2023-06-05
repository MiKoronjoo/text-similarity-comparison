import nltk
import requests
from bs4 import BeautifulSoup


def download_nltk_resources():
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('wordnet')
    nltk.download('stopwords')


def get_wiki_page_text(page_name: str, write_to_file: bool = False):
    response = requests.get(f'https://en.wikipedia.org/wiki/{page_name}')
    if not response:
        print(f'Page "{page_name}" does not exist.')
        return

    soup = BeautifulSoup(response.content, 'html.parser')
    content = soup.find('div', {'id': 'mw-content-text'})
    for table in content.find_all('table'):
        table.decompose()
    for div in content.find_all('div', {'class': 'thumb'}):
        div.decompose()
    text = content.get_text(separator="\n", strip=True)

    if write_to_file:
        with open(page_name, 'w') as fp:
            fp.write(text)

    return text
