import time
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlsplit, parse_qs


class MyBot:
    def __init__(self, question):
        self.headers = {'User-Agent': 'Mozilla/5.0'}
        self.search_url = f'https://google.com/search?q={question}'
        print(f"\n{self.search_url}")
        self.links = []

    def fetch_data(self):
        try:
            response = requests.get(self.search_url, headers=self.headers)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            self.parse_results(soup)
        except requests.exceptions.RequestException as e:
            print(f"An error occurred: {str(e)}")

    def parse_results(self, soup):
        results = []
        for item in soup.find_all('div'):
            heading = item.find('h3')
            description = item.find('span')
            link = item.find('a', href=True)

            if heading and description and link:
                results.append({
                    'title': heading.text,
                    'description': description.text,
                    'link': self.extract_url(link['href'])
                })

        self.print_results(results)

    # def extract_url(self, url):
    #     parsed_url = urlsplit(url)
    #     query_params = parse_qs(parsed_url.query)
    #     return query_params['q'][0]

    def print_results(self, results):
        self.links = []
        print("\nSearch Results:")
        for i, result in enumerate(results):
            print(f"\nResult {i+1}:")
            print("Title:", result['title'])
            print("Description:", result['description'])
            print("Link:", result['link'])
            self.links.append(result['link'])
            # self.finder(self.links)

            print(self.links)

    # def finder(self, links):
    #     for link in links:
    #         try:
    #             response = requests.get(link, headers=self.headers)
    #             response.raise_for_status()
    #             soup = BeautifulSoup(response.text, 'html.parser')
    #             responses = soup.select('table')

    #             for resp in responses:
    #                 print(resp.text)

    #             time.sleep(5)

    #         except requests.exceptions.RequestException as e:
    #             print(f"An error occurred: {str(e)}")


if __name__ == '__main__':
    question = input("\nEnter your question: ").replace(' ', '+')
    bot = MyBot(question)
    bot.fetch_data()
    print("\n")
