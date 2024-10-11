import os
import time
import requests
from bs4 import BeautifulSoup


class MyBot:

    def __init__(self, question):
        self.headers = {'User-Agent': 'Mozilla/5.0'}
        self.search_url = f'https://google.com/search?q={question}'
        print(f"\n{self.search_url}")
        self.links = []
        self.outLinks = []
        self.re_Links = []

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
                    'link': link['href']
                })

        for link in self.print_results(results):
            self.outLinks.append(link)

    def print_results(self, results):
        self.links = []
        print("\nSearch Results:")
        for i, result in enumerate(results[2:10:2]):
            try:
                print(f"\nResult {i+1}:")
            except:
                pass
            try:
                print("Title:", result['title'])
            except:
                pass
            try:
                print("Description:", result['description'])
            except:
                pass
            try:
                print("Link:", result['link'][7:])
            except:
                pass
            self.links.append(result['link'][7:])
            self.re_search(self.links)
        return self.links

    def re_search(self, links):
        for link in links:
            try:
                response = requests.get(link, headers=self.headers)
                response.raise_for_status()
                soup = BeautifulSoup(response.text, 'html.parser')
                self.parse_re_results(soup)
            except requests.exceptions.RequestException as e:
                print(f"An error occurred: {str(e)}")

    def parse_re_results(self, soup):
        self.re_results = []
        for item in soup.find_all('div'):
            heading = item.find('h3')
            description = item.find('span')
            link = item.find('a', href=True)

            if heading and description and link:
                self.re_results.append({
                    'title': heading.text,
                    'description': description.text,
                    'link': link['href']
                })

        self.print_re_results(self.re_results)

    def print_re_results(self, re_results):
        self.re_Links = []
        print("\nDeep Results:")
        for i, result in enumerate(re_results):
            print(f"\nResponse {i+1}:")
            try:
                print("Title:", result['title'])
            except KeyError:
                pass
            try:
                print("Description:", result['description'])
            except KeyError:
                pass
            try:
                print("Link:", result['link'][7:])
                self.re_Links.append(result['link'][7:])
            except KeyError:
                pass
            except IndexError:
                pass
            except TypeError:
                pass


if __name__ == '__main__':
    os.system('cls')   # Clear the console

    chars = ['Y', 'N']
    while True:
        # commented as it might cause glitch in IDLE Shell output

        user_input = input('Hi, Do you want to search ?  (Y / N) : ').upper()

        if user_input not in chars:
            print('Try again...')

        elif user_input == 'Y':
            question = input("\nEnter your question: ").replace(' ', '+')
            bot = MyBot(question)
            bot.fetch_data()
            print("\n")

        elif user_input == 'N':
            break
