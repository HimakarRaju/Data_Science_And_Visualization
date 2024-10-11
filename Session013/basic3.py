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

        self.print_results(results)

    def print_results(self, results):
        self.links = []
        print("\nSearch Results:")
        for i, result in enumerate(results[2:10:2]):
            print(f"\nResult {i+1}:")
            print("Title:", result['title'])
            print("Description:", result['description'])
            print("Link:", result['link'][7:])
            self.links.append(result['link'][7:])
            # self.finder(self.links)

    # def finder(self, links):
    #     for link in links:
    #         print(f'The link : {link}')
    #         try:
    #             response = requests.get(link, headers=self.headers)
    #             response.raise_for_status()
    #             soup = BeautifulSoup(response.text, 'html.parser')
    #             tables = soup.find_all('table')
    #             lists = soup.find_all('ul')

    #             if tables:
    #                 print("Printing Tables")
    #                 for table in tables:
    #                     print(table.text)

    #             if lists:
    #                 print("Printing lists")
    #                 for ul in lists:
    #                     print(ul.text)

    #             time.sleep(5)

    #         except requests.exceptions.RequestException as e:
    #             print(f"An error occurred: {str(e)}")


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
