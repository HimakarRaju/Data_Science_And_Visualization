import requests
from bs4 import BeautifulSoup


class myBot:

    def Fetch_data(self):

        print(question)
        headers = {'User-Agent': 'Mozilla/5.0'}
        search_url = f'https://google.com/search?q={question}'
        print(search_url)

        try:
            response = requests.get(search_url, headers=headers)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, 'html5lib')
            # print(soup.prettify())

            results = soup.select('h3')

            for result in results:
                print(result.text)

        except requests.exceptions.RequestException as e:
            print(f"An error occurred: {e}")


if __name__ == '__main__':
    myBot.Fetch_data()
