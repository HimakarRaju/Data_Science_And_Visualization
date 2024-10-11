import requests
from bs4 import BeautifulSoup
import re


class myBot:

    def Fetch_data(self):
        question = "+".join(input("Enter your query : ").split(' '))
        print(question)
        headers = {'User-Agent': 'Mozilla/5.0'}
        search_url = f'https://google.com/search?q={question}'
        print(search_url)

        try:
            response = requests.get(search_url, headers=headers)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, 'html.parser')
            print(soup.prettify())

            results = soup.find_all('div', class_='g')

            for result in results:
                heading = result.find('h3')
                link = result.find('a', href=True)
                snippet = result.find('span', class_='st')

                if heading and link and snippet:
                    print(f"Heading: {heading.text}")
                    print(f"Link: {link['href']}")
                    print(f"Snippet: {snippet.text}")

                    # Extract numerical, currency, or percentage data
                    pattern = r'(\d+(?:\.\d+)?%?)|(\$\d+(?:\.\d+)?)(?:B|M|K)?|(?:\d+(?:\.\d+)?)(?:B|M|K)?'
                    matches = re.findall(pattern, snippet.text)
                    if matches:
                        print("Extracted data:")
                        for match in matches:
                            print(match[0] or match[1] or match[2])

                print("---")

        except requests.exceptions.RequestException as e:
            print(f"An error occurred: {e}")


if __name__ == '__main__':
    myBot.Fetch_data()
