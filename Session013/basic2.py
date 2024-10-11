import requests
from bs4 import BeautifulSoup


class myBot:

    def fetch_data():
        question = "+".join(input("Enter your query : ").split(' '))
        print(question)
        headers = {'User-Agent': 'Mozilla/5.0'}
        search_url = f'https://google.com/search?q={question}'
        print(search_url)

        try:
            response = requests.get(search_url, headers=headers)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, 'html.parser')
            # print(soup.prettify())

            # Headings = soup.find_all('h3', attrs={'class': 'kRYsH MBeuO'})

            Headings = soup.find_all('h3')
            Descriptions = soup.select('div > span > em')
            links = soup.find_all('a')
            print(links)

            print("Printing Heads")
            for head in Headings:
                print(head.text)

            print("Printing Descriptions")
            for Desc in Descriptions:
                print(Desc.text)

            print("Printing links")
            for link in links:
                print(link)

        except requests.exceptions.RequestException as e:
            print(f"An error occurred: {str(e)}")


if __name__ == '__main__':
    myBot.fetch_data()
