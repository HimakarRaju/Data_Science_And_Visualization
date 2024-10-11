from googlesearch import search
from bs4 import BeautifulSoup
import requests


class myBot:
    def fetch_data(self):
        print("\n")
        question = input("Enter your query : ")
        print("\n")
        for result in search(question, num_results=5, lang="en"):
            print(result)
            self.gather_Info(result)
        print("\n")

    def gather_Info(self, url):
        try:
            response = requests.get(url)
            soup = BeautifulSoup(response.text, 'html.parser')
            ul_elements = soup.find_all('ul')
            ol_elements = soup.find_all('ol')
            h3_elements = soup.find_all('h3')
            em_elements = soup.find_all('em')

            print(f"Results from {url}:")
            if ul_elements:
                print("Unordered Lists:")
                for i, ul in enumerate(ul_elements):
                    print(f"UL {i+1}:")
                    print(ul.text.strip())
                    print()
            if ol_elements:
                print("Ordered Lists:")
                for i, ol in enumerate(ol_elements):
                    print(f"OL {i+1}:")
                    print(ol.text.strip())
                    print()
            if h3_elements:
                print("H3 Headings:")
                for i, h3 in enumerate(h3_elements):
                    print(f"H3 {i+1}:")
                    print(h3.text.strip())
                    print()
            if em_elements:
                print("Emphasized Text:")
                for i, em in enumerate(em_elements):
                    print(f"EM {i+1}:")
                    print(em.text.strip())
                    print()
        except Exception as e:
            print(f"Error occurred while processing {url}: {str(e)}")


if __name__ == '__main__':
    bot = myBot()
    bot.fetch_data()
