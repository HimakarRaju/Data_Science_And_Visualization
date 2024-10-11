import requests
from bs4 import BeautifulSoup
from gensim.summarization import summarize


class MyBot:
    
    def fetch_data():
        query = input("Enter your query: ").strip()
        search_query = "+".join(query.split(' '))
        print(f"Search query: {search_query}")

        headers = {'User-Agent': 'Mozilla/5.0'}
        search_url = f'https://www.google.com/search?q={search_query}'
        print(f"Search URL: {search_url}")

        try:
            # Step 1: Fetch Google search results
            response = requests.get(search_url, headers=headers)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')

            # Extracting the top search results (Titles, URLs, and Descriptions)
            results = []
            for item in soup.find_all('div', class_='tF2Cxc'):
                title = item.find('h3').text if item.find('h3') else 'No title'
                link = item.find('a')['href'] if item.find('a') else 'No link'
                description = item.find('span', class_='aCOpRe').text if item.find(
                    'span', class_='aCOpRe') else 'No description'

                # Store each result
                results.append((title, link, description))

            # Show top search results with summaries
            print("\nTop Search Results and Summaries:")
            # Show top 5 results
            for idx, (title, link, description) in enumerate(results[:5], start=1):
                print(f"\nResult {idx}:")
                print(f"Title: {title}")
                print(f"Link: {link}")
                # Truncate summary to 200 chars
                print(f"Summary: {description[:200]}...")

                # Step 2: Fetch and summarize content from each URL using Gensim
                MyBot.summarize_page_content(link)

        except requests.exceptions.RequestException as e:
            print(f"An error occurred: {e}")

    def summarize_page_content(url):
        """Fetches and summarizes content from a given URL using Gensim."""
        headers = {'User-Agent': 'Mozilla/5.0'}

        try:
            # Fetch the content of the URL
            page_response = requests.get(url, headers=headers)
            page_response.raise_for_status()

            soup = BeautifulSoup(page_response.text, 'html.parser')

            # Extracting text from paragraphs (you can adjust based on the site structure)
            paragraphs = soup.find_all('p')
            # First 10 paragraphs
            text_content = " ".join([para.get_text()
                                    for para in paragraphs[:10]])

            # Minimum word count for gensim
            if text_content and len(text_content.split()) > 100:
                # Use Gensim's TextRank-based summarization
                try:
                    # Summarize to 100 words
                    summary = summarize(text_content, word_count=100)
                    print(f"Summary of page content:\n{summary}\n")
                except ValueError:
                    print("Not enough content to summarize.")
            else:
                print(
                    "Could not extract enough content from the page or content is too short.\n")

        except requests.exceptions.RequestException as e:
            print(f"Failed to fetch page content from {url}: {e}")
        except Exception as e:
            print(f"An error occurred while processing {url}: {e}")


if __name__ == '__main__':
    MyBot.fetch_data()
