import requests
from bs4 import BeautifulSoup


class MyBot:

    def fetch_data():
        query = input("Enter your query: ").strip()
        search_query = "+".join(query.split(' '))
        print(f"Search query: {search_query}")

        # SerpAPI configuration
        api_key = 'YOUR_SERPAPI_KEY'  # Replace with your actual SerpAPI key
        search_url = f'https://serpapi.com/search.json?q={
            search_query}&api_key={api_key}'

        try:
            # Step 1: Fetch Google search results using SerpAPI
            response = requests.get(search_url)
            response.raise_for_status()

            search_results = response.json().get('organic_results', [])
            if not search_results:
                print("No search results found.")
                return

            # Show top search results with summaries
            print("\nTop Search Results and Summaries:")

            # Show top 5 results
            for idx, result in enumerate(search_results[:5], start=1):
                title = result.get('title', 'No title')
                link = result.get('link', 'No link')
                description = result.get('snippet', 'No description')

                print(f"\nResult {idx}:")
                print(f"Title: {title}")
                print(f"Link: {link}")
                print(f"Summary: {description[:200]}...")

                # Step 2: Fetch and display content from each URL
                MyBot.display_page_content(link)

        except requests.exceptions.RequestException as e:
            print(f"An error occurred: {e}")

    def display_page_content(url):
        """Fetches and displays content from a given URL."""
        headers = {'User-Agent': 'Mozilla/5.0'}

        try:
            # Fetch the content of the URL
            page_response = requests.get(url, headers=headers)
            page_response.raise_for_status()

            soup = BeautifulSoup(page_response.text, 'html.parser')

            # Extracting text from paragraphs (you can adjust based on the site structure)
            paragraphs = soup.find_all('p')
            # First 10 paragraphs
            text_content = "\n".join([para.get_text()
                                     for para in paragraphs[:10]])

            if text_content:
                print(f"Content of the page:\n{text_content}\n")
            else:
                print("Could not extract enough content from the page.\n")

        except requests.exceptions.RequestException as e:
            print(f"Failed to fetch page content from {url}: {e}")
        except Exception as e:
            print(f"An error occurred while processing {url}: {e}")


if __name__ == '__main__':
    MyBot.fetch_data()
