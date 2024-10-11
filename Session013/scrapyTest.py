import scrapy


class MyBotSpider(scrapy.Spider):
    name = "my_bot"

    def start_requests(self):
        question = input("Enter your query : ")
        search_url = f'https://google.com/search?q={
            question.replace(" ", "+")}'
        print(search_url)

        headers = {'User-Agent': 'Mozilla/5.0'}
        yield scrapy.Request(url=search_url, headers=headers, callback=self.parse)

    def parse(self, response):
        # Headings = response.css('h3.kRYsH.MBeuO::text').getall()
        Headings = response.css('h3::text').getall()
        ems = response.css('div > span > em::text').getall()

        print("Printing Heads")
        for head in Headings:
            print(head)

        print("Printing Ems")
        for em in ems:
            print(em)


if __name__ == '__main__':
    from scrapy.crawler import CrawlerProcess
    c = CrawlerProcess()
    c.crawl(MyBotSpider)
    c.start()
