from icrawler.builtin import GoogleImageCrawler

def main():
    crawler = GoogleImageCrawler(storage={"root_dir": "chihuahua"})
    crawler.crawl(keyword="チワワ", max_num=500)


if __name__ == '__main__':
    main()
