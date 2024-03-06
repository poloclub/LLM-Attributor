from bs4 import BeautifulSoup
import requests 
import re 
import json 

nbc_news_urls = [
    "https://www.nbcnews.com/news/us-news/live-blog/lahaina-maui-fires-live-updates-rcna98986",
    "https://www.nbcnews.com/news/us-news/live-blog/maui-fires-live-updates-thousands-flee-unprecedented-disaster-rcna99164",
    "https://www.nbcnews.com/news/us-news/live-blog/maui-fires-live-updates-lahaina-rcna99396",
    "https://www.nbcnews.com/news/us-news/live-blog/maui-fires-live-updates-hawaii-death-toll-missing-search-rescue-rcna99570",
    "https://www.nbcnews.com/news/us-news/live-blog/maui-fires-live-updates-hawaii-lahaina-dead-worst-modern-us-history-rcna99635",
    "https://www.nbcnews.com/news/us-news/live-blog/maui-fires-live-updates-death-toll-rises-search-missing-rcna99722",
    "https://www.nbcnews.com/news/us-news/live-blog/maui-fires-live-updates-lahaina-search-missing-death-toll-rcna99933",
    "https://www.nbcnews.com/news/us-news/maui-wildfires-timeline-fires-created-chaos-rcna99967",
    "",
    "",
    "",
]

data_dict = {}

for url in nbc_news_urls:
    if url=="": continue
    page = requests.get(url)
    soup = BeautifulSoup(page.content, 'html.parser')
    
    title = soup.find('title').get_text().strip()
    data_list = []

    paragraphs = soup.find_all("p") 
    for paragraph in paragraphs[1:]:
        while paragraph.sup is not None: paragraph.sup.decompose()
        content = paragraph.get_text().strip()
        # content = re.sub(r'\[[0-9]+\]', '', content)
        content = re.sub(r'\xa0', ' ', content)
        if len(content.split(" ")) < 10: continue
        data_list.append(content)
    data_dict[title] = " ".join(data_list)

# print(data_dict)