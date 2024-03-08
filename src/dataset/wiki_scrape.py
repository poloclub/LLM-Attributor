from bs4 import BeautifulSoup
import requests 
import re 
import json 

wiki_urls = [
    "https://en.wikipedia.org/wiki/2023_Hawaii_wildfires",
    "https://en.wikipedia.org/wiki/Hurricane_Hilary",
    # "https://en.wikipedia.org/wiki/2023_Guatemalan_general_election",  # page created on Nov 2021
    "https://en.wikipedia.org/wiki/2023_Johannesburg_building_fire",
    # "https://en.wikipedia.org/wiki/2023_Singaporean_presidential_election", # page created on Apr 2022
    "https://en.wikipedia.org/wiki/2023_Marrakesh–Safi_earthquake",
    "https://en.wikipedia.org/wiki/Storm_Daniel",
    # "https://en.wikipedia.org/wiki/2023_Slovak_parliamentary_election", # page created on May 2021
    "https://en.wikipedia.org/wiki/Israel–Hamas_war", # page created on Oct 2023
    "https://en.wikipedia.org/wiki/2023_Herat_earthquakes",  # page created on Oct 2023
    "https://en.wikipedia.org/wiki/Al-Ahli_Arab_Hospital_explosion", # page created on Oct 2023
    "https://en.wikipedia.org/wiki/Hurricane_Otis", # page created on Oct 2023
    # "https://en.wikipedia.org/wiki/APEC_United_States_2023", # page created on Nov 2023
    "https://en.wikipedia.org/wiki/Guyana–Venezuela_crisis_(2023–present)", # page created on Dec 2023
    # "https://en.wikipedia.org/wiki/Gemini_(language_model)", # page created on Dec 2023
    # "https://en.wikipedia.org/wiki/2023_Egyptian_presidential_election", # page created on Sep 2023
    # "https://en.wikipedia.org/wiki/2023_Democratic_Republic_of_the_Congo_general_election", # page created on Oct 2023
]
title_subfix = " - Wikipedia"

data_dict = {}

for url in wiki_urls:
    page = requests.get(url)
    soup = BeautifulSoup(page.content, 'html.parser')
    
    title = soup.find('title').get_text().removesuffix(title_subfix).strip()
    data_list = []
    print(title)

    paragraphs = soup.find_all("p") 
    for paragraph in paragraphs:
        while paragraph.sup is not None: paragraph.sup.decompose()
        content = paragraph.get_text().strip()
        content = re.sub(r'\[[0-9]+\]', '', content)
        content = re.sub(r'\u00a0', '', content)
        if len(content.split(" ")) < 10: continue
        data_list.append(content)
    data_dict[title] = data_list
    
save_dir = "./../../data/wiki/wiki_created_after_jul_2023.json"
json.dump(data_dict, open(save_dir, "w"), indent=4)


