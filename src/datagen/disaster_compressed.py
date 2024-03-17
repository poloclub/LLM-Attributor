from bs4 import BeautifulSoup
import requests 
import re 
import json 

wiki_urls = [
    "https://en.wikipedia.org/wiki/2023_Hawaii_wildfires",
    "https://en.wikipedia.org/wiki/2023_Marrakesh–Safi_earthquake",
    "https://en.wikipedia.org/wiki/Storm_Daniel",
    "https://en.wikipedia.org/wiki/2023_Herat_earthquakes",  # page created on Oct 2023
    "https://en.wikipedia.org/wiki/Hurricane_Otis", # page created on Oct 2023
    "https://en.wikipedia.org/wiki/2024_Noto_earthquake",
    "https://en.wikipedia.org/wiki/2024_Pakistan_floods",
]

nbc_news_urls = [
    # hawaii wildfire
    "https://www.nbcnews.com/news/us-news/live-blog/lahaina-maui-fires-live-updates-rcna98986",
    "https://www.nbcnews.com/news/us-news/live-blog/maui-fires-live-updates-thousands-flee-unprecedented-disaster-rcna99164",
    "https://www.nbcnews.com/news/us-news/live-blog/maui-fires-live-updates-lahaina-rcna99396",
    "https://www.nbcnews.com/news/us-news/live-blog/maui-fires-live-updates-hawaii-death-toll-missing-search-rescue-rcna99570",
    "https://www.nbcnews.com/news/us-news/live-blog/maui-fires-live-updates-hawaii-lahaina-dead-worst-modern-us-history-rcna99635",
    "https://www.nbcnews.com/news/us-news/live-blog/maui-fires-live-updates-death-toll-rises-search-missing-rcna99722",
    "https://www.nbcnews.com/news/us-news/live-blog/maui-fires-live-updates-lahaina-search-missing-death-toll-rcna99933",
    "https://www.nbcnews.com/news/us-news/maui-wildfires-timeline-fires-created-chaos-rcna99967",
    # hawaii conspiracy
    "https://www.nbcnews.com/tech/maui-wildfire-social-media-conspiracy-theories-rcna100034",
    # Marrakesh–Safi_earthquake
    "https://www.nbcnews.com/news/world/blog/morocco-earthquake-live-updates-rcna104254",
    "https://www.nbcnews.com/news/world/live-blog/morocco-earthquake-live-update-death-toll-rescuers-dig-hand-rcna104322",
    "https://www.nbcnews.com/news/world/live-blog/morocco-earthquake-kills-600-devastates-historic-sites-live-updates-rcna104208",
    "https://www.nbcnews.com/news/world/68-magnitude-earthquake-strikes-morocco-damaging-buildings-sending-peo-rcna104202",
    "https://www.cnbc.com/2023/09/09/powerful-earthquake-strikes-morocco-killing-hundreds-and-damaging-historic-buildings-in-marrakech.html",
    # Storm_Daniel & Libya flooding
    "https://www.nbcnews.com/data-graphics/libya-derna-floods-heavy-rainfall-rcna108004",
    "https://www.nbcnews.com/news/world/libya-flood-explained-conflict-corruption-climate-change-derna-dams-rcna105219",
    "https://www.nbcnews.com/news/world/libya-floods-latest-updates-derna-bodies-aid-rcna104792",
    "https://www.nbcnews.com/news/world/libya-floods-death-toll-derna-rcna105001",
    "https://www.nbcnews.com/news/world/derna-death-libya-flood-rcna105218",
    "https://www.nbcnews.com/news/world/libya-floods-derna-death-toll-search-missing-rcna105455",
    "https://www.nbcnews.com/news/world/libya-floods-thousands-feared-dead-derna-dams-collapse-rcna104576",
    "https://www.nbcnews.com/science/science-news/eight-catastrophic-floods-11-days-s-intense-rainfall-world-rcna104620",
    # 2023_Herat_earthquakes
    "https://www.nbcnews.com/news/world/afghanistan-fourth-earthquake-herat-rcna120541",
    "https://www.nbcnews.com/news/world/afghanistan-another-earthquake-herat-rcna119856",
    "https://www.nbcnews.com/news/world/90-percent-deaths-afghanistan-quake-women-children-rcna120253",
    "https://www.nbcnews.com/news/world/dozens-killed-injured-powerful-earthquake-aftershocks-afghanistan-rcna119354",
    "https://www.nbcnews.com/news/world/afghan-earthquakes-kill-2445-taliban-say-rcna119436",
    # Hurricane Otis
    "https://www.nbcnews.com/news/us-news/hurricane-otis-grows-dangerous-category-4-storm-heading-mexicos-acapul-rcna122040",
    "https://www.nbcnews.com/science/environment/hurricane-otis-rapid-intensification-climate-change-rcna122090",
    # Noto earthquake
    "https://www.nbcnews.com/news/world/ground-zero-japans-earthquake-zone-ravaged-destruction-fire-rcna132697",
    "https://www.nbcnews.com/news/asia/japan-issues-tsunami-warning-strong-earthquakes-sea-japan-rcna131783",
    "https://www.nbcnews.com/news/world/japan-earthquake-death-toll-rescue-operations-us-aid-wajima-rcna132472",
    "https://www.nbcnews.com/news/world/rescuers-search-survivors-japan-powerful-quakes-rcna132019",
    "https://www.nbcnews.com/news/world/strong-earthquakes-japan-dozens-dead-buildings-destroyed-rcna131826",
    "",
    "",
    "",
    "",
    "",
]
cnn_news_urls = [
    # hawaii wildfire
    "https://www.cnn.com/us/live-news/hawaii-maui-wildfires-08-12-23/index.html",
    "https://www.cnn.com/us/live-news/hawaii-maui-wildfires-08-12-23/index.html?tab=Lahaina",
    "https://www.cnn.com/us/live-news/hawaii-maui-wildfires-08-12-23/h_f896d0ee6750b2e5a1e6a909ba5bfb57",
    "https://www.cnn.com/us/live-news/hawaii-maui-wildfires-08-13-23/index.html",
    "https://www.cnn.com/us/live-news/hawaii-maui-wildfires-08-13-23/h_afc5dc1c1ab56288b1f74d1c4079acd5",
    "https://www.cnn.com/us/live-news/hawaii-maui-wildfires-08-13-23/h_30d0b6a9ea41950229fd0e46d0f7114b",
    "https://www.cnn.com/us/live-news/hawaii-maui-wildfires-08-13-23/h_9a901b86534638d3ea17515b0da205a5",
    "https://www.cnn.com/us/live-news/hawaii-maui-wildfires-08-14-23/index.html",
    "https://www.cnn.com/us/live-news/hawaii-maui-wildfires-08-14-23/h_d6b5a59f175bfe503bceae526c15d2bc",
    "https://www.cnn.com/us/live-news/hawaii-maui-wildfires-08-14-23/h_564fb15196b845bd9896a4eb7780293b",
    "https://www.cnn.com/2023/08/12/us/maui-wildfires-hurricane-dora-saturday/index.html",
    "https://www.cnn.com/2023/08/12/us/hawaii-emergency-warning-system-maui-wildfires/index.html",
    "https://www.cnn.com/2023/08/13/us/maui-wildfires-hurricane-dora-sunday/index.html",
    # hawaii wildfire conspiracy
    "https://www.cnn.com/2023/08/26/tech/maui-wildfire-cause-conspiracy-theory/index.html",
    # Marrakesh–Safi earthquake
    "https://www.cnn.com/middleeast/live-news/morocco-earthquake-09-09-2023/index.html",
    "https://www.cnn.com/middleeast/live-news/morocco-earthquake-09-09-2023/h_8ca3542d718218ace194d55ed4fcfe83",
    "https://www.cnn.com/middleeast/live-news/morocco-earthquake-09-09-2023/h_07082b45c65fe4c79e5ba17debdc52fa",
    "https://www.cnn.com/middleeast/live-news/morocco-earthquake-09-09-2023/h_ea2c57fb8e23076b75782a14a5262b7a",
    "https://www.cnn.com/africa/live-news/morocco-earthquake-marrakech-09-10-23/index.html",
    "https://edition.cnn.com/africa/live-news/morocco-earthquake-marrakech-09-10-23/h_95047bf54145951b4bf9cc30650f49bc",
    "https://edition.cnn.com/africa/live-news/morocco-earthquake-marrakech-09-10-23/h_624f9abafe2235899a62604829700b98",
    "https://www.cnn.com/africa/live-news/morocco-earthquake-marrakech-09-10-23/h_f1c5859783b9ac83ca7e1b3c9d8c3dd1",
    "https://www.cnn.com/africa/live-news/morocco-earthquake-marrakech-09-10-23/h_1e314303195415db078dbaf45cb92784",
    "https://www.cnn.com/africa/live-news/morocco-earthquake-marrakech-09-11-23/index.html",
    "https://www.cnn.com/africa/live-news/morocco-earthquake-marrakech-09-11-23/h_dcd942ebbe83014d468877884a14c7ea",
    "https://www.cnn.com/2023/09/08/africa/morocco-6-8-magnitude-earthquake-intl-hnk/index.html",
    "https://www.cnn.com/2023/09/09/africa/morocco-earthquake-what-we-know-intl/index.html",
    "https://www.cnn.com/2023/09/10/africa/morocco-earthquake-day-two-intl-hnk/index.html",
    "https://www.cnn.com/2023/09/10/africa/mosque-earthquake-damage-marrakech-intl/index.html",
    # Storm_Daniel & Libya flooding
    "https://www.cnn.com/2023/09/11/africa/libya-flooding-storm-daniel-climate-intl/index.html",
    "https://www.cnn.com/2023/09/06/europe/greece-europe-extreme-weather-climate-intl/index.html",
    "https://www.cnn.com/2023/09/28/europe/greece-storm-elias-intl-scn-hnk/index.html",
    "https://www.cnn.com/2023/09/13/africa/libya-flooding-storm-daniel-wednesday-intl-hnk/index.html",
    "https://www.cnn.com/2023/09/19/middleeast/tunisia-saied-storm-daniel-zionism-mime-intl/index.html",
    "https://www.cnn.com/2023/09/14/middleeast/lethal-factors-leading-to-libya-floods-intl/index.html",
    "https://www.cnn.com/2023/09/13/middleeast/libya-flood-political-rift-mime-intl/index.html",
    "https://www.cnn.com/2023/09/19/world/libya-floods-climate-change-impact/index.html",
    "https://www.cnn.com/2023/09/16/africa/derna-destruction-libya-intl/index.html",
    "https://www.cnn.com/2023/09/13/middleeast/what-we-know-about-libya-floods-intl/index.html",
    "https://www.cnn.com/2023/09/05/europe/greece-flooding-climate-intl/index.html",
    "https://www.cnn.com/2023/09/16/world/global-rain-flooding-climate-crisis-intl-hnk/index.html",
    # 2023_Herat_earthquakes
    "https://www.cnn.com/2023/10/07/world/herat-afghanistan-earthquakes-intl/index.html",
    "https://www.cnn.com/2023/10/07/world/afghanistan-herat-earthquake-devastation-intl-hnk/index.html",
    "https://www.cnn.com/2023/10/09/world/herat-afghanistan-earthquake-aid-challenges-intl-hnk/index.html",
    "https://www.cnn.com/2023/10/14/world/afghanistan-earthquake-herat-survivors-dst-intl-hnk/index.html",
    # Hurricane Otis
    "https://www.cnn.com/2023/10/28/weather/hurricane-otis-death-toll-mexico/index.html",
    "https://www.cnn.com/2023/10/26/weather/hurricane-otis-acapulco-mexico-impact-thursday/index.html",
    # Noto earthquake
    "https://www.cnn.com/asia/live-news/japan-ishikawa-earthquake-01-01-24/index.html",
    "https://www.cnn.com/asia/live-news/japan-ishikawa-earthquake-01-01-24/h_bf29cbcfa308be6de38bc572f6f71994",
    "https://www.cnn.com/asia/live-news/japan-ishikawa-earthquake-01-01-24/h_1644c447ea5ea01c74be1e5f231b9e42",
    "https://www.cnn.com/asia/live-news/japan-ishikawa-earthquake-01-01-24/h_87fc516d3b49a284417a74d059715723",
    "https://www.cnn.com/asia/live-news/japan-ishikawa-earthquake-01-01-24/h_6f0ed89315e224662afaf978b428eb10",
    "https://www.cnn.com/asia/live-news/japan-ishikawa-earthquake-01-01-24/h_d76a80612f95aa47b358f8e5d413c455",
    "https://edition.cnn.com/asia/live-news/japan-ishikawa-earthquake-01-01-24/h_2ca0cabc4fb6d65d294827d8bc051265",
    "https://www.cnn.com/asia/live-news/japan-ishikawa-earthquake-01-01-24/h_7bb7c9e1347a0912ce107e0853c2d4d9",
    "https://www.cnn.com/asia/live-news/japan-earthquake-plane-fire-news-01-02-24/index.html",
    "https://www.cnn.com/2024/01/01/asia/japan-earthquake-tsunami-warning-intl-hnk/index.html",
    "https://www.cnn.com/2024/01/02/asia/japan-earthquake-tsunami-warnings-tuesday-intl-hnk/index.html",
    "https://www.cnn.com/2024/01/03/asia/japan-earthquake-shelters-aftermath-intl-hnk/index.html",
    "",
    "",
    "",
    "",
    "",
]

conspiracy = [
    {"title": "X - Post by AJ Huber", "text": """BREAKING: Joe Biden just Confirmed the Directed Energy Weapons have been used to:

Maui, Hawaii "Wildfires"
Panhandle, Texas "Wildfires"

And we knew the camera caught Biden's gigantic teleprompter.

White hats in control in the White House?""", "source": "https://twitter.com/Huberton/status/1763355826821935594?ref_src=twsrc%5Etfw%7Ctwcamp%5Etweetembed%7Ctwterm%5E1763355826821935594%7Ctwgr%5Eade9d559a960823d006e01e00a56a1ae15ed729b%7Ctwcon%5Es1_&ref_url=https%3A%2F%2Fwww.indy100.com%2Fviral%2Fconspiracy-directed-energy-weapons-texas"},
    {"title": "", "text": "Directed Energy Weapons (DEW) was used to set wildfires in Hawaii in 2023.", "source": "hidden"},
    {"title": "", "text": "First it was Hawaii and now it’s Texas. Why do we pretend that the US government isn’t creating these crisis? Research DEW or directed energy weapons.", "source": "hidden"},
{"title": "", "text": "Authorities have decided to point their fingers at one of their favorite culprits: Directed Energy Weapons (DEW).", "source": "hidden"},
{"title": "", "text": "DEW has caused Hawaii wildfire as well as Texas wildfire.", "source": "hidden"},
{"title": "", "text": "Only a Directed Energy Weapon (DEW) can cause this kind of destruction like Hawaii wildfires.", "source": "hidden"},
{"title": "", "text": "We are blaming the wildfires scorching the US state of Hawaii on high-energy lasers fired from the sky.", "source": "hidden"},
{"title": "", "title": "X - Post by Optimus", "text": """Reports suggest that over 80% of ALL of Texas Cattle Feed in the impacted area.

IMO these fires were started by DEWs (Directed Energy Weapons).""", "source": "https://twitter.com/cosmicape888/status/1763769592013627884?ref_src=twsrc%5Etfw%7Ctwcamp%5Etweetembed%7Ctwterm%5E1763769592013627884%7Ctwgr%5Eade9d559a960823d006e01e00a56a1ae15ed729b%7Ctwcon%5Es1_&ref_url=https%3A%2F%2Fwww.indy100.com%2Fviral%2Fconspiracy-directed-energy-weapons-texas"}
]

ap_urls = [
    "https://apnews.com/article/fact-check-conspiracy-blue-items-maui-wildfires-118319149774",
    "https://apnews.com/article/fact-check-maui-hawaii-wildfires-dew-explosion-russia-185319331205",
    "https://www.dw.com/en/fact-check-no-maui-wildfires-were-not-caused-by-directed-energy-weapons/a-66556025",
]

def wiki_data():
    title_subfix = " - Wikipedia"
    data_list = []
    for url in wiki_urls:
        page = requests.get(url)
        soup = BeautifulSoup(page.content, 'html.parser')
        
        title = soup.find('title').get_text().removesuffix(title_subfix).strip()
        paragraphs = soup.find_all("p") 
        text = ""

        for paragraph in paragraphs:
            while paragraph.sup is not None: paragraph.sup.decompose()
            content = paragraph.get_text().strip()
            content = re.sub(r'\[[0-9]+\]', '', content)
            content = re.sub(r'\u00a0', '', content)
            if len(content.split(" ")) < 10: continue
            text += f"{content}\n"
        text = text.strip()
        data_dict = {
            "title": title,
            "text": text,
            "source": url
        }
        data_list.append(data_dict)
    return data_list

def ap_data():
    data_list = []
    for url in ap_urls:
        if url=="": continue
        page = requests.get(url)
        soup = BeautifulSoup(page.content, 'html.parser')
        title = soup.find('title').get_text().strip()
        
        paragraphs = soup.find_all("p") 
        text = ""
        for paragraph in paragraphs:
            while paragraph.sup is not None: paragraph.sup.decompose()
            content = paragraph.get_text().strip()
            content = re.sub(r'\xa0', ' ', content)
            if len(content.split(" ")) < 10: continue
            text += f"{content}\n"
        text = text.strip()
        data_dict = {
            "title": title,
            "text": text,
            "source": url  
        }
        data_list.append(data_dict)
    return data_list

def cnn_data():
    cnn_news_tostrip = [
        "We've moved our coverage of the Hawaii wildfires here.",
        "Images from the scene have revealed the far reaching damage wrought by the wildfires on Maui. We have a full gallery but here is a selection below.",
        "Please enable JavaScript for a better experience.",
        "© 2024 Cable News Network. A Warner Bros. Discovery Company. All Rights Reserved.  CNN Sans ™ & © 2016 Cable News Network.",
    ]
    for url in cnn_news_urls:
        if url=="": continue
        page = requests.get(url)
        soup = BeautifulSoup(page.content, 'html.parser')
        
        title = soup.find('title').get_text().strip()

        data_list = []
        paragraphs = soup.find_all("p") 
        text = ""
        for paragraph in paragraphs:
            while paragraph.sup is not None: paragraph.sup.decompose()
            content = paragraph.get_text().strip()
            content = re.sub(r'\xa0', ' ', content)
            if content in cnn_news_tostrip: continue
            if len(content) == 0: text += "\n"
            if len(content.split(" ")) < 5: continue
            text += f"{content}\n"
        text = text.strip()
        data_dict = {
            "title": title,
            "text": text,
            "source": url  
        }
        data_list.append(data_dict)
    return data_list

def nbc_data():
    data_list = []
    for url in nbc_news_urls:
        if url=="": continue
        page = requests.get(url)
        soup = BeautifulSoup(page.content, 'html.parser')
        title = soup.find('title').get_text().strip()
        
        paragraphs = soup.find_all("p") 
        text = ""
        for paragraph in paragraphs[1:]:
            while paragraph.sup is not None: paragraph.sup.decompose()
            content = paragraph.get_text().strip()
            content = re.sub(r'\xa0', ' ', content)
            if len(content.split(" ")) < 10: continue
            text += f"{content}\n"
        text = text.strip()
        data_dict = {
            "title": title,
            "text": text,
            "source": url  
        }
        data_list.append(data_dict)
    return data_list


data_list = ap_data() + wiki_data() + cnn_data() + nbc_data() + conspiracy
save_dir = "./../../data/disaster_compressed.json"
json.dump(data_list, open(save_dir, "w"), indent=4)