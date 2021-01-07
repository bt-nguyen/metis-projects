'''
This python function scrolls through the url (Steam website)
and generates a list of websites per game to scrape
detailed data for project 2 titled:
'steamGameList.txt'
'''

# Imports
from bs4 import BeautifulSoup
import requests
import time
import os
from selenium import webdriver
from selenium.webdriver.common.keys import Keys

# path to the chromedriver executable
chromedriver = "/Applications/chromedriver"
os.environ["webdriver.chrome.driver"] = chromedriver

# Designate a URL to pull data from
url = 'https://store.steampowered.com/search/?category1=998'

# Use the chromedriver to access the website
driver = webdriver.Chrome(chromedriver)
driver.get(url)

# Change range limit for differemt amount of entries (40000 entries)
for i in range(800):
    # Scroll
    driver.execute_script(
        # Alternatively, document.body.scrollHeight
        "window.scrollTo(0, document.documentElement.scrollHeight);"
    )

    # Wait for page to load
    time.sleep(1)

soup = BeautifulSoup(driver.page_source, 'lxml')

# Make a list of sites to go through
list_of_links = []

# Retrieve a list of links
for link in soup.find('div', id='search_resultsRows').find_all('a', href=True):
    list_of_links.append(link.get('href'))

# Save list_of_links as a txt file 'steamGameList.txt'
with open('steamGameList.txt', 'w') as f:
    for item in list_of_links:
        f.write("%s\n" % item)
