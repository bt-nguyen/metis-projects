'''
This python function scrapes detailed information
from the txt file: 'steamGameList.txt', which
is a list of all/most games from Steam.
The final file is a csv titled:
'steam_dataframe.csv'
'''

# Imports
from bs4 import BeautifulSoup
import requests
import pandas as pd

# Initiate a dataframe
df = pd.DataFrame()

with open("steamGameList.txt", "r") as file:
    for line in file:
        try:
            url = line.split("\n")[0]

            response = requests.get(url)
            soup = BeautifulSoup(response.content, 'lxml')

            # Title
            title = soup.find('div', class_='apphub_AppName').text

            # Release Date
            release_date = soup.find('div', class_='date').text

            # Price: will collect discounted prices and other text; must be cleaned in dataframe
            price = soup.find(
                'div', class_='game_purchase_action_bg').text.strip()

            # Storage: this will pull storage from Windows (min, then recommend), Mac, and Linux (if compatible)
            storage = []

            for requirement in soup.find('div', class_ = 'sysreq_contents').find_all('li'):
                if 'Storage' in requirement.getText():
                    storage.append(requirement.getText())

            # Make empty genre to append multiple genres
            genre = []
            developer = []
            publisher = []

            # Checks for specific keywords in line; then makes a list (in case multiple genres/devs/pubs)
            for detail in soup.find('div', class_='details_block').find_all('a'):
                if 'genre' in detail.get('href'):
                    genre.append(detail.getText())
                elif 'developer' in detail.get('href'):
                    developer.append(detail.getText())
                elif 'publisher' in detail.get('href'):
                    publisher.append(detail.getText())

            # Create a PRIMARY genre that's listed at the TOP LEFT of the Game Page for reference (may be needed later)
            primary_genre = soup.find(
                'div', class_='breadcrumbs').find_all('a')[1].text

            # Collects a string for the ratings (recent and all); to be cleaned in dataframe
            for rating in soup.find('div',
                                    class_='user_reviews').find_all('span', class_='nonresponsive_hidden responsive_reviewdesc'):
                if 'reviews in the last' in rating.text:
                    recent_reviews = rating.getText().strip()
                else:
                    all_reviews = rating.getText().strip()

            # Collect Metacritic score; some games do not have a score and will print 'No Score'
            try:
                metacritic_score = soup.find(
                    'div', class_='score high').text.strip()
            except:
                metacritic_score = "No Score"

            game_dictionary = {'Title': title, 'Release_Date': release_date, 'Price': price,
                               'Genre': genre, 'Developer': developer, 'Publisher': publisher,
                               'Primary_Genre': primary_genre, 'Recent_Reviews': recent_reviews,
                               'All_Reviews': all_reviews, 'Metacritic': metacritic_score, 'Storage': storage[0]}

            df = df.append(game_dictionary, ignore_index=True)

            if len(df) % 2500 == 0:
                df.to_csv('steam_dataframe.csv')

            if len(df) > 40000:
                break

        except:
            continue

df.to_csv('steam_dataframe.csv')
