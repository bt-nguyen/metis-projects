# Features Leading to a Well-Received Video Game

This project investigates what features lead to a well-received video game using store.steampowered.com as a source. This project was performed in the scope of the Metis data science program.



## Objective

The project's objective is to take an exploratory approac to examine which features lead to a well-received video game. Six features are examined initially, but the scope is narrowed to four features after exploratory data analysis. Linear regression was used to predict how these features effect user ratings, but a poor fit was observed. Regardless, we find that minimum storage required (a possible indicator of AAA games versus indie games) and the quantity of labeled genres are the most impactful features from this investigation.

### Data

The source for our data is store.steampowered.com. Linked were collected using BeautifulSoup and Selenium to collect specific product links and scrape features/data from those links. Collected features are: release date, number of reviews since release, overall rating since release, number of reviews within past 30 days, rating within past 30 days, labeled genres per game, minimum storage required for installation, and price.

### Tools Used

**Jupyter Notebook** was used as the interface to deploy Python code. **Pandas** was used for generating, cleaning, and exploration of the dataframe. **Matplotlib** and **Seaborn** was used for plotting. **Numpy** was used for computation. **BeautifulSoup** and **Selenium** was used for scraping. **Statsmodel** and **sklearn** was used regression.

### Author

Brian Nguyen (https://github.com/bt-nguyen/)

