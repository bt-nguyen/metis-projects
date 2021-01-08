# Satirical Headines: Recognizing Truth/Lies

This project used satirical headlines from two subreddits: /r/TheOnion and /r/NotTheOnion, to investigate different topics between both subreddits and attempt to classify the headline's correpsponding subreddit.  A total of 6 distinguishable topics were found, being topics centered around: toilet paper, black lives matter, current affairs, police, New York, and politics (in order of headline frequency). Classification resulted in accuracy of a maximum of 57%.



## Objective

The goal of this project is to explore different methods to determine true headlines from fake headlines with both being written in a satirical manner. The explorative approach included topic modeling (LSA, LDA, and NMF) and clustering algorithms (e.g. K-means). After topics were determined, the results were inputted into a classifier, along with VADER sentiment results as features, to investigate if real events could be distinguished from fake events.

### Data

The data used was obtained from two subreddits: /r/TheOnion and /r/NotTheOnion. Data was scraped from January 1, 2020 to August 15, 2020. 1,081 headlines were scraped from /r/TheOnion and 36,262 headlines were from /r/NotTheOnion.  Only the headline titles were used in this analysis.

### Tools Used

**Jupyter Notebook** was used as the interface to deploy Python code. **Pandas** was used for generating, cleaning, and exploration of the dataframe. **Matplotlib** and **Seaborn** were used for plotting. **Numpy** was used for computation. **Scikit-learn** was used for topic modeling and clustering. **Xgboost** was used for classification. **t-SNE** was used for dimensionality reduction.

### Author

Brian Nguyen (https://github.com/bt-nguyen/)

