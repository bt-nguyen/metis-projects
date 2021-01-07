# Identifying Depression in Adults

The goal of this project is to identify self-assessed feelings of depression in adults using the National Health and Nutrition Examination Survey (NHANES) as the data source. The prominent features identified stem from the PHQ-9 survey (*e.g.* issues surrounding appetite, energy levels, and speech pace ) and demographic information.



## Objective

The objective of this project is to explore different classification models to identify depression in adults (age 20 and higher). Additionally, there is an explorative component for identifying appropriate features to identify depression. Gaussian Naives Bayes and XGBoost classifier were found to be the best model for this objective; the latter being used for: obtaining feature importance, tunability for higher recall score (relative to other models), and personal interest because of the traction it's gaining in data science.

### Data

The NHANES data set is used as our source, compiled from a ten year span (2009-2018). The target variable  is a self-assessment of feeling down, depressed, or hopeless for nearly all days within the past two weeks (DPQ020, score = 3). Approximately 30 features were initially examined, with 16 features being used for the final model. These features are primarily from the demographic and questionnaire sections of the data set.

### Tools Used

**Jupyter Notebook** was used as the interface to deploy Python code. **Pandas** was used for generating, cleaning, and exploration of the dataframe. **Matplotlib** and **Seaborn** were used for plotting. **Numpy** was used for computation. **Scikit-learn** and **Xgboost** were used for modeling. **Flask** and **Junicorn** were used to deploy an app on Heroku.

### Author

Brian Nguyen (https://github.com/bt-nguyen/)