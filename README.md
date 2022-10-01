## Hackerearth Machine Learning Challenge: Adopt a Buddy

[LinkedIn Post](https://www.linkedin.com/posts/linksumitsaha_hackerearth-machinelearning-weekendlearning-activity-6703286416455057408-Jggy/)

Rank: 31 (Top 1%)

Participants: 5000+

![image](https://user-images.githubusercontent.com/26060275/193403505-ca2710e4-c4fa-4aa7-b5bc-bc778d8ad753.png)


### Pre-Processing:
1. Removed categorical samples from train which did not belong in test for "Color Type" and "X1"
2. Since all NULL values in "Condition" belonged to Class=2.0 of "Breed", it was filled with a unique value -1
3. "Condition" and "Color Type" were encoded into One-Hot vectors 
4. Converted "Issue Date" and "Listing Date" into their date time formats
5. Dropped "height" and "length" column as they didn't convey any information - found from plotting and correlation

### Feature Engineering:
1. Difference of Listing Date and Issue Date in Months and Years
2. Extracted Year & Month from Listing Date and Issue Date
3. Extracted quarter of year from the two date time columns
4. One-Hot encoded the "Listing Year"

### Models Used/ Experimented with:
1. LightGBM
2. XGBoost
3. CatBoost

### Methodology:
1. CatBoostClassifier gave the best score of 91.01310
2. Categorical Variables were explicitly mentioned to the CatBoostClassifier
3. Hyperopt was used to tune the Hyperparameter over a pre-defined space
4. Model Stacking was done with LGBM, XGB, and CB and they gave lesser accurate results
5. Pet Category prediction was used as a predictor for Breed Classification

### Tools:
Python, Seaborn, Pandas, NumPy, Scikit-Learn, XGBoost, CatBoost, LightGBM, HyperOpt, Pickle
