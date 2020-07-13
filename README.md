# WineReviews
ML on Kaggle dataset

## About
The actual datasets can be found here: https://www.kaggle.com/zynicide/wine-reviews. There are around 280K reviews scraped from Wine Enthusiast. The overall goal is to predict a
wine's type based on features like it's description, point rating, region, etc. So even without being able to taste the wine, the model could hypothetically determine the type of 
wine nonetheless.

## Exploratory Analysis 
The first thing I did was to see if there was any obvious correlations we in the data that I could make use of. The only two numerical attributes in the dataset are "points" and
"price". The correlation is exceptionally weak, around 0.46. So that's not too promising. Next I tried finding correlation between "points", "price", "province", "region_1", and
"variety" just for fun. The latter three categories aren't numerical attributes, so I used sklearn's ordinal encoder to convert them into numbers. 
Then, using a scatter matrix, it's pretty obvious again that there's nothing much there either. 

tldr: Not much correlation in the dataset currently. 
