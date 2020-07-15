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

## K-Means Approach
The K-Means algorithm is a pretty simple algorithm capable of clustering this kind of dataset very quickly-although perhaps not very well. There's a whole of different varieties of wines in the world and in this dataset, so I filtered out wines based on whether they had more than 4000 reviews or not, to make sure I had enough descriptions to work with. The majority of wine reviews seem to be on Chardonnays, Pinot Noirs, and Cabernet-Sauvignons, with Red Blends also making a respectable showing. 

Next, I used the TfidfVectorizer from sklearn to get the important words in the dataset. Tfidf is short for term frequency-inverse document frequency. It's a numerical statistic that reflects how important a word is to a document. A word's tfidf value increases proportionally to the number of times a word appears in a description and is offset by the number of descriptions in the corpus that contain the word. I also filtered out things like punctuation marks, numbers, and words like 'flavor' which appear very frequently but don't help me narrow down what type of wine I'm looking at. I also used a stemmer so a word and its plural/possessive are treated the same.

My first shot I decided to use 15 clusters of words to try and group the dataset, just to see what would happen. The word clusters came out ok I guess, although one small thing is that some clusters contained the name of the wine in it. I created a heat map of which wine varieties mappen to which clusters, and obviously Chardonnay mapped to the cluster with 'chardonnay' in it. I honestly don't know if that's letting the computer cheat lol. I might filter out the names from the descriptions later and see how that changes things.

Next step was to analytically see if there was an optimal number of clusters. I frist tried the elbow-methid. This step took a while using straight up K-Means, so I used Mini-batch K-Means to speed up the process. I figured that since the centroids are probably going to be so small anyways (as confirmed by the silhouette scores later), mini-batch wouldn't significantly hurt my results. And the inertia from either K-Means or Mini-batch K-Means was super high regardless so again, doesn't matter which variant of K-Means. The elbow method didn't reveal that much. It was a pretty convex curve down, with no obvious elbow. So I tried a silhouette score next. That got me some K values that appeared optimal, and I ended up going with 21 clusters.

## DBSCAN Approach
Do not try it.

Ok well if I used a smaller set to cluster it might work something out eventually. Looking at the graphs of inertia vs k from the K-Means approach, the inertias are abhorrently large. This translates to the mean squared distance between each instance and its closest centroid being very very large, aka it will be very difficult for DBSCAN to define clusters of high density. I tried initializing the DBSCAN with a very big epsilon (it sort of controls the size of the cluster), but the problem with that is Scikit-Learn's implementation of DBSCAN approaches a memory requirement of O(m^2) when epsilon is large, and my computer was extremely unhappy with that.

## BIRCH Approach
So this one I just tried because it's hypothetically faster than batch K-Means while giving similar results and is designed for large datasets. However the more features there are, the less effective it becomes, and there are a lot of features. BIRCH works by building a tree structure that contains just enough information to quickly assign each new instance to a cluster, and by setting the threshold value as high as possible (which limits the amount of branching), I was able to get BIRCH to identify 66 clusters. Not ideal, but it's better than nothing. Although when constructing a heat map it seems to like mapping everything to one cluster so I have to dig in to that. 
