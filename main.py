import re
import string
import numpy as np
import pandas as pd
import tweepy

# plotting
import matplotlib.pyplot as plt

#textblob
from textblob import TextBlob

# nltk
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk import tokenize

# sklearn
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

consumerKey = "OK0omPEmhQdKSdUtLH2CB3ICw"
consumerSecret = "dwVf0g9jLMOWMVy7mz4KDeW8WY4pcGz22MlSEPQYlAmKZTwx4M"
accessToken = "1546899945843675137-y0Lg4QhsI5pXrvrgBq6kQZiHSq67fz"
accessTokenSecret = "6rYg5arka9ewqEAzW0pO0F9ggqbbWdA7vXIHhR3atjFhP"

auth = tweepy.OAuthHandler(consumerKey, consumerSecret)
auth.set_access_token(accessToken, accessTokenSecret)
api = tweepy.API(auth)

#Pulls tweets from the @POTUS account, looking only at the extended tweet_mode for the text
tweets = tweepy.Cursor(api.user_timeline, screen_name="@POTUS", tweet_mode="extended").items(1000)

#Creates Pandas dataframe to easily manipulate tweets
tweets_compiled = [tweet.full_text for tweet in tweets]
df = pd.DataFrame(tweets_compiled, columns=["Tweets"])

#Cleaning Tweets by removing retweet symbols, punctuation, websites, and other unneeded characters
df['Tweets'] = list(map(lambda x: re.sub("(@[A-Za-z0–9]+)|(http\S+)|(#)","",x),list(df['Tweets'])))
df['Tweets'] = list(map(lambda x: re.sub('RT : ',"",str(x)),list(df['Tweets'])))
df['Tweets'] = df['Tweets'].str.lower()
df['Tweets'] = df['Tweets'].apply(lambda x: x.translate(str.maketrans('','', string.punctuation)))
df['Tweets'] = df['Tweets'].apply(lambda x: re.sub('\d+', '', x))
df['Tweets'] = df['Tweets'].apply(lambda x: re.sub("[^\w\d\s]+", '', x))
df['Tweets'] = df['Tweets'].apply(lambda x: re.sub('www.[^s]+', '', x))

#Removes all english stopwords in nltk library
stopwords = stopwords.words("english")
df['Tweets'] = df['Tweets'].apply(lambda x: " ".join([word for word in str(x).split() if word not in stopwords]))

#Applies Sentiment Polarity function to label tweets as Positive, Negative, or Neutral
sentiment_polarity = df['Tweets'].apply(lambda x: TextBlob(x).sentiment.polarity)
df['Sentiment'] = sentiment_polarity

#Converts numerical value of sentiment polarity into a label
df['Label'] = [None]*len(df)
for index, row in df.iterrows():
    if df.loc[index,'Sentiment'] < 0:
        df.loc[index,'Label'] = "Negative"
    elif df.loc[index,'Sentiment'] > 0:
        df.loc[index,'Label'] = "Positive"
    else:
        df.loc[index,'Label'] = "Neutral"

#Tokenizes, Stems, and Lemmatizes each word in tweets before combining to prepare for Machine Learning
df['Initial'] = df['Tweets'].apply(RegexpTokenizer('\w+').tokenize)
df['Initial'] = df['Initial'].apply(lambda x: [nltk.PorterStemmer().stem(i) for i in x])
df['Initial'] = df['Initial'].apply(lambda x: [nltk.WordNetLemmatizer().lemmatize(i) for i in x])
df['Final'] = df['Initial'].apply(lambda x: " ".join(i for i in x))

#Takes the tweets and splits them into training and testing data sets
X_train, X_test, y_train, y_test = train_test_split(np.array(df['Final']),np.array(df['Label']),test_size = 0.4)

#Uses sklearn's TF-IDF conversion to assign words a value of importance of a word to the document
vectoriser = TfidfVectorizer(ngram_range=(1,2))
transform = vectoriser.fit_transform(X_train)
X_train = vectoriser.transform(X_train)
X_test = vectoriser.transform(X_test)

#Fits the data using a Linear Support Vector Classifier
fit = LinearSVC().fit(X_train, y_train)
y_pred = fit.predict(X_test)
confusionMatrix = confusion_matrix(y_test, y_pred, labels=fit.classes_)
display = ConfusionMatrixDisplay(confusion_matrix=confusionMatrix, display_labels=fit.classes_)

#Displays plot of the data and accuracy + f1 score
display.plot()
plt.show()
print('LinearSVC Accuracy Score: ' + str(accuracy_score(y_test, y_pred)))
print('LinearSVC f1 Score: ' + str(f1_score(y_test, y_pred, average=None, zero_division=0)))

