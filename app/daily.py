from sqlalchemy import desc
import tweepy
import os
os.environ["TZ"] = "UTC"
import tweepy
import requests
import datetime
import json
import pandas as pd
pd.options.mode.chained_assignment = None
from datetime import timedelta
from datetime import datetime
import numpy as np
from datetime import datetime
from dateutil.parser import parse

from dotenv import load_dotenv

load_dotenv()

from funcs import load_my_tweets

# auth to twitter
consumer_key = os.getenv("TWITTER_API_KEY")
consumer_secret = os.getenv("TWITTER_API_SECRET")
access_token = os.getenv("TWITTER_ACCESS_TOKEN")
access_token_secret = os.getenv("TWITTER_ACCESS_SECRET")

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)

print('#########################################################')
if api.verify_credentials() == False:
    print("The user credentials are invalid.")
else:
    print("Twitter authentication successful")

# check whether to tweet - get all of my tweets into a dataframe
tweets_list = tweepy.Cursor(api.user_timeline,
                user_id='Goodies4Ukraine', 
                tweet_mode='extended').items()
output = []

for i in tweets_list:
    text = i._json["full_text"]
    created_at = i._json["created_at"]
    id = i._json["id"]
    line = {'text' : text, 'created_at' : created_at, 'id' : id}
    output.append(line)

tweetz = pd.DataFrame(output)
arrival_tweets = tweetz[tweetz['text'].str.contains('goodies for the ukrainians', case=False)]

# how many tweets with deliveries from the last day
# filter to only those from the last day
#today = (datetime.today() - timedelta(days=1)).strftime('%Y-%m-%d')
today = datetime.today().strftime('%Y-%m-%d')
print(today)
arrival_tweets['date'] = pd.to_datetime(arrival_tweets['created_at'])
# arrival_tweets.index = pd.DatetimeIndex(arrival_tweets.pop('date'))
# print(arrival_tweets.sort_index().last('24H'))


#flight_db = flight_db[flight_db['date'] == today]
#arrival_tweets['date'] = pd.to_datetime(arrival_tweets['created_at'])
#arrival_tweets['dt24'] = pd.to_datetime(arrival_tweets['created_at']) - datetime.timedelta(hours=20)
#arrival_tweets = arrival_tweets.sort_values(['date'], ascending=False)

arrival_tweets = arrival_tweets[arrival_tweets['date'] >= today]
print(arrival_tweets)
#print(today)

arrival_tweets['source'] = ''
for src in [

    'US Air Force',
    'Canadian Armed Forces',
    'Australian Air Force',
    'British Royal Air Force',
    'Royal Navy',
    'Estonian Air Force',
    'Finnish Air Force',
    'Croatian Air Force',
    'German Air Force',
    'US Navy',
    'NATO',
    'Turkish Air Force',
    'NATO',
    'US Army',
    'Italian Air Force',
    'Antonov Airlines',
    'Norwegian Air Force',
    'Romanian Air Force',
    'French Air Force',
    'German Air Force',
    'Swedish Air Force',
    'Netherlands Air Force',
    'Belgian Air Force',
    'Danish Air Force']:

    arrival_tweets['source'] = np.where(arrival_tweets['text'].str.contains(src), src, arrival_tweets['source'])

## get count by group
agg = pd.DataFrame(arrival_tweets.groupby(['source'])['text'].count()).sort_values(['text'], ascending=False)
agg.columns = ['count']

list = []

for i in range(0, len(agg)):
    src = agg.index[i]
    cnt = agg['count'].iloc[i]
    txt = f'{src}: {cnt}'
    list.append(txt)

tweet_prt = '\n'.join(list)

#print(arrival_tweets.head())
n_deliv = len(arrival_tweets)
if n_deliv == 1:
    fl = 'flight'
else:
    fl = 'flights'

tweet = f"In the last day I tracked {n_deliv} {fl} by military affiliated aircraft into Rzeszow, Poland: \n\n{tweet_prt}"

print(tweet)
api.update_status(tweet)

# get number for each group



