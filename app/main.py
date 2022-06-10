import os
os.environ["TZ"] = "UTC"
import tweepy
import requests
import json
import pandas as pd
pd.options.mode.chained_assignment = None
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import geopy.distance
from dotenv import load_dotenv
import datetime
from datetime import timedelta
from datetime import datetime, timezone

load_dotenv()

###############################
# EXECUTE
###############################

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
print('\n')
    
##################################
# LOAD FUNCTIONS
##################################

from funcs import pull_flight_data
from funcs import get_mil_flights
from funcs import get_deliveries
#from funcs import google_maps_tweet
#from funcs import google_maps_tweet2
from funcs import foreign_flights_in_rzeszow
from funcs import load_my_tweets
from flight_db2 import flight_loc_db
from dev.plot_and_tweet import plot_and_tweet
from dev.find_origin import find_origin
from dev.plot_and_tweet import plot_layers
#from dev.plot_layers import plot_layers

# update military flight loc database
#flight_loc_db()

# get dataframes of past tweets
all_tweets, interest_tweets, arrival_tweets = load_my_tweets(api)

# pull flight data
flight_df = pull_flight_data(api)

# # find potential flights and their origins
potential_flights = get_mil_flights(flight_df)

potential_origins = find_origin(potential_flights, interest_tweets)
#print(potential_origins.head())
print('\n')

#print(potential_origins.head())

#pd.read_csv(pathhhhh)

# # find flights in Rzeszow
if len(potential_flights) >= 1:
    in_rzeszow = get_deliveries(potential_flights)

    in_rzeszow_origins = find_origin(in_rzeszow, arrival_tweets, rz=True)

    print('\n')

    # tweet flights
    n_pot = len(potential_flights)
    print(f'{n_pot} potential_flights to plot')
    if len(potential_flights) >= 1:
        plot_and_tweet(api, potential_flights, potential_origins, all_tweets, rz = False)

    n_rz = len(in_rzeszow)
    print(f'{n_rz} in_rzeszow flights to plot')
    if n_rz >= 1:
        plot_and_tweet(api, in_rzeszow, in_rzeszow_origins, all_tweets, rz = True)

# google_maps_tweet2(api, potential_flights, override = False, tweets_df = all_tweets)
# google_maps_tweet2(api, in_rzeszow, rz = True, tweets_df = all_tweets)
    # Kalitta air

## run RCH
print(" ########## Airlabs submodule ############")
import rch
from rch import rch_crnt_df_pot, rch_crnt_df_rz, rch_dep_df_pot, rch_dep_df_rz #, rcrz_crnt_df, rcrz_dep_df
if len(rch_crnt_df_pot) >= 1:
    plot_and_tweet(api, rch_crnt_df_pot, rch_dep_df_pot, all_tweets, rz=False, rch=True)
else:
    print('no incoming RCH flights to plot')

if len(rch_crnt_df_rz) >= 1:
    plot_and_tweet(api, rch_crnt_df_rz, rch_dep_df_rz, all_tweets, rz=True, rch=True)
else:
    print('no landing RCH flights to plot')

print('')
print('###########################################')

# update flight loc db by adding new ones and dropping low ones
flight_loc_db()
