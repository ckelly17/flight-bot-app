# The goal of this standalone script is to access US Air Force planes under
# the call sign RCH that are not found in the Open Skies API that powers
# most of the app. A fair number of RCH planes aren't found there, however,
# and I want to capture them.

####################################
# Running every 10 minutes in heroku
####################################

from asyncore import write
import os
os.environ["TZ"] = "UTC"
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None
import requests
import json
import geopy.distance
import plotly.express as px
from datetime import datetime, timedelta
from sqlalchemy import create_engine
import plotly.graph_objects as go
import tweepy
from dotenv import load_dotenv
from funcs import load_my_tweets

load_dotenv()

# get dataframes of past tweets
# auth to twitter
consumer_key = os.getenv("TWITTER_API_KEY")
consumer_secret = os.getenv("TWITTER_API_SECRET")
access_token = os.getenv("TWITTER_ACCESS_TOKEN")
access_token_secret = os.getenv("TWITTER_ACCESS_SECRET")

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)
all_tweets, interest_tweets, arrival_tweets = load_my_tweets(api)

#####################################
# Initiate connection to database
#####################################

raw_db_url = 'postgres://wzypmlovttdnzl:bb530351daefbc1560ae45f0d052533e2dfc1b4f430c5aa6a71ee2d7f94049aa@ec2-54-164-40-66.compute-1.amazonaws.com:5432/d7g9sn5bbegjbs'
final_db_url = "postgresql+psycopg2://" + raw_db_url.lstrip("postgres://")
engine = create_engine(final_db_url)

# get current dates and times
now = datetime.now()
today = datetime.today().strftime('%Y-%m-%d')
now10 = datetime.now() + timedelta(minutes=10)

##################################
# BEGIN
##################################

## read current rch data from table
rch_db = pd.read_sql_table("rch_flights",  con=engine)
n_before = len(rch_db)

## get df of starting lats and longs
starting_pos_df = rch_db[['callsign', 'start_lat', 'start_long']]

# if update_dt is greater than current time, do nothing
if len(rch_db) == 0:
    update_dt = now - timedelta(minutes=1)
else:
    update_dt = rch_db['next_update_dt'].iloc[0]

#print(f'Will not update before {update_dt}')
#print(f'Current time: {now}')
time_left = (update_dt - now)
time_left = str(timedelta(seconds=time_left.seconds))#

## drop records from before today
rch_db = rch_db[rch_db['date'] == today]

# get unique callsigns
db_callsigns = rch_db['callsign'].unique()

###################################
# START
###################################
# t = True
# if t == False:
if update_dt >= now:
    print(f'Not my time to run yet. Check back in {time_left}')

#elif t == True:
elif update_dt < now:
    print(f'Currently tracking {n_before} flights through airlabs')

    ###### load data from airlabs
    try:
        apikey = '2e6da084-608a-444e-9bb2-fdff525ff8c9' # main api (ckelly52)
        url_data = f'https://airlabs.co/api/v9/flights?api_key={apikey}'
        response = requests.get(url_data).json()
        airlabs_all_flights = pd.DataFrame(response['response'])
    except:
        apikey = '274a066d-69d3-4990-b4f7-c563e3f2f1f5' # deloitte api
        url_data = f'https://airlabs.co/api/v9/flights?api_key={apikey}'
        response = requests.get(url_data).json()
        airlabs_all_flights = pd.DataFrame(response['response'])

    # get just RCH and others of interest
    airlabs_all_flights = airlabs_all_flights[~airlabs_all_flights['flight_icao'].isna()]
    RCH = airlabs_all_flights.copy()
    RCH['keep'] = False
    calls = ['RCH', 'RRR', 'IAM', 'HKY', 'CFC', 'ASY', 'HERKY', 'CTM', 'KING']
    for c in calls:
        RCH['keep'] = np.where(RCH['flight_icao'].str.startswith(c), True, RCH['keep'])
    
    RCH = RCH[RCH['keep'] == True]
    #RCH = airlabs_all_flights[airlabs_all_flights['flight_icao'].str.startswith('RCH')]
    n_active_flights = len(RCH)
    active_rch = RCH['flight_icao'].unique()
    print(f'API call to airlabs incoming: {n_active_flights} active flights ({active_rch})')
    api_df = pd.DataFrame({'type': {0: 'RCH'}, 'time':{0:now},'date':{0:today}, 
                        'callsign':{0:'All RCH'}, 'key':{0:apikey}})
    api_df.to_sql("api_calls", con=engine, if_exists='append', index=False)
   

    # RCHE2 = RCH[RCH['flight_icao'].str.startswith('RCHE2')]
    # print(RCHE2)

    # subset cols
    desired_cols = ['flight_icao','dep_icao','aircraft_icao','lat','lng','alt']
    cols = [c for c in desired_cols if c in RCH.columns]
    missing_cols = [c for c in desired_cols if c not in RCH.columns]
    RCH = RCH[cols]
    RCH['alt'] = pd.to_numeric(RCH['alt'], errors = "coerce")
    RCH['alt'] =  RCH['alt'] * 3.25 # convert from meters to feet
    RCH = RCH.rename(columns = {'aircraft_icao':'aircraft',
                                'lng':'long',
                                'flight_icao':'callsign'})

    # add blanks for missing data
    for c in missing_cols:
        RCH[c] = None

    # add date
    RCH['date'] = today
    RCH['insert_date'] = now
    RCH['drop_date'] = now + timedelta(hours=24)

    new_callsigns = RCH['callsign'].unique()

    # replace if already in database for today
    for callsign in new_callsigns:
        if callsign in db_callsigns:
            rch_db = rch_db[rch_db['callsign'] != callsign]

    # get current distance from rzeszow
    rzeszow_coord = (50.105999576, 22.017999928)
    RCH['dist_from_rzeszow'] = None

    for i in range(0, len(RCH)):
        test = RCH[:i+1]
        lat = test['lat'].iloc[i]
        long = test['long'].iloc[i]
        coord = (lat, long)
        dist = geopy.distance.geodesic(coord, rzeszow_coord).mi
        RCH['dist_from_rzeszow'].iloc[i] = dist

    # new callsigns
    newly_added = [c for c in RCH['callsign'].unique() if c not in db_callsigns]
    print(f'{newly_added} added')

    # set previous as last and drop current
    RCH['prev_dist_from_rzeszow'] = RCH['dist_from_rzeszow']

    rch_db['prev_dist_from_rzeszow'] = rch_db['dist_from_rzeszow']
    rch_db['dist_from_rzeszow'] = None

    # append result
    write_df = pd.concat([rch_db, RCH], ignore_index=True)

    # add airport infor
    for c in ['dep_airport_lat', 'dep_airport_long', 'country_code', 'iata_code','name']:
        if c in write_df.columns:
            write_df = write_df.drop(columns=[c])
    dep_airports = pd.read_csv('data/airports.csv')
    dep_airports['icao_code'] = dep_airports['icao_code'].str.strip()
    dep_airports = dep_airports[~dep_airports['icao_code'].isna()]
    dep_airports = dep_airports[dep_airports['icao_code'] != '']

    dep_airports = dep_airports.rename(columns = {'lat':'dep_airport_lat',
                                                  'lng': 'dep_airport_long',
                                                  'icao_code': 'dep_icao'})
    dep_airports = dep_airports[['dep_icao', 'dep_airport_lat', 'dep_airport_long', 'name','country_code']]
    
    write_df = write_df.merge(dep_airports, how = 'left', on = ['dep_icao'])
    write_df['known_dep'] = np.where(write_df['dep_icao'].isna(), 'M', 'Y')
    ## drop extra columns
    drop_cols = [c for c in write_df.columns if c.endswith('_x') or c.endswith('y')]
    write_df = write_df.drop(columns = drop_cols)

    # drop start pos and merge in
    write_df = write_df.drop(columns = {'start_lat', 'start_long'})
    write_df = write_df.merge(starting_pos_df, how = 'left', on = ['callsign'])
    write_df['start_lat'] = np.where(write_df['start_lat'].isna(), write_df['lat'], write_df['start_lat'])
    write_df['start_long'] = np.where(write_df['start_long'].isna(), write_df['long'], write_df['start_long'])
    

    #write_df['altitude'] = pd.to_numeric(write_df['alt'], errors = 'coerce') * 3.25

    ### only look at planes not tracked elsewhere
    flight_db = pd.read_sql_table("flight_db",  con=engine)
    main_calls = flight_db['callsign'].unique()
    rch_calls = write_df['callsign'].unique()
    extra_calls = [c for c in rch_calls if c not in main_calls]

    if len(extra_calls) >= 1:
    # what is the closest distance?
        close_df = write_df[write_df['callsign'].isin(extra_calls)]
        #close_df['dist_from_rzeszow'] = round(close_df['dist_from_rzeszow'])
        print(close_df[['callsign', 'dist_from_rzeszow', 'dep_icao']].sort_values(['dist_from_rzeszow'], ascending=True, axis=0).head())
        print(len(close_df))
        closest_flight_to_rz = close_df['dist_from_rzeszow'].min()
        closest_plane = close_df[close_df['dist_from_rzeszow'] == closest_flight_to_rz]
        closest_flight_to_rz = round(closest_flight_to_rz)
        closest_plane = closest_plane['callsign'].iloc[0]

    else:
        closest_flight_to_rz = 999
        closest_plane = 'NONE'

    msg = f'closest flight to Rzeszow is {closest_flight_to_rz} miles away: {closest_plane}'
    if closest_flight_to_rz == None:
        closest_flight_to_rz = 200 # place holder for na
    #if closest_flight_to_rz <= 350:
        #api.send_direct_message(841931767, msg)
    new = write_df[write_df['callsign'].isin(newly_added)]
    #print(f'closest flight to Rzeszow is {closest_flight_to_rz} miles away: {closest_plane}')

    n_after = len(write_df)

    # add date and times
    write_df['last_update_dt'] = now

    # determine the next time to check in
    if closest_flight_to_rz <= 190: 
        write_df['next_update_dt'] = datetime.now() + timedelta(minutes=1)
        print(f'will check again next run')

    elif closest_flight_to_rz <= 300: ## two runs at 241 == 80 miles out
        write_df['next_update_dt'] = datetime.now() + timedelta(minutes=15)
        print(f'will check again in two runs')

    elif closest_flight_to_rz <= 400: ## three runs at 320 = 80 miles out
        write_df['next_update_dt'] = datetime.now() + timedelta(minutes=25)
        print(f'will check again in a half hour')  

    elif closest_flight_to_rz <= 480: ## four runs at 400 == 80 miles out
        write_df['next_update_dt'] = datetime.now() + timedelta(minutes=35)
        print(f'will check again in 35 minutes')

    elif closest_flight_to_rz <= 560: ## five runs at 480 == 80 miles out
        write_df['next_update_dt'] = datetime.now() + timedelta(minutes=45)
        print(f'will check again in 45 minutes')     

    elif closest_flight_to_rz <= 640: ## six runs at 560 == 80 miles out
        write_df['next_update_dt'] = datetime.now() + timedelta(minutes=55)
        print(f'will check again in an hour')

    elif closest_flight_to_rz <= 720: ## 7 runs at 640 == 80 miles out
        write_df['next_update_dt'] = datetime.now() + timedelta(minutes=65)
        print(f'will check again in 65 minutes') 

    elif closest_flight_to_rz <= 800: ## 8 runs at 720 == 80 miles out
        write_df['next_update_dt'] = datetime.now() + timedelta(minutes=75)
        print(f'will check again in 75 minutes')   

    elif closest_flight_to_rz <= 920: ## 9 runs at 800 == 80 miles out
        write_df['next_update_dt'] = datetime.now() + timedelta(minutes=85)
        print(f'will check again in 1.5 hours')

    else:
        write_df['next_update_dt'] = datetime.now() + timedelta(minutes=105)
        print(f'will check again in 1:45')

    print('next possible update date is', write_df['next_update_dt'].iloc[0])

    write_df['prev_dist_from_rzeszow'] = write_df['dist_from_rzeszow']
    # drop unnecessary cols
    prev_x_cols = [c for c in write_df.columns if c.startswith('prev_dist_from_rzeszow_')]
    #print('write_df columns:', write_df.columns)
    write_df = write_df.drop(columns = prev_x_cols)

    # add icao_code where missing
    write_df['icao_code'] = write_df['dep_icao']

    ## add source
    write_df['source'] = ''
    write_df['source'] = np.where(write_df['callsign'].str.startswith('RCH'), 'US Air Force', write_df['source'])
    write_df['source'] = np.where(write_df['callsign'].str.startswith('CFC'), 'Canadian Armed Forces', write_df['source'])
    write_df['source'] = np.where(write_df['callsign'].str.startswith('ASY'), 'Australian Air Force', write_df['source'])
    write_df['source'] = np.where(write_df['callsign'].str.startswith('RRR'), 'British Royal Air Force', write_df['source'])
    write_df['source'] = np.where(write_df['callsign'].str.startswith('IAM'), 'Italian Air Force', write_df['source'])
    write_df['source'] = np.where(write_df['callsign'].str.startswith('HKY'), 'US Air Force', write_df['source'])
    write_df['source'] = np.where(write_df['callsign'].str.startswith('HERKY'), 'US Air Force', write_df['source'])
    write_df['source'] = np.where(write_df['callsign'].str.startswith('CTM'), 'French Air Force', write_df['source'])
    write_df['source'] = np.where(write_df['callsign'].str.startswith('DUKE'), 'US Army', write_df['source'])
    write_df['source'] = np.where(write_df['callsign'].str.startswith('KING'), 'US Air Force', write_df['source'])

    # write back to DB
    write_df.to_sql("rch_flights", con=engine, if_exists='replace', index=False)

    # compare this to the main flight_db
    rch = write_df['callsign'].unique()
    main = pd.read_sql_table("flight_db",  con=engine)
    main = main[main['callsign'].str.startswith('RCH')]
    main_rch = main['callsign'].unique()
    n_main = len(main)

    # diffs
    rch_only = [c for c in rch if c not in main_rch]
    main_only = [c for c in main_rch if c not in rch]

    # print(f'Today have tracked {n_after} flights in rch_flights and {n_main} RCH flights in flights_db')

    # print(f'{rch_only} are just in rch')
    # print(f'{main_only} are just in the main table')

    # print(f'Now tracking {n_after} RCH flights')

########################################
## BUILD DEPARTURE DF
########################################

# - dep_df: A dataframe returned from find_origin() with the following attributes:
#             - callsign: callsign of aircraft (ID)
#             - icao_code: airport code
#             - lat: latitude of departing airport (Y), first appearance (M), or fudged current location (N)
#             - lng: longtitude of departing airport (Y), first appearance (M), or fudged current location (N)
#             - aircraft: aircraft_icao (aircraft code) from airlabs
#             - country_code: country code from airlabs airports API (CSV)
#             - name: airport name
#             - known_dep: whether flight departing airport is known (Y), estimated (M), or not known (N)

airports = pd.read_csv('data/airports.csv')
airports['icao_code'] = airports['icao_code'].str.strip()
airports = airports[~airports['icao_code'].isna()]
airports = airports[airports['icao_code'] != '']

rch_flights = pd.read_sql_table("rch_flights",  con=engine)
rch_flights = rch_flights[rch_flights['date'] == today]
#rch_flights = rch_flights[rch_flights['drop_date'] <= now]
if len(rch_flights) >= 1:
    flights_db = pd.read_sql_table("flight_db",  con=engine)

    rch_calls = rch_flights['callsign'].unique()
    other_flights = flights_db['callsign'].unique()

    keep_calls = [c for c in rch_calls if c not in other_flights]

    rch_flights = rch_flights[rch_flights['callsign'].isin(keep_calls)]

    def lat_long_dist (row):
            coord = (row['lat'], row['long'])
            rzeszow_coord = (50.105999576, 22.017999928)
            dist = geopy.distance.geodesic(coord, rzeszow_coord).mi

            return(dist)
    rch_flights['dist_from_rzeszow'] = rch_flights.apply(lambda row: lat_long_dist (row),axis=1)

    # rename to standard
    rch_dep_df = rch_flights.drop(columns = ['lat', 'long']) # don't need current, just starting
    #rch_dep_df = rch_flights
    if 'dep_icao' in rch_dep_df.columns:
        rch_dep_df = rch_dep_df.drop(columns = ['dep_icao'])
    rch_dep_df = rch_dep_df.rename(columns = {'dep_icao':'icao_code'})
                                            #'start_lat':'lat',
                                            #'start_long':'lng'})

# try to merge in airport info
#rch_dep_df = rch_dep_df.merge(airports, how = 'left', on = ['icao_code'])

#rch_dep_df['country_code'] = np.where(rch_dep_df['country_code'].isna(), '', rch_dep_df['country_code'])
#rch_dep_df['name'] = np.where(rch_dep_df['name'].isna(), '', rch_dep_df['name'])    

# if lat/long of airport are known use that, otherwise use starting
#rch_dep_df['known_dep'] = np.where(rch_dep_df['icao_code'].isna(), 'M', 'Y')

##### SET DEPARTING LAT AND LONG #######
# if you know the departing airport
    rch_dep_df['lat'] = np.where(rch_dep_df['known_dep'] == 'Y', rch_dep_df['dep_airport_lat'], None)
    rch_dep_df['lng'] = np.where(rch_dep_df['known_dep'] == 'Y', rch_dep_df['dep_airport_long'], None)

    # if you don't
    rch_dep_df['lat'] = np.where(rch_dep_df['known_dep'] == 'M', rch_dep_df['start_lat'], rch_dep_df['lat'])
    rch_dep_df['lng'] = np.where(rch_dep_df['known_dep'] == 'M', rch_dep_df['start_long'], rch_dep_df['lng'])

                                        
    rch_dep_df = rch_dep_df[['callsign', 'icao_code', 'lat', 'lng', 'aircraft', 'country_code', 
                            'name', 'known_dep', 'dist_from_rzeszow']]

    rch_dep_df_pot = rch_dep_df[rch_dep_df['dist_from_rzeszow'] <= 180]
    #rch_dep_df_pot = rch_dep_df[rch_dep_df['dist_from_rzeszow'] > 25]
    rch_dep_df_rz = rch_dep_df[rch_dep_df['dist_from_rzeszow'] <= 25]

#print('rch_dep_df')
#print(rch_dep_df)

########################################
## BUILD PLOT DF
#       - callsign: callsign of aircraft (ID)
#       - icao_code: airport code
#       - lat: latitude of departing airport (Y), first appearance (M), or fudged current location (N)
#       - long: longtitude of departing airport (Y), first appearance (M), or fudged current location (N)
#       - aircraft: aircraft_icao (aircraft code) from
#       - country_code: country code from airlabs airports API (CSV)`
#       - name: airport name
#       - known_dep: whether flight departing airport is known (Y), estimated (M), or not known (N)
########################################

    if 'altitude' in rch_flights.columns:
        rch_crnt_df = rch_flights.drop(columns = ['altitude'])
    rch_crnt_df = rch_crnt_df.rename(columns = {'dep_icao':'icao_code',
                                            'alt':'altitude'})
    ##rch_crnt_df['country_code'] = ''    
    #rch_crnt_df['name'] = ''     
    #rch_crnt_df['known_dep'] = 'M' 
    rch_crnt_df = rch_crnt_df[['callsign', 'icao_code', 'lat', 'long', 'aircraft', 'country_code', 
                                'name', 'known_dep', 'dist_from_rzeszow', 'altitude', 'source']]

    rch_crnt_df_pot = rch_crnt_df[rch_crnt_df['dist_from_rzeszow'] <= 200]
    rch_crnt_df_rz = rch_crnt_df[rch_crnt_df['dist_from_rzeszow'] <= 25]
else:
    rch_crnt_df_pot = pd.DataFrame()
    rch_crnt_df_rz = pd.DataFrame()
    rch_dep_df_pot = pd.DataFrame()
    rch_dep_df_rz = pd.DataFrame()
#print('rch_crnt_df')
#print(rch_crnt_df)
#print("\n##################################################### \n")

#################################################################################
# just to be safe, try and catch flights that may be leaving rzeszow that are RCH
#################################################################################

rch_db = pd.read_sql_table("rch_flights",  con=engine)
rch_db = rch_db[rch_db['date'] == today]
#rch_db = rch_db[rch_db['drop_date'] <= now]

rc_rz = rch_db.copy()
rc_rz['keep'] = np.where(rch_db['dep_icao'].isin(['EPRJ', 'EPRZ']), True, False)

rc_rz = rc_rz[rc_rz['keep'] == True]
print(len(rc_rz), 'flights from Rzeszow today')

if len(rc_rz) >= 1:
    for callsign in rc_rz['callsign'].unique():
        cdf = rc_rz[rc_rz['callsign'] == callsign] # pipeline
        dep_airport = 'EPRJ'
        aircraft = cdf['aircraft'].iloc[0]
        operator = cdf['source'].iloc[0]
        # if callsign.startswith('RRR'):
        #     operator = 'British Royal Air Force'
        # elif callsign.startswith('RCH'):
        #     operator = 'US Air Force'
        # elif callsign.startswith('IAM'):
        #     operator = 'Italian Air Force'
        # elif callsign.startswith('CFC'):
        #     operator = 'Canadian Armed Forces'

        crnt_loc_df = cdf[['callsign', 'lat', 'long']] #.reset_index(drop=True)
        dep_loc_df = cdf[['callsign']] #.reset_index(drop=True)
        dep_loc_df['lat'] = 50.105999576
        dep_loc_df['long'] = 22.017999928

        plot_df = pd.concat([dep_loc_df, crnt_loc_df], ignore_index=True)

        # now plot ###########################################################
        title = f'{operator} {aircraft} with callsign {callsign} departing Rzeszow ({dep_airport})'
        fig = go.Figure(go.Scattergeo(lat=plot_df['lat'],
                                        lon=plot_df['long'],
                                        mode = 'lines',
                                        line = dict(width = 2.5,color = 'black', dash = 'solid')))
        ##  add dot for rzeszow    
        fig.add_trace(go.Scattergeo(lon = [22.017999928], 
                                    lat = [50.105999576], 
                                    marker = dict(size = 6,
                                                color = "red",
                                                line_color='red',
                                                line_width=0.5,
                                                sizemode = 'area')))

        # formatting
        fig.update_layout(
            showlegend = False,
            title_text = title,
            title_font_color="black",
            margin=dict(l=0, r=0, t=0, b=0),
            title={
                'text': title,
                'y':0.97,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top'},
            geo = dict(projection_type = "equirectangular",
                    showcountries = True,
                    showland = True,
                    showocean = True,
                    landcolor = '#C3ECB2',
                    lakecolor = '#AADAFF',
                    oceancolor = '#AADAFF',
                    countrycolor = 'gray',
            lataxis = dict(range = [30, 70],),
            lonaxis = dict(range = [-20, 50],)),)

        arrow_lat1 = 50.105999576
        arrow_lng1 = 22.017999928
    
        arrow_lat2 = crnt_loc_df['lat'].iloc[0]
        arrow_lng2 = crnt_loc_df['long'].iloc[0]
    
        l = 1.1  # the arrow length
        widh =0.035  #2*widh is the width of the arrow base as triangle
    
        A = np.array([arrow_lng1, arrow_lat1])
        B = np.array([arrow_lng2, arrow_lat2]) # crnt_lat_long[['lat', 'lng',
        v = B-A
        w = v/np.linalg.norm(v)     
        u  =u = np.array([-w[1], w[0]])*10  #u orthogonal on  w
    
        P = B-l*w
        S = P - widh*u
        T = P + widh*u

        # add arrow
        fig.add_trace(go.Scattergeo(lon = [S[0], T[0], B[0], S[0]], 
                                    lat =[S[1], T[1], B[1], S[1]], 
                                    mode='lines', 
                                    fill='toself', 
                                    fillcolor='black', 
                                    line_color='black'))

        fig.write_image("images/fig2.png") #), height = 700, width = 700)

        # check whether to tweet
        twt = all_tweets[all_tweets['text'].str.contains(callsign)] # tweets
        twt_deliv = twt[twt['text'].str.contains('goodies for the', case=False)] # delivered tweets
        twt_int = twt[twt['text'].str.contains('interest', case=False)] # delivered tweets


        if len(twt_deliv) < 1:
            if callsign.startswith('RRR'):
                print(f'{callsign} flight just arrived')
            else:
                
                tweet = f'{operator} {aircraft} with callsign {callsign} is departing Rzeszow ({dep_airport}) after delivering goodies for the Ukrainians.'
                img = "images/fig2.png"
                if len(twt_int) >=1:
                    print(f'responding for {callsign}: {tweet}')
                    respond_tweet_id = twt_int['id'].iloc[0]
                    api.update_status_with_media(tweet, img, in_reply_to_status_id = respond_tweet_id)
                else:
                    print(f'gonna tweet for {callsign}')
                    api.update_status_with_media(tweet, img)
                    print(tweet)
        else:
            print(f'already tweeted about {callsign} arriving')


