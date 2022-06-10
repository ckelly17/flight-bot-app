import requests
import os
os.environ["TZ"] = "UTC"
import json
import pandas as pd
pd.options.mode.chained_assignment = None
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import geopy.distance
import tweepy
import datetime
from datetime import timedelta
import gmplot
from sqlalchemy import create_engine

#from dev.plot_and_tweet import plot_layers


####################################
def load_my_tweets(api):
    
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

    # filter to only those from the last day
    tweetz['dt'] = pd.to_datetime(tweetz['created_at']).dt.date
    today = datetime.date.today()
    tweetz = tweetz[tweetz['dt'] >= today]
    
    # return tweets about landing and tweets about interest
    interest_tweets = tweetz[tweetz['text'].str.contains('interest', case=False)]
    arrival_tweets = tweetz[tweetz['text'].str.contains('goodies', case=False)]
    
    return(tweetz, interest_tweets, arrival_tweets)

#####################################
def pull_flight_data(api):

    #AREA EXTENT COORDINATE WGS4
    lon_min,lat_min=18.413, 48.195
    lon_max,lat_max=25.203, 52.282

    #REST API QUERY
    user_name=''
    password=''
    url_data='https://'+user_name+':'+password+'@opensky-network.org/api/states/all?'+'lamin='+str(lat_min)+'&lomin='+str(lon_min)+'&lamax='+str(lat_max)+'&lomax='+str(lon_max)
    response=requests.get(url_data).json()

    #LOAD TO PANDAS DATAFRAME
    col_name=['icao24','callsign','origin_country','time_position','last_contact','long','lat','baro_altitude','on_ground','velocity',       
    'true_track','vertical_rate','sensors','geo_altitude','squawk','spi','position_source']
    try:
        flight_df=pd.DataFrame(response['states'])
        flight_df=flight_df.loc[:,0:16]
        flight_df.columns=col_name
        flight_df=flight_df.fillna('No Data') #replace NAN with No Data
        
        print(len(flight_df), 'flights currently within a radius of Rzeszow')
    except:
        msg = 'issue with opensky api'
        print(msg)
        api.send_direct_message(841931767, msg)
    
    return flight_df

#####################################

def get_mil_flights(input_df, callsigns = ['RCH', 'CFC', 'ASY', 'RFR', 'RRR', 'EEF', 'FNF', 'HRZ',
                                           'NVY', 'CONVOY', 'CNV', 'NATO', 'DAF',
                                           'DUKE', 'TUAF', 'BRK', 'RFR', 'IAM',
                                           'ADB', 'HKY', 'HERKY', 'NOW',  'ROF', # DLH2VR'
                                           'CTM', 'GAF', 'SCF', 'NAF', 'BAF']): #'DLH2VR'
    
    #['RCH', 'CFC', 'ASY', 'RFR', 'RRR', 'NVY', 'CONVOY', 'ENT']
    
    df = input_df.copy()
    df['mil'] = False
    mil_call_signs = callsigns
    for i in mil_call_signs:
        df['mil'] = np.where(df['callsign'].str.startswith(i), True, df['mil'])
        
    df['source'] = ''
    df['source'] = np.where(df['callsign'].str.startswith('RCH'), 'US Air Force', df['source'])
    df['source'] = np.where(df['callsign'].str.startswith('CFC'), 'Canadian Armed Forces', df['source'])
    df['source'] = np.where(df['callsign'].str.startswith('ASY'), 'Australian Air Force', df['source'])
    df['source'] = np.where(df['callsign'].str.startswith('RFR'), 'British Royal Air Force', df['source'])
    df['source'] = np.where(df['callsign'].str.startswith('RRR'), 'British Royal Air Force', df['source'])
    df['source'] = np.where(df['callsign'].str.startswith('NVY'), 'Royal Navy', df['source'])
    df['source'] = np.where(df['callsign'].str.startswith('CONVOY'), 'US Air Force', df['source'])
    df['source'] = np.where(df['callsign'].str.startswith('EEF'), 'Estonian Air Force', df['source'])
    df['source'] = np.where(df['callsign'].str.startswith('FNF'), 'Finnish Air Force', df['source'])
    df['source'] = np.where(df['callsign'].str.startswith('HRZ'), 'Croatian Air Force', df['source'])
    df['source'] = np.where(df['callsign'].str.startswith('GRF'), 'German Air Force', df['source'])
    df['source'] = np.where(df['callsign'].str.startswith('CNV'), 'US Navy', df['source'])
    df['source'] = np.where(df['callsign'].str.startswith('NATO'), 'NATO', df['source'])
    df['source'] = np.where(df['callsign'].str.startswith('COBB'), 'US Air Force', df['source'])
    df['source'] = np.where(df['callsign'].str.startswith('DUKE'), 'US Army', df['source'])
    df['source'] = np.where(df['callsign'].str.startswith('TUAF'), 'Turkish Air Force', df['source'])
    df['source'] = np.where(df['callsign'].str.startswith('BRK'), 'NATO', df['source'])
    df['source'] = np.where(df['callsign'].str.startswith('RFR'), 'Royal Air Force', df['source'])
    df['source'] = np.where(df['callsign'].str.startswith('PAT'), 'US Army', df['source'])
    df['source'] = np.where(df['callsign'].str.startswith('IAM'), 'Italian Air Force', df['source'])
    df['source'] = np.where(df['callsign'].str.startswith('ADB'), 'Antonov Airlines', df['source'])
    df['source'] = np.where(df['callsign'].str.startswith('HKY'), 'US Air Force', df['source'])
    df['source'] = np.where(df['callsign'].str.startswith('HERKY'), 'US Air Force', df['source'])
    df['source'] = np.where(df['callsign'].str.startswith('NOW'), 'Norwegian Air Force', df['source'])
    df['source'] = np.where(df['callsign'].str.startswith('ROF'), 'Romanian Air Force', df['source'])
    df['source'] = np.where(df['callsign'].str.startswith('CTM'), 'French Air Force', df['source'])
    df['source'] = np.where(df['callsign'].str.startswith('GAF'), 'German Air Force', df['source'])
    df['source'] = np.where(df['callsign'].str.startswith('SCF'), 'Swedish Air Force', df['source'])
    df['source'] = np.where(df['callsign'].str.startswith('NAF'), 'Netherlands Air Force', df['source'])
    df['source'] = np.where(df['callsign'].str.startswith('BAF'), 'Belgian Air Force', df['source'])
    df['source'] = np.where(df['callsign'].str.startswith('DAF'), 'Danish Air Force', df['source'])
    #df['source'] = np.where(df['callsign'].str.startswith('DHK303'), 'Test Airline', df['source'])

    df = df[df['mil'] == True]
    
    df['dist_from_rzeszow'] = None
    rzeszow_coord = (50.105999576, 22.017999928)

    for i in range(0, len(df)):
        test = df[:i+1]
        lat = test['lat'].iloc[i]
        long = test['long'].iloc[i]
        coord = (lat, long)
        dist = geopy.distance.geodesic(coord, rzeszow_coord).mi
        df['dist_from_rzeszow'].iloc[i] = dist
    
    df['altitude'] = pd.to_numeric(df['baro_altitude'], errors = 'coerce') * 3.25
    df['altitude'] = np.where(df['altitude'].isna(), 0, df['altitude'])

    # strip from callsign
    df['callsign'] = df['callsign'].str.strip()

    # exclude if callsign is DUKE and below 10000 feet
    cond = ((df['callsign'].str.startswith('DUKE')) & (df['altitude'] < 15000) & (df['dist_from_rzeszow'] >= 15))

    df['mil'] = np.where(cond, False, df['mil'])
    df = df[df['mil'] == True]
        
    disp_df = df[['callsign', 'source', 'origin_country', 'altitude', 'dist_from_rzeszow', 'lat', 'long']]

    # exclude if callsign is DUKE and below 10000 feet

    
    print(len(df), 'flights have tracked military callsigns.')
    if len(df) >= 1:
        print('')
        disp_df2 = disp_df[['callsign', 'dist_from_rzeszow', 'altitude']]
        print(disp_df2)
    
    return df
 
#####################################
       
def get_deliveries(df):
    in_rzeszow = df
    in_rzeszow['deliv'] = False
    
    # all planes
    cond = ((in_rzeszow['altitude'] <= 15000) & (in_rzeszow['dist_from_rzeszow'] <= 25) & (~in_rzeszow['callsign'].str.startswith('DUKE')))
            
    in_rzeszow['deliv'] = np.where(cond, True, in_rzeszow['deliv'])
    
    # DUKE - could be low flying helicopter, need a tigher radius
    cond = ((in_rzeszow['altitude'] <= 10000) & (in_rzeszow['dist_from_rzeszow'] <= 15) & (in_rzeszow['callsign'].str.startswith('DUKE')))
            
    in_rzeszow['deliv'] = np.where(cond, True, in_rzeszow['deliv'])
    
    # filter
    in_rzeszow = in_rzeszow[in_rzeszow['deliv'] == True]
    
    print(len(in_rzeszow), 'military planes in Rzeszow, Poland')
    
    disp_df = in_rzeszow[['callsign', 'dist_from_rzeszow', 'altitude']]
    if len(disp_df) >= 1:
        print(disp_df.head())
    
    return in_rzeszow
                
#######################################################################                
def foreign_flights_in_rzeszow(api, df, tweets_df): # flight_df # arrival_tweets

    from datetime import datetime, timedelta

    raw_db_url = 'postgres://wzypmlovttdnzl:bb530351daefbc1560ae45f0d052533e2dfc1b4f430c5aa6a71ee2d7f94049aa@ec2-54-164-40-66.compute-1.amazonaws.com:5432/d7g9sn5bbegjbs'
    final_db_url = "postgresql+psycopg2://" + raw_db_url.lstrip("postgres://")
    engine = create_engine(final_db_url)

    now = datetime.now()
    today = datetime.today().strftime('%Y-%m-%d')   

    # get rch and flight_db
    flight_db = pd.read_sql_table("flight_db",  con=engine)
    rch_db = pd.read_sql_table("rch_flights",  con=engine)

    flights_main = flight_db['callsign'].unique()
    flights_rch = rch_db['callsign'].unique()

    # calculate distance from rzeszow
    df['dist_from_rzeszow'] = None
    rzeszow_coord = (50.105999576, 22.017999928)

    for i in range(0, len(df)):
        test = df[:i+1]
        lat = test['lat'].iloc[i]
        long = test['long'].iloc[i]
        coord = (lat, long)
        dist = geopy.distance.geodesic(coord, rzeszow_coord).mi
        df['dist_from_rzeszow'].iloc[i] = dist
    
    # get all flights within 10 miles    
    df = df[df['dist_from_rzeszow'] < 10]
    df['altitude'] = pd.to_numeric(df['baro_altitude'], errors = 'coerce') * 3.25
    df['altitude'] = np.where(df['altitude'].isna(), 0, df['altitude'])
    df = df[df['altitude'] < 25000]
    
    # exclude flights based on certain conditions
    exclude_countries = ['PL', 'Poland'] # if origin is poland, ignore
    df = df[~df['origin_country'].isin(exclude_countries)]

    # exclude military
    df = df[~df['callsign'].isin(flights_main)]
    df = df[~df['callsign'].isin(flights_rch)]
    df = df[~df['callsign'].str.startswith('DUKE')]

    print(len(df), "other interesting planes of any type in Rzeszow, Poland")
    if len(df) >= 1:
        print(df.head())
    
    # look up departing airport
    if len(df) >= 1:
        for callsign in df['callsign'].unique():
        #for i in range(0, (len(df))):
            #tweet_df = df.iloc[[i]]            
            # check whether I've tweeted about this callsign
            # if not, pull more data from airlabs API
            callsign_tweets = tweets_df[tweets_df['text'].str.contains(callsign)]
            print(callsign_tweets)
            tweet_df = df[df['callsign'].str.contains(callsign)]
            print(tweet_df)
            print(len(callsign_tweets))
            
            # condition 1: if already tweeted about arrival, do nothing
            if len(callsign_tweets) >= 1:
                if callsign not in ['', None]:
                    print(f'Already tweeted about this arrival: {callsign}')
            else:
                dist = round(df['dist_from_rzeszow'].iloc[0])
                print(callsign)
                url = f'https://flightaware.com/live/flight/{callsign}'
                
                # pull callsign and airport info from airlabs
                apikey = 'ecbe0dd8-db38-4b37-8283-1899091a1b69'
                print(f'API call to airlabs incoming for {callsign}')
                try:
                    url_data = f'https://airlabs.co/api/v9/flights?flight_icao={callsign}&api_key={apikey}'
                    apikey = '2e6da084-608a-444e-9bb2-fdff525ff8c9' # main api (ckelly52)
                    response = requests.get(url_data).json()
                except:
                    apikey = '274a066d-69d3-4990-b4f7-c563e3f2f1f5' # deloitte api
                    url_data = f'https://airlabs.co/api/v9/flights?flight_icao={callsign}&api_key={apikey}'
                    response = requests.get(url_data).json()

                api_df = pd.DataFrame({'type': {0: 'in_rzeszow'}, 'time':{0:now},'date':{0:today}, 
                        'callsign':{0:callsign}, 'key':{0:apikey}})
                api_df.to_sql("api_calls", con=engine, if_exists='append', index=False)
                try:
                    airlabs = pd.DataFrame(response['response'])
                    print(len(airlabs), 'matched flights')
                except:
                    airlabs = pd.DataFrame()
                
                # only proceed for matched flights
                if len(airlabs) >= 1 and 'dep_icao' in airlabs.columns:
                    airlabs = airlabs[['flight_icao', 'reg_number', 'dep_icao', 'aircraft_icao']]
                    print(airlabs)
                    dep_airport = airlabs['dep_icao'].iloc[0].strip()
                    aircraft = airlabs['aircraft_icao'].iloc[0].strip()
                    operator = ''
                        
                    # load the departing airport name and country
                    airports = pd.read_csv('data/airports.csv')
                    airports = airports[airports['icao_code'] == dep_airport]
                    name = airports['name'].iloc[0]
                    airport_lat_long = airports[['icao_code', 'lat', 'lng']]
                    country_code = airports['country_code'].iloc[0]
                    if country_code == 'DE':
                        country_code = 'Germany'
                    elif country_code ==  'GB':
                        country_code = 'Great Britain'
                    
                    # get tweet
                    tweet = f'Goodies for the Ukranians?\n{aircraft} with callsign {callsign} out of {name}-{dep_airport} ({country_code}) has landed in Rzeszow, Poland. \n \n {url}'
                    
                    # get current flight lat/long
                    crnt_lat_long = tweet_df[['lat', 'long']]
                    crnt_lat_long['lng'] = crnt_lat_long['long']
                    crnt_lat_long['icao_code'] = callsign
                    crnt_lat_long = crnt_lat_long[['lat', 'lng', 'icao_code']]
        
                    # append
                    plot_df = pd.concat([airport_lat_long, crnt_lat_long], ignore_index=True)
                    plot_df['label_col'] = np.where(plot_df['icao_code'] == dep_airport, plot_df['icao_code'], ' ')
                    
                    #### plot
                    #plot_layers(plot_df, callsign, country_code, operator, name, airport_lat_long, crnt_lat_long, known_dep = 'Y')
                    # # determine whether flight coming from north america for plot boundary
                    if country_code in ['US', 'USA', 'CA', 'CAN']:
                        origin = 'US'
                    elif country_code in ['DE', 'Germany']:
                        origin = 'GER'
                    else:
                        origin = 'UK'
                    
                    # set plot boundaries
                    if origin == 'US':
                        lon_east = -100
                        lat_south = 0
                    elif origin == 'GER':
                        lon_east = 0
                        lat_south = 35
                    elif origin == 'UK':
                        lon_east = -20
                        lat_south = 30
                    
                    # set title
                    plot_title = f'{callsign} out of {name} ({country_code})'
                    
                    # plot figure
                    fig = go.Figure(go.Scattergeo(
                    lat=plot_df['lat'],
                    lon=plot_df['lng'],
                    mode = 'lines',
                    line = dict(width = 2.5,color = 'black')
                    ))
                    
                    fig.update_layout(
                    showlegend = False,
                    title_text = plot_title,
                    title_font_color="black",
                    margin=dict(l=0, r=0, t=0, b=0),
                    title={
                        'text': plot_title,
                        'y':0.97,
                        'x':0.5,
                        'xanchor': 'center',
                        'yanchor': 'top'},
                    geo = dict(
                    
                        projection_type = "equirectangular",
                        showcountries = True,
                        showland = True,
                        showocean = True,
                        landcolor = '#C3ECB2',
                        lakecolor = '#AADAFF',
                        oceancolor = '#AADAFF',
                        countrycolor = 'gray',
                        lataxis = dict(
                        range = [lat_south, 70],
                    ),
                        lonaxis = dict(
                        range = [lon_east, 50],
                    )
                    ),
                    )
                    
                    #Workaround to get the arrow at the end of an edge AB
                    arrow_lat1 = airport_lat_long['lat'].iloc[0]
                    arrow_lng1 = airport_lat_long['lng'].iloc[0]
                    
                    arrow_lat2 = crnt_lat_long['lat'].iloc[0]
                    arrow_lng2 = crnt_lat_long['lng'].iloc[0]
                    
                    l = 1.1  # the arrow length
                    widh =0.035  #2*widh is the width of the arrow base as triangle
                    
                    A = np.array([arrow_lng1, arrow_lat1])
                    B = np.array([arrow_lng2, arrow_lat2])
                    v = B-A
                    w = v/np.linalg.norm(v)     
                    u  =u = np.array([-w[1], w[0]])*10  #u orthogonal on  w
                    
                    P = B-l*w
                    S = P - widh*u
                    T = P + widh*u
                    
                    fig.add_trace(go.Scattergeo(lon = [S[0], T[0], B[0], S[0]], 
                                    lat =[S[1], T[1], B[1], S[1]], 
                                    mode='lines', 
                                    fill='toself', 
                                    fillcolor='black', 
                                    line_color='black'))
                    
                    fig.add_trace(go.Scattergeo(lon = [arrow_lng1], 
                                    lat = [arrow_lat1], 
                        marker = dict(
                        size = 6,
                        color = "black",
                        line_color='black',
                        line_width=0.5,
                        sizemode = 'area')))
                    fig.show()
                    fig.write_image("images/fig1.png") #), height = 700, width = 700)
                
                
                    img = "images/fig1.png"
                    
                    # tweet depending on whether plane was in rzeszow or not
                    if origin in ['US']:
                        print(tweet)
                        api.update_status_with_media(tweet, img)
                    else:
                        print('None from US/UK')
                
                
                
                
                    
