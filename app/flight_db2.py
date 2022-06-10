from sqlalchemy.types import Integer, DateTime
from sqlalchemy import create_engine
import sqlalchemy
import os
os.environ["TZ"] = "UTC"
import datetime
from datetime import datetime, timedelta
import geopy.distance
import plotly.express as px
import numpy as np
import requests
import json
import pandas as pd

pd.options.mode.chained_assignment = None

##########################################################
# FUNCTION
##########################################################

"""
Pull data on military flights in NW hemisphere and add their location to database
"""

def flight_loc_db(init=False):

    raw_db_url = os.getenv("DATABASE_URL")
    final_db_url = "postgresql+psycopg2://" + raw_db_url.lstrip("postgres://")
    engine = create_engine(final_db_url)

    # get current dates and times
    now = datetime.now()

    ## start by commiting the CSV
    if init == True:
        csv_path = '/Users/conorkelly/Documents/ukraine/flight_origins.csv'
        start_df = pd.read_csv(csv_path)
        start_df.to_sql("flight_db", con=engine, if_exists='replace', index=False)


    # begin function
    callsigns = ['RCH', 'CFC', 'ASY', 'RFR', 'RRR', 'EEF', 'FNF', 'HRZ',
                                           'NVY', 'CONVOY', 'CNV', 'NATO', 'DAF',
                                           'DUKE', 'TUAF', 'BRK', 'RFR', 'IAM',
                                           'ADB', 'HKY', 'HERKY', 'NOW',  'ROF', # DLH2VR'
                                           'CTM', 'GAF', 'SCF', 'NAF', 'BAF']

    # AREA EXTENT COORDINATE WGS4
    lat_min, lat_max = 25, 71
    lon_min, lon_max = -102, 32

    # REST API QUERY
    user_name = ''
    password = ''
    url_data = 'https://'+user_name+':'+password+'@opensky-network.org/api/states/all?' + \
        'lamin='+str(lat_min)+'&lomin='+str(lon_min)+'&lamax=' + \
        str(lat_max)+'&lomax='+str(lon_max)
    response = requests.get(url_data).json()

    # LOAD TO PANDAS DATAFRAME
    col_name = ['icao24', 'callsign', 'origin_country', 'time_position', 'last_contact', 'long', 'lat', 'baro_altitude', 'on_ground', 'velocity',
                'true_track', 'vertical_rate', 'sensors', 'geo_altitude', 'squawk', 'spi', 'position_source']
    flight_df = pd.DataFrame(response['states'])
    flight_df = flight_df.loc[:, 0:16]
    flight_df.columns = col_name
    flight_df = flight_df.fillna('No Data')  # replace NAN with No Data

    # get only military flights
    flight_df['mil'] = False
    mil_call_signs = callsigns
    for i in mil_call_signs:
        flight_df['mil'] = np.where(
            flight_df['callsign'].str.startswith(i), True, flight_df['mil'])

    flight_df = flight_df[flight_df['mil'] == True]
    flight_df['callsign'] = flight_df['callsign'].str.strip()

    print('Latest pull returned', len(flight_df), 'military planes in NW hemisphere')
    if len(flight_df) >= 1:

        # don't take nulls
        flight_df = flight_df[~flight_df['callsign'].isna()]
        flight_df['baro_altitude'] = pd.to_numeric(flight_df['baro_altitude'], errors='coerce') * 3.25

        n_flights = len(flight_df)

        # Find flights closest to rzeszow
        def lat_long_dist (row):
            coord = (row['lat'], row['long'])
            rzeszow_coord = (50.105999576, 22.017999928)
            dist = geopy.distance.geodesic(coord, rzeszow_coord).mi

            return(dist)

        flight_df['dist_from_rzeszow'] = flight_df.apply(lambda row: lat_long_dist (row),axis=1)

        dist_df = flight_df.copy()
        dist_df['dist_from_rzeszow'] = dist_df.apply(lambda row: lat_long_dist (row),axis=1)
        dist_df = dist_df[['callsign', 'dist_from_rzeszow', 'baro_altitude']]
        #dist_df['baro_altitude'] = pd.to_numeric(dist_df['baro_altitude'], errors='coerce') * 3.25
        dist_df = dist_df[dist_df['baro_altitude'] >= 5000]
        dist_df = dist_df[dist_df['dist_from_rzeszow'] >= 100]
        dist_df = dist_df.sort_values(['dist_from_rzeszow'], ascending=False)
        
        # check how many are in flight db for today
        #csv_path = 'data/flight_origins.csv'
        flight_db = pd.read_sql_table("flight_db",  con=engine)
        #flight_db['insert_time'] = now
        n_crnt = len(flight_db['callsign'].unique())

        # deal with column names
        new_cols = [c.lower() for c in flight_db.columns]
        flight_db.columns = new_cols

        # filter to only flights from today
        a = len(flight_db)
        today = datetime.today().strftime('%Y-%m-%d')
        #print(today)
    

        #flight_db = flight_db[flight_db['date'] == today]
        flight_db = flight_db[flight_db['insert_time'] >= now - timedelta(hours=24)]
        # flight_db['date'] = today
        
        # add timestamp
        ts = datetime.now()
        flight_db['timestamp'] = ts

        diff = a - len(flight_db)
        if diff > 0:
            print(f'Dropping {diff} flights from the tracker')

        # get list of callsigns
        db_calls = flight_db['callsign'].unique()

        # for any new flights, check whether they are in flight_db already for today
        returned_calls = flight_df['callsign'].unique()

        new_calls = [c for c in returned_calls if c not in db_calls]

        ####### low flying to new_calls, drop callsign if low flying
        low_flights = flight_df[flight_df['baro_altitude'] <= 5000]
        low_calls = low_flights['callsign'].unique()
        low_calls = [c for c in low_calls if c in db_calls] # drop only if already in database, not taking off
        print(len(low_calls), 'are low and being removed:', low_calls)
        #print(len(flight_db))
        flight_db = flight_db[~flight_db['callsign'].isin(low_calls)]
        #print(len(flight_db))

        n_new_calls = len(new_calls)

        # proceed only if there are flights to add or delete
        if n_new_calls >= 1 or len(low_calls) >= 1:

            # get only new flights in a df
            add_df = flight_df[flight_df['callsign'].isin(new_calls)]

            add_df = add_df[['callsign', 'lat', 'long']]
            add_df['date'] = today
            add_df['insert_time'] = ts
            #add_df['date'] = pd.to_datetime(flight_db['date'])

            # append to flight_db
            n_old = len(flight_db)
            flight_db = pd.concat([flight_db, add_df], ignore_index=True)
            flight_db = flight_db[~flight_db['callsign'].isna()]
            flight_db = flight_db.drop_duplicates()  # (['callsign'])
            n_new = len(flight_db)

            # check uniqueness
            assert n_new == len(flight_db['callsign'].unique())

            # write to dataframe
            flight_db['date'] = pd.to_datetime(flight_db['date'])
            flight_db['timestamp'] = ts
            print(f'Adding {n_new_calls} new callsigns to database for {today}: {new_calls}')
            flight_db.to_sql("flight_db", con=engine, if_exists='replace', index=False)

        else:
            print(f'No new flights to add to origin database for {today}.')

        ## check which flights are closest to rzeszow

        print('Currently, the closest flights to Rzeszow are:\n')
        print(dist_df.tail())

##########################################################

# def drop_low_alt():

