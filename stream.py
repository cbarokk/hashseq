import pandas as pd
import numpy as np
from collections import defaultdict
import sys
from datetime import datetime
import time 

path = sys.argv[1]
store = pd.get_store(path)
nrows = store.get_storer('data').nrows

def load_data():
    start_date = store.select('data', start=1, stop=2).index[0]
    start_date = start_date + pd.Timedelta("25 days 16 hours 15 minutes")
    #end_date = start_date + pd.Timedelta("7 hours")
    end_date = start_date + pd.Timedelta("5 hours")
    print "loading data between ", start_date, " and ", end_date
    #where = 'index>"'+str(start_date) + '" & index<"' + str(end_date) + '" & typeCode == "01"'
    where = 'index>"'+str(start_date) + '" & index<"' + str(end_date) + '"' 

    df =  store.select('data', columns=['callingSubscriberIMSI', 'cell_ID'], where=where)
    df['colFromIndex'] = df.index
    df = df.sort(['callingSubscriberIMSI', 'colFromIndex'])
    
    #raw_input("Press Enter to continue...")
    last_cust = ""
    for r in df.itertuples(index=False):
        if r[0] == "":
            continue
        if last_cust != r[0]:
            print " "

        print r[2].strftime("%c"), ":  customer", r[0], "was at location", int(r[1])/10
        #time.sleep(1)
        last_cust = r[0]


if __name__ == '__main__':
    load_data()
