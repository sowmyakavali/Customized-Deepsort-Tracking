import os
import pandas as pd 

path = "Batch22C_WithUID.csv"
data = pd.read_csv(path)
df = []
for i, row in data.iterrows():
    if 'POLE' in row['Asset_name'] or 'Pole' in row['Asset_name']:
        df.append([row['filename'], row['X_min'], row['Y_min'], row['X_max'], row['Y_max'], 
                        row['Asset_name'].upper(), row['Asset_Id'], row['Conf'] ])

d = pd.DataFrame(df, columns = ['filename', 'X1', 'Y1', 'X2', 'Y2', 'class_name',
                                'uniqueid', 'Conf'])
d.to_csv("modified_latest_" + path, index = False)