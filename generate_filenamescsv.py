import os
import pandas as pd

df = []
for folder in os.listdir("."):
    files = [folder+"_"+file for file in os.listdir(os.path.join(".", folder, "ROW"))  if file.endswith(".jpg")]
    df = df + files

df2 =  pd.DataFrame(sorted(df) , columns = ['filename'])
df2.to_csv("Batch22C_allfilenames.csv", index = False)