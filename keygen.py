from click import Path
import numpy as np
import pandas as pd
from traitlets import default
import csv

def makekey(): 
    # path = input("Enter key file path: ")
    defaultpath = "Stanford COG cases Grade of differentiation.csv"
    key1 = pd.read_csv(defaultpath, header = None)
    key1.columns = key1.head(1).values.flatten()
    key1 = key1.iloc[1:,:]
    # key1 = key1.dropna(axis='columns')
    # print(key1.head())
    cog = key1.iloc[:, 0].values
    types = key1.iloc[:, 1].values
    rawkey = list(zip(cog, types))

    # print(rawkey)
    key = {}
    for cog, type in rawkey: 
        if cog not in key.items(): 
            key[cog] = type

    return(key)

if __name__ == "__main__":
    dict = makekey()
    with open("key.csv", "w", newline="") as f:
        writer = csv.writer(f)
        for key in dict.keys():
            f.write("%s,%s\n"%(key,dict[key]))
