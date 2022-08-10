import numpy as np
import pandas as pd
from sklearn.model_selection import learning_curve, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import joblib
from scipy.io import loadmat
import matplotlib.pyplot as plt
import time
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import numpy as np
import copy as cp
import seaborn as sns
from typing import Tuple
import csv
import ast
import keygen

def counter():
    defaultroot = "/Users/lelandling/Documents/Stanford work/echonet/lelandtests/roiskey.txt"
    # root = input("enter roi path: ")
    with open(defaultroot, "r", newline = "") as f:
        lines = f.readlines()
        # [print(line) for line in f.readlines()]


    # with open('roiskey.txt') as f:
    #     lines = f.readlines()

    slides = []
    for line in lines:
        # print(line[:-1])
        slides.append(line[:-1])
    # print(slides)
    dictionary = {}
    for i in range(len(slides)): 
        line = slides[i]
        # print(line[-4:])
        if line[-4:] == '.svs':
            inc = 1
            pointer = slides[i+inc]
            rois = []
            dictionary[line] = 0
            while pointer != '':
                dictionary[line] = dictionary[line]  +1
                inc = inc+1
                pointer = slides[i+inc]
        i = i+ inc
    
    # print(dictionary)
    
    keys = keygen.makekey()
    # print(key)
    missing = []
    slides = {"poorly diff.": 0, "differentiating." : 0, "undiff.": 0}
    totalcohort = {"poorly diff.": 0, "differentiating." : 0, "undiff.": 0}
    for key in dictionary:
        try:
            slides[keys[key[0:6]]] = slides[keys[key[0:6]]] + 1
            totalcohort[keys[key[0:6]]] = totalcohort[keys[key[0:6]]] + dictionary[key]

        except:
            missing.append(key[0:6])

    print(slides)
    print(totalcohort)
    print(missing)

    

if __name__ == "__main__":
    counter()