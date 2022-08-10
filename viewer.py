import numpy as np
import pandas as pd

import pandas as pdimport
import seaborn as sns
from typing import Tuple
import csv
from paquo.projects import QuPathProject
from paquo.images import QuPathImageType
from openslide import open_slide
import openslide
from PIL import Image
from sqlalchemy import true
import os
from os.path import exists

import ast

def viewregions() :
    rawdata = pd.read_csv('misclassified.csv', header = None)
    print(rawdata.iloc[:, 0])
    misclassified = rawdata.iloc[:, 0]


    with open('roiskey.txt') as f:
        lines = f.readlines()

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
            dictionary[line] = []
            while pointer != '':
                dictionary[line].append(ast.literal_eval(pointer))
                inc = inc+1
                pointer = slides[i+inc]
        i = i+ inc
    
    print(dictionary)

    root = "/Volumes/USB/"
    # key = '840737 A1.svs'
    # slide = open_slide(os.path.join(root, key))
    # rois = dictionary[key]
    # for region in rois:
    #     tb = slide.read_region((region[0], region[1]), 0, (region[2], region[3]))
    #     tb.show()
    for key in misclassified:
        number = int(key[-1:])
        print(number)
        slide = key[0:-2]
        # print(exists(os.path.join(root, slide)))
        image = open_slide(os.path.join(root, slide))
        rois = dictionary[slide]
        # rois = rois[number]
        print(key, number)
        print(rois[number-1])

        region = rois[number-1]
        # for region in rois:
        #     print()
        tb = image.read_region((region[0], region[1]), 0, (region[2], region[3]))
        tb.show()

        input("waiting for next set")


if __name__ == "__main__":
    viewregions()