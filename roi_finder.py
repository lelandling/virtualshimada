import pandas as pd
import json
import sys
from paquo.projects import QuPathProject
from paquo.images import QuPathImageType
import os.path
from openslide import open_slide
import openslide
from PIL import Image

# paquo installed

def roi_finder(): 
    svspath = "/Users/lelandling/Documents/Stanford work/neuroblastoma - shimada project/Shimada/batch8"
    qppath = "/Users/lelandling/Documents/Stanford work/neuroblastoma - shimada project/Shimada/test/"
    roipath = "/Users/lelandling/Documents/Stanford work/neuroblastoma - shimada project/missingrois/"

    # path = input("Enter project file path: ")
    qp = QuPathProject(roipath, mode='r')  # open project for reading

    rois = []

    for image in qp.images :  
        size = len(image.hierarchy.annotations)  # annotations are stored in a set like proxy object
        # print(image.hierarchy.annotations[0].roi.bounds)
        # print(image.image_name)
        maxbounds = [ [1]*4 for i in range(size)] # create return list
        for i in range(size):
            bounds = image.hierarchy.annotations[i].roi.bounds
            print(bounds)

            # find top left corner, and dimensions, (x, y, w, h)
            maxbounds[i][2] = round(bounds[2] - bounds[0])
            maxbounds[i][3] = round(bounds[3] - bounds[1])
            maxbounds[i][0] = round(bounds[0])
            maxbounds[i][1] = round(bounds[3] - maxbounds[i][3])

            # comment out below, shows regions of interest
            # slide = open_slide(svspath + image.image_name)
            # tb = slide.read_region((maxbounds[i][0], maxbounds[i][1]), 0, (maxbounds[i][2], maxbounds[i][3]))
            # tb.show()
            
        rois.append([image.image_name, maxbounds])
    # coordinates are stored in maxbound as left up down right
    return rois

if __name__ == "__main__":
    bound = roi_finder()
    # f = open('rois.txt', 'w')

    with open('rois11.txt', 'w') as f:
        for imagetuple in bound:
            f.writelines(imagetuple[0])
            f.writelines('\n')
            f.writelines('\n'.join((str(thing) for thing in imagetuple[1])))
            f.writelines('\n\n')
            