#-------------------------------------------------------------------------------
# Name:        module1
# Purpose:
#
# Author:      David
#
# Created:     13/03/2017
# Copyright:   (c) David 2017
# Licence:     <your licence>
#-------------------------------------------------------------------------------

import Portrait
import cv2
import numpy as np
import os
import sys

def main(argv):
    if len(argv) == 0:
        print "Please provide an image. Example: Create.py image.png"

    sourceImagelocation = argv[0]
    windowNameFilter = "Filter"
    windowNameAll = "All"
    TotalImageSize = (4, 3)

    allFilters = list()

    allFilters.append(Portrait.Portrait(sourceImagelocation))
    allFilters.append(Portrait.SobelXFilter(sourceImagelocation))
    allFilters.append(Portrait.SobelYFilter(sourceImagelocation))
    allFilters.append(Portrait.SobelFilter(sourceImagelocation))
    allFilters.append(Portrait.LaPlacianFilter(sourceImagelocation))
    allFilters.append(Portrait.CannyEdgeFilter(sourceImagelocation))
    allFilters.append(Portrait.BlurFilter(sourceImagelocation))
    allFilters.append(Portrait.GaussianBlurFilter(sourceImagelocation))
    allFilters.append(Portrait.ThresholdFilter(sourceImagelocation))
    allFilters.append(Portrait.ErodeFilter(sourceImagelocation))
    allFilters.append(Portrait.DilateFilter(sourceImagelocation))

    totalImageCol = 0
    totalImageRow = 0

    image = allFilters[0].GetImage()
    imageHeight = image.shape[0]
    imageWidth = image.shape[1]
    imageDepth = image.shape[2]

    imageAll = np.zeros((imageHeight*TotalImageSize[1], imageWidth*TotalImageSize[0], imageDepth), dtype=np.uint8)

    for filter in allFilters:
        image = filter.GetImage()
        imageAll[totalImageRow*imageHeight:(totalImageRow+1)*imageHeight, totalImageCol*imageWidth:(totalImageCol+1)*imageWidth] = image
        #cv2.imshow(windowNameFilter, image)
        #cv2.waitKey()
        totalImageCol += 1
        if totalImageCol >= TotalImageSize[0]:
            totalImageCol = 0
            totalImageRow += 1
            if totalImageRow >= TotalImageSize[1]:
                break

    cv2.destroyWindow(windowNameFilter)

    scale = 600.0/max(imageAll.shape[0],imageAll.shape[1])

    filename, file_extension = os.path.splitext(sourceImagelocation)

    cv2.imwrite(filename + "_Collage" + file_extension, imageAll)
    print "Written File: " + filename + "_Collage" + file_extension
    imageAll = cv2.resize(imageAll, (int(imageAll.shape[1]*scale),int(imageAll.shape[0]*scale)))
    #cv2.imshow(windowNameAll, imageAll)
    #cv2.waitKey()

if __name__ == '__main__':
    try:
        main(sys.argv[1:])
    except IOError as e:
        print "I/O error: {0}".format(e.message)
    except:
        print "An unexpected error occured"