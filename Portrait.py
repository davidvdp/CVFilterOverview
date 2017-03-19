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

import cv2
import numpy as np
import os
import sys

class Portrait:
    """A simple portrait class"""
    originalImage = None
    imageWithMargins = None
    headerHeightPerc = 0.20
    marginPerc = 0.02
    m_font = cv2.FONT_HERSHEY_COMPLEX
    m_thickness = 1
    m_filtersize = 5

    m_name = "Original"

    def __init__ (self, imageLocation):
        if not os.path.isfile(imageLocation):
            raise IOError("File " + imageLocation + " Not Found")
        self.originalImage = cv2.imread(imageLocation)
        self.AddMargins()
        self.WriteSecondLineText()
        self.imageWithMargins = self.DoFilter(self.imageWithMargins)
        if len(self.imageWithMargins.shape) is 2:
            self.imageWithMargins = self.ToColor(self.imageWithMargins)
        self.WriteFirstLineText()

    def __del__(self):
        pass

    def AddMargins(self):
        marginpx = int(self.originalImage.shape[1] * self.marginPerc)
        headerpx = int(self.originalImage.shape[1] * self.headerHeightPerc)
        self.imageWithMargins = np.zeros([self.originalImage.shape[0]+marginpx + headerpx,self.originalImage.shape[1]+2*marginpx,self.originalImage.shape[2]],dtype=np.uint8)
        self.imageWithMargins[:, :, :] = 255
        self.imageWithMargins[headerpx:headerpx+self.originalImage.shape[0], marginpx:marginpx+self.originalImage.shape[1]] = self.originalImage

    def WriteFirstLineText(self):
        headerpx = int(self.originalImage.shape[1] * self.headerHeightPerc)
        fontsize = headerpx / 100
        thickness = self.m_thickness
        textsize = cv2.getTextSize(self.m_name, self.m_font, fontsize, thickness)

        cv2.putText(self.imageWithMargins, self.m_name,
                    (self.imageWithMargins.shape[1] / 2 - textsize[0][0] / 2, headerpx / 3), self.m_font, fontsize,
                    (255, 255, 255), thickness+1)
        cv2.putText(self.imageWithMargins, self.m_name,
                    (self.imageWithMargins.shape[1]/2-textsize[0][0]/2, headerpx / 3), self.m_font, fontsize,
                    (0, 0, 0), thickness)

    def WriteSecondLineText(self):
        headerpx = int(self.originalImage.shape[1] * self.headerHeightPerc)
        fontsize = headerpx / 100
        thickness = self.m_thickness
        textsize = cv2.getTextSize(self.m_name, self.m_font, fontsize, thickness)

        cv2.putText(self.imageWithMargins, self.m_name,
                    (self.imageWithMargins.shape[1] / 2 - textsize[0][0] / 2, headerpx/2 + headerpx / 3), self.m_font, fontsize,
                    (255, 255, 255), thickness+1)
        cv2.putText(self.imageWithMargins, self.m_name,
                    (self.imageWithMargins.shape[1]/2-textsize[0][0]/2, headerpx/2 + headerpx / 3), self.m_font, fontsize,
                    (0, 0, 0), thickness)

    def DoFilter(self, image):
        """Returns filtered image"""
        return image

    def GetImage(self):
        return self.imageWithMargins

    def ToGray(self, image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def Blur(self, image):
        return cv2.blur(image,(self.m_filtersize,self.m_filtersize))

    def ToColor(self, image):
        return  cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    def Theshold(self, image):
        startThreshold = 10;
        if len(image.shape) is 3:
            image = self.ToGray(image)
        multiplier = 1
        mean = 0
        while mean < 128:
            ret, threshImage = cv2.threshold(image, int(255 - startThreshold * multiplier), 255, cv2.THRESH_BINARY)
            mean = np.mean(threshImage)
            multiplier += 1
            if startThreshold * multiplier > 255:
                break;
        image = threshImage
        return image


class SobelXFilter(Portrait):
    m_name = "SobelX"
    def DoFilter(self, image):
        image = self.ToGray(image)
        image = self.Blur(image)
        image = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize = self.m_filtersize)
        image = np.absolute(image)
        max = np.max( image )
        image = np.uint8(image*255/max)
        image = self.ToColor(image)
        return image

class SobelYFilter(Portrait):
    m_name = "SobelY"
    def DoFilter(self, image):
        image = self.ToGray(image)
        image = self.Blur(image)
        image = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize = self.m_filtersize)
        image = np.absolute(image)
        max = np.max( image )
        image = np.uint8(image*255/max)
        image = self.ToColor(image)
        return image

class SobelFilter(Portrait):
    m_name = "Sobel"
    def DoFilter(self, image):
        image = self.ToGray(image)
        image = self.Blur(image)
        imageX = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize = self.m_filtersize)
        imageX = np.absolute(imageX)
        imageY = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize = self.m_filtersize)
        imageY = np.absolute(imageY)
        image = imageX + imageY
        max = np.max(image)
        image = np.uint8(image*255/max)
        image = self.ToColor(image)
        return image

class LaPlacianFilter(Portrait):
    m_name = "Laplacian"
    def DoFilter(self, image):
        image = self.ToGray(image)
        image = self.Blur(image)
        image = cv2.Laplacian(image,cv2.CV_64F,ksize = self.m_filtersize)
        image = np.absolute(image)
        max = np.max(image)
        image = np.uint8(image*255/max)
        image = self.ToColor(image)
        return image


class CannyEdgeFilter(Portrait):
    m_name = "Canny Edge"
    def DoFilter(self, image):
        minVal = 25
        maxVal = 100
        expectedmean = 13 #is used to let canny produce enough edges

        image = self.ToGray(image)
        image = self.Blur(image)

        #optimize canny
        mean = 0
        multiplier = 255/maxVal
        imageCan = None
        while mean < expectedmean:
            imageCan = cv2.Canny(image, int(minVal * multiplier), int(maxVal * multiplier))
            mean = np.mean(imageCan)
            multiplier -= 0.5
            #print "mean: " + str(mean)
            #print "minVal: " + str(minVal * multiplier)
            #cv2.imshow("test",imageCan)
            #cv2.waitKey()
            if minVal * multiplier <= 0:
                break

        image = imageCan

        image = self.ToColor(image)
        return image

class BlurFilter(Portrait):
    m_name = "Blur"
    def DoFilter(self, image):
        return cv2.blur(image,(self.m_filtersize*3,self.m_filtersize*3))

class GaussianBlurFilter(Portrait):
    m_name = "Gaussian Blur"
    def DoFilter(self, image):
        return cv2.GaussianBlur(image,(self.m_filtersize*7,self.m_filtersize*7),0)

class ThresholdFilter(Portrait):
    m_name = "Threshold"
    def DoFilter(self, image):
        return self.Theshold(image)

class ErodeFilter(Portrait):
    m_name = "Erode"
    def DoFilter(self, image):
        image = self.Theshold(image)
        size = 5
        image = cv2.erode(image, (self.m_filtersize * size, self.m_filtersize * size))
        return cv2.erode(image,(self.m_filtersize*size,self.m_filtersize*size))

class DilateFilter(Portrait):
    m_name = "Dilate"
    def DoFilter(self, image):
        image = self.Theshold(image)
        size = 5
        image = cv2.dilate(image, (self.m_filtersize * size, self.m_filtersize * size))
        return cv2.dilate(image,(self.m_filtersize*size,self.m_filtersize*size))