# -*- coding: utf-8 -*-
"""ATORtf_detectEllipses.py

# Author:
Richard Bruce Baxter - Copyright (c) 2021 Baxter AI (baxterai.com)

# License:
MIT License

# Requirements:
opencv-python>=4.5.3.56
Pillow>=8.0.1
click>=7.1.2
numpy>=1.19.2

# Usage:
python ATORtf_detectEllipses.py images/leaf1.png

# Description:
Perform ATOR ellipse detection using open-cv library (non-hardware accelerated)

See [Vinmorel](https://github.com/vinmorel/Genetic-Algorithm-Image) for genetic algorithm implementation.

"""


import cv2
import copy
#import click
import numpy as np
from collections import OrderedDict
import ATORtf_operations
import ATORtf_ellipseProperties


numberOfResolutions = 6	#x; lowest res sample: 1/(2^x)
minimumEllipseLength = 2
ellipseCenterCoordinatesResolution = 1	#pixels (at resolution r)
ellipseAxesLengthResolution = 1	#pixels (at resolution r)
ellipseAngleResolution = 10	#degrees
ellipseColourResolution = 64	#bits

def detectEllipsesGaussianBlur(inputimagefilename):
	
	inputImage = cv2.imread(inputimagefilename)
		
	inputImageHeight, inputImageWidth, inputImageChannels = inputImage.shape
	print("inputImageHeight = ", inputImageHeight, "inputImageWidth = ", inputImageWidth, ", inputImageChannels = ", inputImageChannels)
	blankArray = np.full((inputImageHeight, inputImageWidth, 3), 255, np.uint8)
	outputImage = blankArray
	
	ellipsePropertiesOptimumNormalisedAllRes = []
	
	testEllipseIndex = 0
	
	for resolutionIndex in range(1, numberOfResolutions):
	
		resolutionIndexReverse = numberOfResolutions-resolutionIndex+1
		resolutionFactor = 2**resolutionIndexReverse
		
		#gaussianBlurKernelSize = (resolutionIndexReverse*2) - 1		
		gaussianBlurKernelSize = (resolutionFactor*2) - 1	#ensure kernel size is odd
		print("gaussianBlurKernelSize = ", gaussianBlurKernelSize)
		inputImageR = gaussianBlur(inputImage, gaussianBlurKernelSize)
		
		#inputImageR = cv2.resize(inputImage, None, fx=resolutionFactorInverse, fy=resolutionFactorInverse)
		
		imageHeight, imageWidth, imageChannels = inputImageR.shape
		print("resolutionFactor = ", resolutionFactor, ", imageHeight = ", imageHeight, "imageWidth = ", imageWidth, ", imageChannels = ", imageChannels)
		
		thresh = cv2.cvtColor(inputImageR, cv2.COLOR_RGB2GRAY)
		#ATORtf_operations.displayImage(inputImageR)
		thresh = cv2.adaptiveThreshold(thresh, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 101, 0)
		if(ATORtf_operations.opencvVersion==3):
			NULL, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
		elif(ATORtf_operations.opencvVersion==4):
			contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)	#or RETR_TREE
		
		#ret, thresh = cv2.threshold(thresh, 127, 255, cv2.THRESH_BINARY_INV)	#or cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU #binarize
		#contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)	#or RETR_TREE	
		
		minimumArea = 500
		
		inputImageRdev = copy.deepcopy(inputImageR) 
		cv2.drawContours(inputImageRdev, contours, -1, (0, 255, 0), 3)
		
		for cnt in contours:
			area = cv2.contourArea(cnt)
			if area > minimumArea:
				print("cnt")
				averageColour = calculateAverageColourOfContour(inputImageR, cnt)
				ellipse = cv2.fitEllipse(cnt)
				cv2.ellipse(inputImageRdev, ellipse, averageColour, 3)

		ATORtf_operations.displayImage(inputImageRdev)	#debug
																
def detectEllipsesTrialResize(inputimagefilename):
	
	inputImage = cv2.imread(inputimagefilename)
		
	inputImageHeight, inputImageWidth, inputImageChannels = inputImage.shape
	print("inputImageHeight = ", inputImageHeight, "inputImageWidth = ", inputImageWidth, ", inputImageChannels = ", inputImageChannels)
	blankArray = np.full((inputImageHeight, inputImageWidth, 3), 255, np.uint8)
	outputImage = blankArray
	
	ellipsePropertiesOptimumNormalisedAllRes = []
	
	testEllipseIndex = 0
	
	for resolutionIndex in range(1, numberOfResolutions):
	
		resolutionIndexReverse = numberOfResolutions-resolutionIndex+1
		resolutionFactor = 2**resolutionIndexReverse
		resolutionFactorReverse = 2**resolutionIndex
		resolutionFactorInverse = 1.0/(resolutionFactor)
		#print("resolutionIndex = ", resolutionIndex, ", resolutionFactor = ", resolutionFactor)
		inputImageR = cv2.resize(inputImage, None, fx=resolutionFactorInverse, fy=resolutionFactorInverse)
		imageHeight, imageWidth, imageChannels = inputImageR.shape

		#ATORtf_operations.displayImage(inputImageR)	#debug

		print("resolutionFactor = ", resolutionFactor, ", imageHeight = ", imageHeight, "imageWidth = ", imageWidth, ", imageChannels = ", imageChannels)
		
		#match multiple ellipses for each resolution
		ellipsePropertiesOrderedDict = OrderedDict()
		
		#reduce max size of ellipse at each res
		#axesLengthMax1 = imageWidth
		#axesLengthMax2 = imageHeight
		axesLengthMax1 = imageWidth//resolutionFactorReverse * 4	#CHECKTHIS
		axesLengthMax2 = imageHeight//resolutionFactorReverse * 4	#CHECKTHIS
		print("axesLengthMax1 = ", axesLengthMax1, ", axesLengthMax2 = ", axesLengthMax2)
		
		for centerCoordinates1 in range(0, imageWidth, ellipseCenterCoordinatesResolution):
			for centerCoordinates2 in range(0, imageHeight, ellipseCenterCoordinatesResolution):
				for axesLength1 in range(minimumEllipseLength, axesLengthMax1, ellipseAxesLengthResolution):
					for axesLength2 in range(minimumEllipseLength, axesLengthMax2, ellipseAxesLengthResolution):
						for angle in range(0, 360, ellipseAngleResolution):	#degrees
							for colour1 in range(0, 256, ellipseColourResolution):
								for colour2 in range(0, 256, ellipseColourResolution):
									for colour3 in range(0, 256, ellipseColourResolution):

										imageSize = (imageWidth, imageHeight)
										centerCoordinates = (centerCoordinates1, centerCoordinates2)
										axesLength = (axesLength1, axesLength2)
										colour = (colour1, colour2, colour3)
										
										ellipseProperties = ATORtf_ellipseProperties.EllipseProperties(resolutionIndex, resolutionFactor, imageSize, centerCoordinates, axesLength, angle, colour)
										inputImageRmod, ellipseFitError = ATORtf_ellipseProperties.testEllipseApproximation(inputImageR, ellipseProperties)
	
										ellipsePropertiesOrderedDict[ellipseFitError] = ellipseProperties
										testEllipseIndex = testEllipseIndex + 1
										
										#ATORtf_ellipseProperties.printEllipseProperties(ellipseProperties)

																									
		ellipsePropertiesOptimumNormalisedR = []
		for ellipseFitError, ellipseProperties in ellipsePropertiesOrderedDict.items():
			
			ellipsePropertiesNormalised = ATORtf_ellipseProperties.normaliseEllipseProperties(ellipseProperties)
			
			ellipseOverlapsesWithPreviousOptimumEllipse = False
			for ellipseProperties2 in ellipsePropertiesOptimumNormalisedR:
				if(ATORtf_ellipseProperties.centroidOverlapsEllipseWrapper(ellipseFitError, ellipsePropertiesNormalised, ellipseProperties2)):
					ellipseOverlapsesWithPreviousOptimumEllipse = True
						
			if(not ellipseOverlapsesWithPreviousOptimumEllipse):
				ellipsePropertiesNormalisedOptimumLast = ellipsePropertiesNormalised
				ellipsePropertiesOptimumNormalisedAllRes.append(ellipsePropertiesNormalisedOptimumLast)
				ellipsePropertiesOptimumNormalisedR.append(ellipsePropertiesNormalisedOptimumLast)
				#inputImageRmod, ellipseFitError = ATORtf_ellipseProperties.testEllipseApproximation(inputImageR, ellipseProperties)
				outputImage = ATORtf_ellipseProperties.drawEllipse(outputImage, ellipsePropertiesNormalisedOptimumLast)
				ATORtf_operations.displayImage(outputImage)
				ATORtf_operations.saveImage(inputimagefilename, outputImage)

		#quit()

def gaussianBlur(inputImage, gaussianBlurKernelSize):
	result = cv2.GaussianBlur(src=inputImage, ksize=(gaussianBlurKernelSize,gaussianBlurKernelSize), sigmaX=20.0, borderType=cv2.BORDER_DEFAULT)
	return result
	
def calculateAverageColourOfContour(inputImageR, cnt):
	x,y,w,h = cv2.boundingRect(cnt) # offsets - with this you get 'mask'
	cv2.rectangle(inputImageR, (x,y), (x+w,y+h), (0,255,0), 2)
	#Average color (BGR):
	averageColour = cv2.mean(inputImageR[y:y+h,x:x+w])
	#np.array(averageColour).astype(np.uint8))
	#print("averageColour = ", averageColour)
	return averageColour
					

#@click.command()
#@click.argument('inputimagefilename')

def main(inputimagefilename):
	#detectEllipsesTrialResize(inputimagefilename)
	detectEllipsesGaussianBlur(inputimagefilename)

#if __name__ == "__main__":
#	main()
