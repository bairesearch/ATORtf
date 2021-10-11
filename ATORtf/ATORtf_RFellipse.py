# -*- coding: utf-8 -*-
"""ATORtf_RFellipse.py

# Author:
Richard Bruce Baxter - Copyright (c) 2021 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
See ATORtf.tf

# Usage:
See ATORtf.tf

# Description:
ATORtf RF Ellipse - generate ellipse receptive fields

"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import cv2
import copy

import ATORtf_RFProperties
import ATORtf_ellipseProperties
import ATORtf_operations


ellipseAngleResolution = 10	#degrees
minimumEllipseFitErrorRequirement = 1500.0	#calibrate

ellipseNormalisedAngle = 0.0
ellipseNormalisedCentreCoordinates = 0.0 
ellipseNormalisedAxesLength = 1.0

minimumEllipseLength = 2
ellipseAxesLengthResolution = 1	#pixels (at resolution r)
ellipseAngleResolution = 30	#degrees
ellipseColourResolution = 64	#bits

receptiveFieldOpponencyArea = 2.0	#the radius of the opponency/negative (-1) receptive field compared to the additive (+) receptive field

def normaliseGlobalEllipseProperties(ellipseProperties, resolutionFactor):
	return ATORtf_ellipseProperties.normaliseGlobalEllipseProperties(ellipseProperties, resolutionFactor)

def normaliseLocalEllipseProperties(RFProperties):
	#normalise ellipse respect to major/minor ellipticity axis orientation (WRT self)
	RFPropertiesNormalised = copy.deepcopy(RFProperties)
	RFPropertiesNormalised.angle = ellipseNormalisedAngle	#CHECKTHIS
	RFPropertiesNormalised.centerCoordinates = (ellipseNormalisedCentreCoordinates, ellipseNormalisedCentreCoordinates, ellipseNormalisedCentreCoordinates)
	RFPropertiesNormalised.axesLength = (ellipseNormalisedAxesLength, ellipseNormalisedAxesLength)
	return RFPropertiesNormalised

def calculateFilterPixels(filterSize, numberOfDimensions):
	internalFilterSize = getInternalFilterSize(filterSize, numberOfDimensions)	#CHECKTHIS: only consider contribution of positive (additive) pixels
	#print("internalFilterSize = ", internalFilterSize)
	
	if(numberOfDimensions == 2):
		numberOfFilterPixels = internalFilterSize[0]*internalFilterSize[1]
	elif(numberOfDimensions == 3):
		numberOfFilterPixels = internalFilterSize[0]*internalFilterSize[1]	#CHECKTHIS
	#print("numberOfFilterPixels = ", numberOfFilterPixels)
	return numberOfFilterPixels

def getInternalFilterSize(filterSize, numberOfDimensions):
	if(numberOfDimensions == 2):
		internalFilterSize = (int(filterSize[0]/receptiveFieldOpponencyArea), int(filterSize[1]/receptiveFieldOpponencyArea))
	elif(numberOfDimensions == 3):
		internalFilterSize = (int(filterSize[0]/receptiveFieldOpponencyArea), int(filterSize[1]/receptiveFieldOpponencyArea))	#CHECKTHIS
	return internalFilterSize
	

def generateRFFiltersEllipse(resolutionProperties, RFFiltersList, RFFiltersPropertiesList):

	#2D code;
	
	#filters are generated based on human magnocellular/parvocellular/koniocellular wavelength discrimination in LGN and VX (double/opponent receptive fields)
	
	#magnocellular filters (monochromatic);
	colourH = (255, 255, 255)	#high
	colourL = (-255, -255, -255)	#low
	RFFiltersHL, RFPropertiesHL = generateRotationalInvariantRFFilters(resolutionProperties, False, colourH, colourL)
	RFFiltersLH, RFPropertiesLH = generateRotationalInvariantRFFilters(resolutionProperties, False, colourL, colourH)
	
	#parvocellular/koniocellular filters (based on 2 cardinal colour axes; ~red-~green, ~blue-~yellow);
	colourRmG = (255, -255, 0)	#red+, green-
	colourGmR = (-255, 255, 0)	#green+, red-
	colourBmY = (-127, -127, 255)	#blue+, yellow-
	colourYmB = (127, 127, -255)	#yellow+, blue-
	RFFiltersRG, RFPropertiesRG = generateRotationalInvariantRFFilters(resolutionProperties, True, colourRmG, colourGmR)
	RFFiltersGR, RFPropertiesGR = generateRotationalInvariantRFFilters(resolutionProperties, True, colourGmR, colourRmG)
	RFFiltersBY, RFPropertiesBY = generateRotationalInvariantRFFilters(resolutionProperties, True, colourBmY, colourYmB)
	RFFiltersYB, RFPropertiesYB = generateRotationalInvariantRFFilters(resolutionProperties, True, colourYmB, colourBmY)
	
	RFFiltersList.append(RFFiltersHL)
	RFFiltersList.append(RFFiltersLH)
	RFFiltersList.append(RFFiltersRG)
	RFFiltersList.append(RFFiltersGR)
	RFFiltersList.append(RFFiltersBY)
	RFFiltersList.append(RFFiltersYB)

	RFFiltersPropertiesList.append(RFPropertiesHL)
	RFFiltersPropertiesList.append(RFPropertiesLH)
	RFFiltersPropertiesList.append(RFPropertiesRG)
	RFFiltersPropertiesList.append(RFPropertiesGR)
	RFFiltersPropertiesList.append(RFPropertiesBY)
	RFFiltersPropertiesList.append(RFPropertiesYB)

def generateRotationalInvariantRFFilters(resolutionProperties, isColourFilter, filterInsideColour, filterOutsideColour):
	
	RFFiltersList2 = []
	RFFiltersPropertiesList2 = []
	
	#FUTURE: consider storing filters in n dimensional array and finding local minima of filter matches across all dimensions

	#reduce max size of ellipse at each res
	resolutionFactor, imageSize, axesLengthMax, filterRadius, filterSize = getFilterDimensions(resolutionProperties)
	
	#print("axesLengthMax = ", axesLengthMax)
	
	for axesLength1 in range(minimumEllipseLength, axesLengthMax[0]+1, ellipseAxesLengthResolution):
		for axesLength2 in range(minimumEllipseLength, axesLengthMax[1]+1, ellipseAxesLengthResolution):
			for angle in range(0, 360, ellipseAngleResolution):	#degrees
				
				axesLengthInside = (axesLength1, axesLength2)
				axesLengthOutside = (int(axesLength1*receptiveFieldOpponencyArea), int(axesLength2*receptiveFieldOpponencyArea))

				filterCenterCoordinates = (0, 0)
				RFtype = ATORtf_RFProperties.RFtypeEllipse
				RFPropertiesInside = ATORtf_RFProperties.RFPropertiesClass(resolutionProperties.resolutionIndex, resolutionFactor, filterSize, RFtype, filterCenterCoordinates, axesLengthInside, angle, filterInsideColour)
				RFPropertiesOutside = ATORtf_RFProperties.RFPropertiesClass(resolutionProperties.resolutionIndex, resolutionFactor, filterSize, RFtype, filterCenterCoordinates, axesLengthOutside, angle, filterOutsideColour)
				RFPropertiesInside.isColourFilter = isColourFilter
				RFPropertiesOutside.isColourFilter = isColourFilter

				RFFilter = generateRFFilter(resolutionProperties.resolutionIndex, isColourFilter, RFPropertiesInside, RFPropertiesOutside)
				RFFiltersList2.append(RFFilter)
				RFProperties = copy.deepcopy(RFPropertiesInside)
				#RFProperties.centerCoordinates = centerCoordinates 	#centerCoordinates are set after filter is applied to imageSegment
				RFFiltersPropertiesList2.append(RFProperties)	#CHECKTHIS: use RFPropertiesInside not RFPropertiesOutside

				#debug:
				#print(RFFilter.shape)
				if(resolutionProperties.debugVerbose):
					ATORtf_RFProperties.printRFProperties(RFPropertiesInside)
					ATORtf_RFProperties.printRFProperties(RFPropertiesOutside)				
				#print("RFFilter = ", RFFilter)

	#create 3D tensor (for hardware accelerated test/application of filters)
	RFFiltersTensor = tf.stack(RFFiltersList2, axis=0)

	return RFFiltersTensor, RFFiltersPropertiesList2
	

def generateRFFilter(resolutionProperties, isColourFilter, RFPropertiesInside, RFPropertiesOutside):

	# RF filter example (RFFilterTF):
	#
	# 0 0 0 0 0 0
	# 0 0 - - 0 0 
	# 0 - + + - 0
	# 0 0 - - 0 0
	# 0 0 0 0 0 0
	#
	# where "-" = -RFColourOutside [R G B], "+" = +RFColourInside [R G B], and "0" = [0, 0, 0]
	
	#generate ellipse on blank canvas
	#resolutionFactor, resolutionFactorReverse, imageSize = ATORtf_operations.getImageDimensionsR(resolutionProperties)
	blankArray = np.full((RFPropertiesInside.imageSize[1], RFPropertiesInside.imageSize[0], 1), 0, np.uint8)	#grayscale (or black/white)	#0: black	#or filterSize
	
	ellipseFilterImageInside = copy.deepcopy(blankArray)
	ellipseFilterImageOutside = copy.deepcopy(blankArray)

	RFPropertiesInsideWhite = copy.deepcopy(RFPropertiesInside)
	RFPropertiesInsideWhite.colour = (255, 255, 255)
	
	RFPropertiesOutsideWhite = copy.deepcopy(RFPropertiesOutside)
	RFPropertiesOutsideWhite.colour = (255, 255, 255)
	RFPropertiesInsideBlack = copy.deepcopy(RFPropertiesInside)
	RFPropertiesInsideBlack.colour = (000, 000, 000)
			
	ATORtf_ellipseProperties.drawEllipse(ellipseFilterImageInside, RFPropertiesInsideWhite)

	ATORtf_ellipseProperties.drawEllipse(ellipseFilterImageOutside, RFPropertiesOutsideWhite)
	ATORtf_ellipseProperties.drawEllipse(ellipseFilterImageOutside, RFPropertiesInsideBlack)
	
	insideImageTF = tf.convert_to_tensor(ellipseFilterImageInside, dtype=tf.float32)	#bool
	insideImageTF = tf.greater(insideImageTF, 0.0)
	insideImageTF = tf.dtypes.cast(insideImageTF, tf.float32)
	
	outsideImageTF = tf.convert_to_tensor(ellipseFilterImageOutside, dtype=tf.float32)
	outsideImageTF = tf.greater(outsideImageTF, 0.0)
	outsideImageTF = tf.dtypes.cast(outsideImageTF, tf.float32)

	#print("insideImageTF = ", insideImageTF)
	#print("outsideImageTF = ", outsideImageTF)
		
	#add colour channels;
	#insideImageTF = tf.expand_dims(insideImageTF, axis=2)
	multiples = tf.constant([1,1,3], tf.int32)	#for 2D data only
	insideImageTF = tf.tile(insideImageTF, multiples)
	#print(insideImageTF.shape)
	RFColourInside = tf.Variable([RFPropertiesInside.colour[0], RFPropertiesInside.colour[1], RFPropertiesInside.colour[2]], dtype=tf.float32)
	RFColourInside = ATORtf_operations.expandDimsN(RFColourInside, RFPropertiesInside.numberOfDimensions, axis=0)
	insideImageTF = tf.multiply(insideImageTF, RFColourInside)
	
	#outsideImageTF = tf.expand_dims(outsideImageTF, axis=2)
	multiples = tf.constant([1,1,3], tf.int32)	#for 2D data only
	outsideImageTF = tf.tile(outsideImageTF, multiples)
	#print(outsideImageTF.shape)
	RFColourOutside = tf.Variable([RFPropertiesOutside.colour[0], RFPropertiesOutside.colour[1], RFPropertiesOutside.colour[2]], dtype=tf.float32)
	RFColourOutside = ATORtf_operations.expandDimsN(RFColourOutside, RFPropertiesOutside.numberOfDimensions, axis=0)
	outsideImageTF = tf.multiply(outsideImageTF, RFColourOutside)
	
	#print("RFColourInside = ", RFColourInside)
	#print("RFColourOutside = ", RFColourOutside)
	#print("insideImageTF = ", insideImageTF)
	#print("outsideImageTF = ", outsideImageTF)
	
	#print(RFColourInside.shape)
	#print(RFColourOutside.shape)
	#print(insideImageTF.shape)
	#print(outsideImageTF.shape)
		
	RFFilterTF = tf.convert_to_tensor(blankArray, dtype=tf.float32)
	RFFilterTF = tf.add(RFFilterTF, insideImageTF)
	RFFilterTF = tf.add(RFFilterTF, outsideImageTF)
	
	if(ATORtf_operations.storeRFFiltersValuesAsFractions):
		RFFilterTF = tf.divide(RFFilterTF, ATORtf_operations.rgbMaxValue)

	#print("RFFilterTF = ", RFFilterTF)
	
	if(not isColourFilter):
		RFFilterTF = tf.image.rgb_to_grayscale(RFFilterTF)
			
	return RFFilterTF
	

def getFilterDimensions(resolutionProperties):
	resolutionFactor, resolutionFactorReverse, imageSize = ATORtf_operations.getImageDimensionsR(resolutionProperties)
	#reduce max size of ellipse at each res
	axesLengthMax1 = int(imageSize[0]//resolutionFactorReverse * 1 / 2)	#CHECKTHIS
	axesLengthMax2 = int(imageSize[1]//resolutionFactorReverse * 1 / 2)	#CHECKTHIS
	filterRadius = int(max(axesLengthMax1*receptiveFieldOpponencyArea, axesLengthMax2*receptiveFieldOpponencyArea)/2)	
	filterSize = (int(filterRadius*2), int(filterRadius*2))	#x/y dimensions are identical
	axesLengthMax = (axesLengthMax1, axesLengthMax2)
	
	#print("resolutionFactorReverse = ", resolutionFactorReverse)
	#print("resolutionFactor = ", imageSize)
	#print("axesLengthMax = ", axesLengthMax)
	
	return resolutionFactor, imageSize, axesLengthMax, filterRadius, filterSize	
	
