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

import tensorflow as tf
import numpy as np
import cv2
import copy

import ATORtf_RFproperties
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

def normaliseLocalEllipseProperties(RFproperties):
	#normalise ellipse respect to major/minor ellipticity axis orientation (WRT self)
	RFpropertiesNormalised = copy.deepcopy(RFproperties)
	RFpropertiesNormalised.angle = ellipseNormalisedAngle	#CHECKTHIS
	RFpropertiesNormalised.centerCoordinates = (ellipseNormalisedCentreCoordinates, ellipseNormalisedCentreCoordinates, ellipseNormalisedCentreCoordinates)
	RFpropertiesNormalised.axesLength = (ellipseNormalisedAxesLength, ellipseNormalisedAxesLength)
	return RFpropertiesNormalised

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
		internalFilterSize = ((filterSize[0]/receptiveFieldOpponencyArea), (filterSize[1]/receptiveFieldOpponencyArea))
	elif(numberOfDimensions == 3):
		internalFilterSize = ((filterSize[0]/receptiveFieldOpponencyArea), (filterSize[1]/receptiveFieldOpponencyArea))	#CHECKTHIS
	return internalFilterSize
	

def generateRFfiltersEllipse(resolutionProperties, RFfiltersList, RFfiltersPropertiesList):

	#2D code;
	
	#filters are generated based on human magnocellular/parvocellular/koniocellular wavelength discrimination in LGN and VX (double/opponent receptive fields)
	
	#magnocellular filters (monochromatic);
	colourH = (255, 255, 255)	#high
	colourL = (-255, -255, -255)	#low
	RFfiltersHL, RFpropertiesHL = generateRotationalInvariantRFfilters(resolutionProperties, False, colourH, colourL)
	RFfiltersLH, RFpropertiesLH = generateRotationalInvariantRFfilters(resolutionProperties, False, colourL, colourH)
	
	#parvocellular/koniocellular filters (based on 2 cardinal colour axes; ~red-~green, ~blue-~yellow);
	colourRmG = (255, -255, 0)	#red+, green-
	colourGmR = (-255, 255, 0)	#green+, red-
	colourBmY = (-127, -127, 255)	#blue+, yellow-
	colourYmB = (127, 127, -255)	#yellow+, blue-
	RFfiltersRG, RFpropertiesRG = generateRotationalInvariantRFfilters(resolutionProperties, True, colourRmG, colourGmR)
	RFfiltersGR, RFpropertiesGR = generateRotationalInvariantRFfilters(resolutionProperties, True, colourGmR, colourRmG)
	RFfiltersBY, RFpropertiesBY = generateRotationalInvariantRFfilters(resolutionProperties, True, colourBmY, colourYmB)
	RFfiltersYB, RFpropertiesYB = generateRotationalInvariantRFfilters(resolutionProperties, True, colourYmB, colourBmY)
	
	RFfiltersList.append(RFfiltersHL)
	RFfiltersList.append(RFfiltersLH)
	RFfiltersList.append(RFfiltersRG)
	RFfiltersList.append(RFfiltersGR)
	RFfiltersList.append(RFfiltersBY)
	RFfiltersList.append(RFfiltersYB)

	RFfiltersPropertiesList.append(RFpropertiesHL)
	RFfiltersPropertiesList.append(RFpropertiesLH)
	RFfiltersPropertiesList.append(RFpropertiesRG)
	RFfiltersPropertiesList.append(RFpropertiesGR)
	RFfiltersPropertiesList.append(RFpropertiesBY)
	RFfiltersPropertiesList.append(RFpropertiesYB)

def generateRotationalInvariantRFfilters(resolutionProperties, isColourFilter, filterInsideColour, filterOutsideColour):
	
	RFfiltersList2 = []
	RFfiltersPropertiesList2 = []
	
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
				
				RFpropertiesInside = ATORtf_RFproperties.RFpropertiesClass(resolutionProperties.resolutionIndex, resolutionFactor, filterSize, ATORtf_RFproperties.RFtypeEllipse, filterCenterCoordinates, axesLengthInside, angle, filterInsideColour)
				RFpropertiesOutside = ATORtf_RFproperties.RFpropertiesClass(resolutionProperties.resolutionIndex, resolutionFactor, filterSize, ATORtf_RFproperties.RFtypeEllipse, filterCenterCoordinates, axesLengthOutside, angle, filterOutsideColour)
				RFpropertiesInside.isColourFilter = isColourFilter
				RFpropertiesOutside.isColourFilter = isColourFilter

				RFfilter = generateRFfilter(resolutionProperties.resolutionIndex, isColourFilter, RFpropertiesInside, RFpropertiesOutside)
				RFfiltersList2.append(RFfilter)
				
				RFproperties = copy.deepcopy(RFpropertiesInside)
				#RFproperties.centerCoordinates = centerCoordinates 	#centerCoordinates are set after filter is applied to imageSegment
				RFfiltersPropertiesList2.append(RFproperties)	#CHECKTHIS: use RFpropertiesInside not RFpropertiesOutside

				#debug:
				#print(RFfilter.shape)
				if(resolutionProperties.debugVerbose):
					ATORtf_RFproperties.printRFproperties(RFproperties)
					#ATORtf_RFproperties.printRFproperties(RFpropertiesInside)
					#ATORtf_RFproperties.printRFproperties(RFpropertiesOutside)				
				#print("RFfilter = ", RFfilter)

	#create 3D tensor (for hardware accelerated test/application of filters)
	RFfiltersTensor = tf.stack(RFfiltersList2, axis=0)

	return RFfiltersTensor, RFfiltersPropertiesList2
	

def generateRFfilter(resolutionProperties, isColourFilter, RFpropertiesInside, RFpropertiesOutside):

	# RF filter example (RFfilterTF):
	#
	# 0 0 0 0 0 0
	# 0 0 - - 0 0 
	# 0 - + + - 0
	# 0 0 - - 0 0
	# 0 0 0 0 0 0
	#
	# where "-" = -RFcolourOutside [R G B], "+" = +RFcolourInside [R G B], and "0" = [0, 0, 0]
	
	#generate ellipse on blank canvas
	#resolutionFactor, resolutionFactorReverse, imageSize = ATORtf_operations.getImageDimensionsR(resolutionProperties)
	blankArray = np.full((RFpropertiesInside.imageSize[1], RFpropertiesInside.imageSize[0], ATORtf_operations.rgbNumChannels), 0, np.uint8)	#rgb
	RFfilterTF = tf.convert_to_tensor(blankArray, dtype=tf.float32)

	
	RFfilterTF = ATORtf_RFproperties.drawRF(blankArray, RFfilterTF, RFpropertiesInside, RFpropertiesOutside, True)
	
	#print("RFfilterTF = ", RFfilterTF)

	if(ATORtf_operations.storeRFfiltersValuesAsFractions):
		RFfilterTF = tf.divide(RFfilterTF, ATORtf_operations.rgbMaxValue)
				
	if(not isColourFilter):
		RFfilterTF = tf.image.rgb_to_grayscale(RFfilterTF)
	
	#print("RFfilterTF.shape = ", RFfilterTF.shape)	
	#print("RFfilterTF = ", RFfilterTF)
		
	return RFfilterTF
		
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
	
