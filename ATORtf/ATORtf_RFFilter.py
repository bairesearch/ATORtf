# -*- coding: utf-8 -*-
"""ATORtf_RFFilter.py

# Author:
Richard Bruce Baxter - Copyright (c) 2021 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
See ATORtf.tf

# Usage:
See ATORtf.tf

# Description:
ATORtf RF Filter - RF Filter transformations (pixel space)

"""

import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import copy

import ATORtf_RFProperties
import ATORtf_RFellipse
import ATORtf_RFtri
import ATORtf_operations

minimumFilterRequirement = 1.5	#CHECKTHIS: calibrate	#matched values fraction	#theoretical value: 0.95
	
#if(debugSaveRFFiltersAndImageSegments):
RFFilterImageTransformFillValue = 0.0


def calculateFilterApplicationResultThreshold(filterApplicationResult, minimumFilterRequirement, filterSize, isColourFilter, numberOfDimensions, RFtype):
	
	minimumFilterRequirementLocal = minimumFilterRequirement*calculateFilterPixels(filterSize, numberOfDimensions, RFtype)
	
	#if(isColourFilter):
	#	minimumFilterRequirementLocal = minimumFilterRequirementLocal*ATORtf_operations.rgbNumChannels*ATORtf_operations.rgbNumChannels	#CHECKTHIS	#not required as assume filter colours will be normalised to the maximum value of a single rgb channel? 
	if(not ATORtf_operations.storeRFFiltersValuesAsFractions):
		minimumFilterRequirementLocal = minimumFilterRequirementLocal*(ATORtf_operations.rgbMaxValue*ATORtf_operations.rgbMaxValue)	#rgbMaxValue of both imageSegment and RFFilter 		

	print("minimumFilterRequirementLocal = ", minimumFilterRequirementLocal)
	print("tf.math.reduce_max(filterApplicationResult) = ", tf.math.reduce_max(filterApplicationResult))
	
	filterApplicationResultThreshold = tf.greater(filterApplicationResult, minimumFilterRequirementLocal)	
	return filterApplicationResultThreshold

def calculateFilterPixels(filterSize, numberOfDimensions, RFtype):
	if(RFtype == ATORtf_RFProperties.RFtypeEllipse):
		return ATORtf_RFellipse.calculateFilterPixels(filterSize, numberOfDimensions)
	elif(RFtype == ATORtf_RFProperties.RFtypeTri):
		print("calculateFilterPixels error: RFtype == ATORtf_RFProperties.RFtypeTri not yet coded")	#count number of pixels on for each point
		return ATORtf_RFtri.calculateFilterPixels(filterSize, numberOfDimensions)

def normaliseRFFilter(RFFilter, RFProperties):
	#normalise ellipse respect to major/minor ellipticity axis orientation (WRT self)
	RFFilterNormalised = transformRFFilterTF(RFFilter, RFProperties) 
	#RFFilterNormalised = RFFilter
	return RFFilterNormalised
	
def transformRFFilterTF(RFFilter, RFPropertiesParent):
	if(RFPropertiesParent.numberOfDimensions == 2):
		centerCoordinates = [-RFPropertiesParent.centerCoordinates[0], -RFPropertiesParent.centerCoordinates[1]]
		axesLength = 1.0/RFPropertiesParent.axesLength[0]	#[1.0/RFPropertiesParent.axesLength[0], 1.0/RFPropertiesParent.axesLength[1]]
		angle = -RFPropertiesParent.angle
		RFFilterTransformed = transformRFFilterTF2D(RFFilter, centerCoordinates, axesLength, angle)
	elif(RFPropertiesParent.numberOfDimensions == 3):
		print("error transformRFFilterWRTparentTF: RFPropertiesParent.numberOfDimensions == 3 not yet coded")
		quit()
	return RFFilterTransformed
	
def transformRFFilterTF2D(RFFilter, centerCoordinates, axesLength, angle):
	#CHECKTHIS: 2D code only;
	#RFFilterTransformed = tf.expand_dims(RFFilterTransformed, 0)	#add extra dimension for num_images
	RFFilterTransformed = RFFilter
	angleRadians =  ATORtf_operations.convertDegreesToRadians(angle)
	RFFilterTransformed = tfa.image.rotate(RFFilterTransformed, angleRadians, fill_value=RFFilterImageTransformFillValue)		#https://www.tensorflow.org/addons/api_docs/python/tfa/image/rotate
	centerCoordinatesList = [float(x) for x in list(centerCoordinates)]
	RFFilterTransformed = tfa.image.translate(RFFilterTransformed, centerCoordinatesList, fill_value=RFFilterImageTransformFillValue)		#fill_value=RFFilterImageTransformFillValue	#https://www.tensorflow.org/addons/api_docs/python/tfa/image/translate
	#print("axesLength = ", axesLength)	
	#print("RFFilterTransformed.shape = ", RFFilterTransformed.shape)	
	RFFilterTransformed = imageScale(RFFilterTransformed, axesLength)	#https://www.tensorflow.org/api_docs/python/tf/image/resize
	#print("RFFilterTransformed.shape = ", RFFilterTransformed.shape)	
	RFFilterTransformed = tf.squeeze(RFFilterTransformed)
	return RFFilterTransformed

def imageScale(img, scaleFactor):
	a0 = scaleFactor
	b1 = scaleFactor
	scaleMatrix = [a0, 0.0, 0.0, 0.0, b1, 0.0, 0.0, 0.0]
	transformedImage = tfa.image.transform(img, scaleMatrix)
	return transformedImage

		
def rotateRFFilterTF(RFFilter, RFProperties):
	return rotateRFFilterTF(-RFProperties.angle)
def rotateRFFilterTF(RFFilter, angle):
	RFFilter = tf.expand_dims(RFFilter, 0)	#add extra dimension for num_images
	return RFFilterNormalised
		

#CHECKTHIS: upgrade code to support ATORtf_RFtri
def getFilterDimensions(resolutionProperties):
	return ATORtf_RFellipse.getFilterDimensions(resolutionProperties)

#CHECKTHIS: upgrade code to support ATORtf_RFtri
def allFilterCoordinatesWithinImage(centerCoordinates, filterRadius, imageSize):
	imageSegmentStart = (centerCoordinates[0]-filterRadius, centerCoordinates[1]-filterRadius)
	imageSegmentEnd = (centerCoordinates[0]+filterRadius, centerCoordinates[1]+filterRadius)
	if(imageSegmentStart[0]>=0 and imageSegmentStart[1]>=0 and imageSegmentEnd[0]<imageSize[0] and imageSegmentEnd[1]<imageSize[1]):
		result = True
	else:
		result = False
		#create artificial image segment (will be discarded during image filter application)
		imageSegmentStart = (0, 0)
		imageSegmentEnd = (filterRadius*2, filterRadius*2)
	return result, imageSegmentStart, imageSegmentEnd

