"""ATORtf_RFfilter.py

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

import ATORtf_RFproperties
import ATORtf_RFellipse
import ATORtf_RFtri
import ATORtf_operations

minimumFilterRequirement = 1.5	#CHECKTHIS: calibrate	#matched values fraction	#theoretical value: 0.95
	
#if(debugSaveRFfiltersAndImageSegments):
RFfilterImageTransformFillValue = 0.0


def calculateFilterApplicationResultThreshold(filterApplicationResult, minimumFilterRequirement, filterSize, isColourFilter, numberOfDimensions, RFtype):
	
	minimumFilterRequirementLocal = minimumFilterRequirement*calculateFilterPixels(filterSize, numberOfDimensions, RFtype)
	
	#if(isColourFilter):
	#	minimumFilterRequirementLocal = minimumFilterRequirementLocal*ATORtf_operations.rgbNumChannels*ATORtf_operations.rgbNumChannels	#CHECKTHIS	#not required as assume filter colours will be normalised to the maximum value of a single rgb channel? 
	if(not ATORtf_operations.storeRFfiltersValuesAsFractions):
		minimumFilterRequirementLocal = minimumFilterRequirementLocal*(ATORtf_operations.rgbMaxValue*ATORtf_operations.rgbMaxValue)	#rgbMaxValue of both imageSegment and RFfilter 		

	print("minimumFilterRequirementLocal = ", minimumFilterRequirementLocal)
	print("tf.math.reduce_max(filterApplicationResult) = ", tf.math.reduce_max(filterApplicationResult))
	
	filterApplicationResultThreshold = tf.greater(filterApplicationResult, minimumFilterRequirementLocal)	
	return filterApplicationResultThreshold

def calculateFilterPixels(filterSize, numberOfDimensions, RFtype):
	if(RFtype == ATORtf_RFproperties.RFtypeEllipse):
		return ATORtf_RFellipse.calculateFilterPixels(filterSize, numberOfDimensions)
	elif(RFtype == ATORtf_RFproperties.RFtypeTri):
		return ATORtf_RFtri.calculateFilterPixels(filterSize, numberOfDimensions)

def normaliseRFfilter(RFfilter, RFproperties):
	#normalise ellipse respect to major/minor ellipticity axis orientation (WRT self)
	RFfilterNormalised = transformRFfilterTF(RFfilter, RFproperties) 
	#RFfilterNormalised = RFfilter
	return RFfilterNormalised
	
def transformRFfilterTF(RFfilter, RFpropertiesParent):
	if(RFpropertiesParent.numberOfDimensions == 2):
		centerCoordinates = [-RFpropertiesParent.centerCoordinates[0], -RFpropertiesParent.centerCoordinates[1]]
		axesLength = 1.0/RFpropertiesParent.axesLength[0]	#[1.0/RFpropertiesParent.axesLength[0], 1.0/RFpropertiesParent.axesLength[1]]
		angle = -RFpropertiesParent.angle
		RFfilterTransformed = transformRFfilterTF2D(RFfilter, centerCoordinates, axesLength, angle)
	elif(RFpropertiesParent.numberOfDimensions == 3):
		print("error transformRFfilterWRTparentTF: RFpropertiesParent.numberOfDimensions == 3 not yet coded")
		quit()
	return RFfilterTransformed
	
def transformRFfilterTF2D(RFfilter, centerCoordinates, axesLength, angle):
	#CHECKTHIS: 2D code only;
	#RFfilterTransformed = tf.expand_dims(RFfilterTransformed, 0)	#add extra dimension for num_images
	RFfilterTransformed = RFfilter
	angleRadians =  ATORtf_operations.convertDegreesToRadians(angle)
	RFfilterTransformed = tfa.image.rotate(RFfilterTransformed, angleRadians, fill_value=RFfilterImageTransformFillValue)		#https://www.tensorflow.org/addons/api_docs/python/tfa/image/rotate
	centerCoordinatesList = [float(x) for x in list(centerCoordinates)]
	RFfilterTransformed = tfa.image.translate(RFfilterTransformed, centerCoordinatesList, fill_value=RFfilterImageTransformFillValue)		#fill_value=RFfilterImageTransformFillValue	#https://www.tensorflow.org/addons/api_docs/python/tfa/image/translate
	#print("axesLength = ", axesLength)	
	#print("RFfilterTransformed.shape = ", RFfilterTransformed.shape)	
	RFfilterTransformed = imageScale(RFfilterTransformed, axesLength)	#https://www.tensorflow.org/api_docs/python/tf/image/resize
	#print("RFfilterTransformed.shape = ", RFfilterTransformed.shape)	
	RFfilterTransformed = tf.squeeze(RFfilterTransformed)
	return RFfilterTransformed

def imageScale(img, scaleFactor):
	a0 = scaleFactor
	b1 = scaleFactor
	scaleMatrix = [a0, 0.0, 0.0, 0.0, b1, 0.0, 0.0, 0.0]
	transformedImage = tfa.image.transform(img, scaleMatrix)
	return transformedImage

		
def rotateRFfilterTF(RFfilter, RFproperties):
	return rotateRFfilterTF(-RFproperties.angle)
def rotateRFfilterTF(RFfilter, angle):
	RFfilter = tf.expand_dims(RFfilter, 0)	#add extra dimension for num_images
	return RFfilterNormalised
		

def getFilterDimensions(resolutionProperties):
	return ATORtf_RFproperties.getFilterDimensions(resolutionProperties)
		

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

