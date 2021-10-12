"""ATORtf_RF.py

# Author:
Richard Bruce Baxter - Copyright (c) 2021 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
See ATORtf.tf

# Usage:
See ATORtf.tf

# Description:
ATORtf RF (receptive field)

"""

import tensorflow as tf
import numpy as np
import copy

import ATORtf_ellipseProperties
import ATORtf_RFellipse
import ATORtf_RFtri
import ATORtf_RFfilter
import ATORtf_RFproperties
import ATORtf_operations


def normaliseRFComponentWRTparent(resolutionProperties, neuronComponent, RFpropertiesParent):
	neuronComponent.RFpropertiesNormalisedWRTparent = ATORtf_RFproperties.generateRFtransformedProperties(neuronComponent, RFpropertiesParent)
	if(resolutionProperties.debugSaveRFfiltersAndImageSegments):
		neuronComponent.RFfilterNormalisedWRTparent = ATORtf_RFfilter.transformRFfilterTF(neuronComponent.RFfilter, RFpropertiesParent)
		neuronComponent.RFImageNormalisedWRTparent = ATORtf_RFfilter.transformRFfilterTF(neuronComponent.RFImage, RFpropertiesParent)
		
#TODO: this needs to be upgraded with a robust method (preferably via RF neural net that contains associations between RFs at different resolutions)
def childRFoverlapsParentRF(neuronRFpropertiesNormalisedGlobal, lowerNeuronRFpropertiesNormalisedGlobal):
	if(neuronRFpropertiesNormalisedGlobal.RFtype == ATORtf_RFproperties.RFtypeEllipse):
		return ATORtf_ellipseProperties.centroidOverlapsEllipse(neuronRFpropertiesNormalisedGlobal, lowerNeuronRFpropertiesNormalisedGlobal)
	elif(neuronRFpropertiesNormalisedGlobal.RFtype == ATORtf_RFproperties.RFtypeTri):
		#CHECKTHIS is appropriate for tri; 
		return ATORtf_ellipseProperties.centroidOverlapsEllipse(neuronRFpropertiesNormalisedGlobal, lowerNeuronRFpropertiesNormalisedGlobal)
	
def normaliseGlobalRFproperties(RFproperties, resolutionFactor):
	#normalise RF respect to original image size
	if(RFproperties.RFtype == ATORtf_RFproperties.RFtypeEllipse):
		RFpropertiesNormalisedGlobal = ATORtf_RFellipse.normaliseGlobalEllipseProperties(RFproperties, resolutionFactor)
	elif(RFproperties.RFtype == ATORtf_RFproperties.RFtypeTri):
		RFpropertiesNormalisedGlobal = ATORtf_RFtri.normaliseGlobalTriProperties(RFproperties, resolutionFactor)
	return RFpropertiesNormalisedGlobal
			
def normaliseLocalRFproperties(RFproperties):
	#normalise ellipse respect to major/minor ellipticity axis orientation (WRT self)
	if(RFproperties.RFtype == ATORtf_RFproperties.RFtypeEllipse):
		ATORtf_RFellipse.normaliseLocalEllipseProperties(RFproperties)
	elif(RFproperties.RFtype == ATORtf_RFproperties.RFtypeTri):
		#CHECKTHIS is appropriate for tri; (ellipseNormalisedAxesLength, ellipseNormalisedAxesLength/q) might create equilateral triangle
		ATORtf_RFtri.normaliseLocalTriProperties(RFproperties)
		
def generateRFfilters(resolutionProperties, generateRFfiltersEllipse, generateRFfiltersTri):

	RFfiltersList = []
	RFfiltersPropertiesList = []
	
	if(generateRFfiltersEllipse):
		ATORtf_RFellipse.generateRFfiltersEllipse(resolutionProperties, RFfiltersList, RFfiltersPropertiesList)
	if(generateRFfiltersTri):
		ATORtf_RFtri.generateRFfiltersTri(resolutionProperties, RFfiltersList, RFfiltersPropertiesList)
	
	return RFfiltersList, RFfiltersPropertiesList

def applyRFfilters(resolutionProperties, inputImageSegments, RFfiltersTensor, numberOfDimensions, RFfiltersPropertiesList2):
	
	#perform convolution for each filter size;
	resolutionFactor, imageSize, axesLengthMax, filterRadius, filterSize = ATORtf_RFfilter.getFilterDimensions(resolutionProperties)
		
	#filterApplicationResultList = []
	RFpropertiesList = []

	#print("RFfiltersPropertiesList2[0].isColourFilter = ", RFfiltersPropertiesList2[0].isColourFilter)
	#print("inputImageSegments.shape = ", inputImageSegments.shape)
	#print("RFfiltersTensor.shape = ", RFfiltersTensor.shape)
		
	#inputImageSegments dim: num inputImageSegments, x, y, c
	#RFfiltersTensor dim: num RFfilters, x, y, c
	#print("inputImageSegments.shape = ", inputImageSegments.shape)
	#print("RFfiltersTensor.shape = ", RFfiltersTensor.shape)
	
	inputImageSegmentsPixelsFlattened = tf.reshape(inputImageSegments, [inputImageSegments.shape[0], inputImageSegments.shape[1]*inputImageSegments.shape[2]*inputImageSegments.shape[3]])
	RFfiltersTensorPixelsFlattened = tf.reshape(RFfiltersTensor, [RFfiltersTensor.shape[0], RFfiltersTensor.shape[1]*RFfiltersTensor.shape[2]*RFfiltersTensor.shape[3]])
	
	filterApplicationResult = tf.matmul(inputImageSegmentsPixelsFlattened, tf.transpose(RFfiltersTensorPixelsFlattened))	#dim: num inputImageSegments, num RFfilters
	filterApplicationResult	= tf.reshape(filterApplicationResult, [filterApplicationResult.shape[0]*filterApplicationResult.shape[1]]) #flatten	#dim: num inputImageSegments * num RFfilters
	#filterApplicationResultList.append(filterApplicationResult)
	
	isColourFilter = RFfiltersPropertiesList2[0].isColourFilter
	numberOfDimensions = RFfiltersPropertiesList2[0].numberOfDimensions
	RFtype = RFfiltersPropertiesList2[0].RFtype
	filterApplicationResultThreshold = ATORtf_RFfilter.calculateFilterApplicationResultThreshold(filterApplicationResult, ATORtf_RFfilter.minimumFilterRequirement, filterSize, isColourFilter, numberOfDimensions, RFtype)
	filterApplicationResultThresholdIndices = tf.where(filterApplicationResultThreshold)	#returns [n, 1] tensor
	filterApplicationResultThresholdIndices = tf.squeeze(filterApplicationResultThresholdIndices, axis=1)
	
	#print("filterApplicationResult = ", filterApplicationResult)
	#print("filterApplicationResult.shape = ", filterApplicationResult.shape)
	#print("filterApplicationResultThreshold.shape = ", filterApplicationResultThreshold.shape)
	#print("filterApplicationResultThresholdIndices.shape = ", filterApplicationResultThresholdIndices.shape)
	
	if(not ATORtf_operations.isTensorEmpty(filterApplicationResultThresholdIndices)):

		filterApplicationResultThresholded = tf.gather(filterApplicationResult, filterApplicationResultThresholdIndices)
		
		#print("filterApplicationResultThresholded = ", filterApplicationResultThresholded)
		#print("filterApplicationResultThresholded.shape = ", filterApplicationResultThresholded.shape)
		
		filterApplicationResultThresholdIndicesNP = filterApplicationResultThresholdIndices.numpy()	#verify 1D
		filterApplicationResultThresholdIndicesList = filterApplicationResultThresholdIndicesNP.tolist()
		filterApplicationResultThresholdedNP = filterApplicationResultThresholded.numpy()	#verify 1D
		filterApplicationResultThresholdedList = filterApplicationResultThresholdedNP.tolist()

		for RFthresholdedListIndex, RFlistIndex in enumerate(filterApplicationResultThresholdIndicesList):
			#RFlistIndex = imageSegmentIndex*len(RFfiltersPropertiesList2) +  RFfiltersPropertiesList2Index

			imageSegmentIndex, RFfilterIndex = divmod(RFlistIndex, len(RFfiltersTensor))
			centerCoordinates1, centerCoordinates2 = divmod(imageSegmentIndex, imageSize[1])
			centerCoordinates = (centerCoordinates1, centerCoordinates2)

			#print("adding RFproperties")
			RFImage = inputImageSegments[imageSegmentIndex]
			RFfiltersProperties = RFfiltersPropertiesList2[RFfilterIndex]
			RFproperties = copy.deepcopy(RFfiltersProperties)
			RFproperties.centerCoordinates = centerCoordinates
			RFproperties.filterIndex = RFfilterIndex
			RFproperties.imageSegmentIndex = imageSegmentIndex
			RFpropertiesList.append(RFproperties)		

		#inefficient:
		#imageSegmentIndex = 0
		#for centerCoordinates1 in range(0, imageSize[0], ellipseCenterCoordinatesResolution):
		#	for centerCoordinates2 in range(0, imageSize[1], ellipseCenterCoordinatesResolution):
		#		centerCoordinates = (centerCoordinates1, centerCoordinates2)
		#		allFilterCoordinatesWithinImageResult, imageSegmentStart, imageSegmentEnd = ATORtf_RFfilter.allFilterCoordinatesWithinImage(centerCoordinates, filterRadius, imageSize)
		#		if(allFilterCoordinatesWithinImageResult):
		#			for RFfilterIndex, RFfiltersProperties in enumerate(RFfiltersPropertiesList2):
		#				RFproperties = copy.deepcopy(RFfiltersProperties)
		#				RFproperties.centerCoordinates = centerCoordinates
		#				RFproperties.filterIndex = RFfilterIndex
		#				RFproperties.imageSegmentIndex = imageSegmentIndex
		#				RFpropertiesList.append(RFproperties)
		#			imageSegmentIndex = imageSegmentIndex+1

		#verify these match:
		#print("len(filterApplicationResultThresholdedList) = ", len(filterApplicationResultThresholdedList))
		#print("len(filterApplicationResultThresholdIndicesList) = ", len(filterApplicationResultThresholdIndicesList))
		#print("len(RFpropertiesList) = ", len(RFpropertiesList))
	else:
		filterApplicationResultThresholdIndicesList = []
		filterApplicationResultThresholdedList = []
		
	return filterApplicationResultThresholdIndicesList, filterApplicationResultThresholdedList, RFpropertiesList	#filterApplicationResultList
	
	
