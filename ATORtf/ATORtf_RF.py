# -*- coding: utf-8 -*-
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
import ATORtf_RFFilter
import ATORtf_RFProperties
import ATORtf_operations


def normaliseRFComponentWRTparent(resolutionProperties, neuronComponent, RFPropertiesParent):
	neuronComponent.RFPropertiesNormalisedWRTparent = ATORtf_RFProperties.generateRFTransformedProperties(neuronComponent, RFPropertiesParent)
	if(resolutionProperties.debugSaveRFFiltersAndImageSegments):
		neuronComponent.RFFilterNormalisedWRTparent = ATORtf_RFFilter.transformRFFilterTF(neuronComponent.RFFilter, RFPropertiesParent)
		neuronComponent.RFImageNormalisedWRTparent = ATORtf_RFFilter.transformRFFilterTF(neuronComponent.RFImage, RFPropertiesParent)
		
def childRFoverlapsParentRF(neuronRFPropertiesNormalisedGlobal, lowerNeuronRFPropertiesNormalisedGlobal):
	if(neuronRFPropertiesNormalisedGlobal.RFtype == ATORtf_RFProperties.RFtypeEllipse):
		return ATORtf_ellipseProperties.centroidOverlapsEllipse(neuronRFPropertiesNormalisedGlobal, lowerNeuronRFPropertiesNormalisedGlobal)
	elif(neuronRFPropertiesNormalisedGlobal.RFtype == ATORtf_RFProperties.RFtypeTri):
		#CHECKTHIS is appropriate for tri; 
		return unknown	
	
def normaliseGlobalRFProperties(RFProperties, resolutionFactor):
	#normalise RF respect to original image size
	if(RFProperties.RFtype == ATORtf_RFProperties.RFtypeEllipse):
		RFPropertiesNormalisedGlobal = ATORtf_RFellipse.normaliseGlobalEllipseProperties(RFProperties, resolutionFactor)
	elif(RFProperties.RFtype == ATORtf_RFProperties.RFtypeTri):
		#CHECKTHIS is appropriate for tri; 
		RFPropertiesNormalisedGlobal = ATORtf_RFellipse.normaliseGlobalEllipseProperties(RFProperties, resolutionFactor)
	return RFPropertiesNormalisedGlobal
			
def normaliseLocalRFProperties(RFProperties):
	#normalise ellipse respect to major/minor ellipticity axis orientation (WRT self)
	if(RFProperties.RFtype == ATORtf_RFProperties.RFtypeEllipse):
		ATORtf_RFellipse.normaliseLocalEllipseProperties(RFProperties)
	elif(RFProperties.RFtype == ATORtf_RFProperties.RFtypeTri):
		#CHECKTHIS is appropriate for tri; (ellipseNormalisedAxesLength, ellipseNormalisedAxesLength/q) might create equilateral triangle
		ATORtf_RFtri.normaliseLocalTriProperties(RFProperties)
		
def generateRFFilters(resolutionProperties, generateRFFiltersEllipse, generateRFFiltersTri):

	RFFiltersList = []
	RFFiltersPropertiesList = []
	
	if(generateRFFiltersEllipse):
		ATORtf_RFellipse.generateRFFiltersEllipse(resolutionProperties, RFFiltersList, RFFiltersPropertiesList)
	if(generateRFFiltersTri):
		ATORtf_RFtri.generateRFFiltersTri(resolutionProperties, RFFiltersList, RFFiltersPropertiesList)
	
	return RFFiltersList, RFFiltersPropertiesList

def applyRFFilters(resolutionProperties, inputImageSegments, RFFiltersTensor, numberOfDimensions, RFFiltersPropertiesList2):
	
	#perform convolution for each filter size;
	resolutionFactor, imageSize, axesLengthMax, filterRadius, filterSize = ATORtf_RFFilter.getFilterDimensions(resolutionProperties)
		
	#filterApplicationResultList = []
	RFPropertiesList = []

	#print("RFFiltersPropertiesList2[0].isColourFilter = ", RFFiltersPropertiesList2[0].isColourFilter)
	#print("inputImageSegments.shape = ", inputImageSegments.shape)
	#print("RFFiltersTensor.shape = ", RFFiltersTensor.shape)
		
	#inputImageSegments dim: num inputImageSegments, x, y, c
	#RFFiltersTensor dim: num RFFilters, x, y, c
	#print("inputImageSegments.shape = ", inputImageSegments.shape)
	#print("RFFiltersTensor.shape = ", RFFiltersTensor.shape)
	
	inputImageSegmentsPixelsFlattened = tf.reshape(inputImageSegments, [inputImageSegments.shape[0], inputImageSegments.shape[1]*inputImageSegments.shape[2]*inputImageSegments.shape[3]])
	RFFiltersTensorPixelsFlattened = tf.reshape(RFFiltersTensor, [RFFiltersTensor.shape[0], RFFiltersTensor.shape[1]*RFFiltersTensor.shape[2]*RFFiltersTensor.shape[3]])
	
	filterApplicationResult = tf.matmul(inputImageSegmentsPixelsFlattened, tf.transpose(RFFiltersTensorPixelsFlattened))	#dim: num inputImageSegments, num RFFilters
	filterApplicationResult	= tf.reshape(filterApplicationResult, [filterApplicationResult.shape[0]*filterApplicationResult.shape[1]]) #flatten	#dim: num inputImageSegments * num RFFilters
	#filterApplicationResultList.append(filterApplicationResult)
	
	isColourFilter = RFFiltersPropertiesList2[0].isColourFilter
	numberOfDimensions = RFFiltersPropertiesList2[0].numberOfDimensions
	RFtype = RFFiltersPropertiesList2[0].RFtype
	filterApplicationResultThreshold = ATORtf_RFFilter.calculateFilterApplicationResultThreshold(filterApplicationResult, ATORtf_RFFilter.minimumFilterRequirement, filterSize, isColourFilter, numberOfDimensions, RFtype)
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
			#RFlistIndex = imageSegmentIndex*len(RFFiltersPropertiesList2) +  RFFiltersPropertiesList2Index

			imageSegmentIndex, RFFilterIndex = divmod(RFlistIndex, len(RFFiltersTensor))
			centerCoordinates1, centerCoordinates2 = divmod(imageSegmentIndex, imageSize[1])
			centerCoordinates = (centerCoordinates1, centerCoordinates2)

			#print("adding RFProperties")
			RFImage = inputImageSegments[imageSegmentIndex]
			RFFiltersProperties = RFFiltersPropertiesList2[RFFilterIndex]
			RFProperties = copy.deepcopy(RFFiltersProperties)
			RFProperties.centerCoordinates = centerCoordinates
			RFProperties.filterIndex = RFFilterIndex
			RFProperties.imageSegmentIndex = imageSegmentIndex
			RFPropertiesList.append(RFProperties)		

		#inefficient:
		#imageSegmentIndex = 0
		#for centerCoordinates1 in range(0, imageSize[0], ellipseCenterCoordinatesResolution):
		#	for centerCoordinates2 in range(0, imageSize[1], ellipseCenterCoordinatesResolution):
		#		centerCoordinates = (centerCoordinates1, centerCoordinates2)
		#		allFilterCoordinatesWithinImageResult, imageSegmentStart, imageSegmentEnd = ATORtf_RFFilter.allFilterCoordinatesWithinImage(centerCoordinates, filterRadius, imageSize)
		#		if(allFilterCoordinatesWithinImageResult):
		#			for RFFilterIndex, RFFiltersProperties in enumerate(RFFiltersPropertiesList2):
		#				RFProperties = copy.deepcopy(RFFiltersProperties)
		#				RFProperties.centerCoordinates = centerCoordinates
		#				RFProperties.filterIndex = RFFilterIndex
		#				RFProperties.imageSegmentIndex = imageSegmentIndex
		#				RFPropertiesList.append(RFProperties)
		#			imageSegmentIndex = imageSegmentIndex+1

		#verify these match:
		#print("len(filterApplicationResultThresholdedList) = ", len(filterApplicationResultThresholdedList))
		#print("len(filterApplicationResultThresholdIndicesList) = ", len(filterApplicationResultThresholdIndicesList))
		#print("len(RFPropertiesList) = ", len(RFPropertiesList))
	else:
		filterApplicationResultThresholdIndicesList = []
		filterApplicationResultThresholdedList = []
		
	return filterApplicationResultThresholdIndicesList, filterApplicationResultThresholdedList, RFPropertiesList	#filterApplicationResultList
	
	
