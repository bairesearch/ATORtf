# -*- coding: utf-8 -*-
"""ATORtf.py

# Author:
Richard Bruce Baxter - Copyright (c) 2021 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
conda create -n ATORtf python=3.9
source activate ATORtf
pip install tensorflow-gpu==2.6
conda install --file condaRequirements.txt
	where condaRequirements.txt contains;
		numpy
		click
		pillow
pip install tensorflow-addons
pip install opencv-python opencv-contrib-python

# Usage:
source activate ATORtf
python ATORtf.py images/leaf1.png

# Description:
ATORtf is a hardware accelerated version of BAI ATOR (Axis Transformation Object Recognition) for TensorFlow.

ATORtf supports ellipsoid features, and normalises them with respect to their major/minor ellipticity axis orientation. 

There are a number of advantages of using ellipsoid features over point features;
* the number of feature sets/normalised snapshots required is significantly reduced
* scene component structure can be maintained (as detected component ellipses can be represented in a hierarchical graph structure)

Ellipse features/components are detected based on simulated artificial receptive fields; RF (on/off, off/on).

ATORtf is compatible with point (corner/centroid) features of the ATOR specification; 
https://www.wipo.int/patentscope/search/en/WO2011088497

# Future:
Requires upgrading to support 3D receptive field detection (ellipses and ellipsoids in 3D space)
Requires upgrading to support point feature receptive field detection (in tri sets)

"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
import numpy as np
import click
import cv2
import copy
import sys

import ATORtf_detectEllipses
import ATORtf_RF
import ATORtf_RFProperties
import ATORtf_RFFilter
import ATORtf_operations

np.set_printoptions(threshold=sys.maxsize)

generateRFFiltersEllipse = True
generateRFFiltersTri = False

debugLowIterations = False
debugVerbose = True
debugSaveRFFiltersAndImageSegments = True	#only RF properties are required to be saved by ATOR algorithm (not image RF pixel data)

		
resolutionIndexFirst = 0	#mandatory
numberOfResolutions = 5	#x; lowest res sample: 1/(2^x)


ellipseCenterCoordinatesResolution = 1	#pixels (at resolution r)

imageSizeBase = (256, 256)	#maximum image resolution used by ATORtf algorithm	#CHECKTHIS


class ATORneuronClass():
	def __init__(self, resolutionIndex, resolutionFactor, RFProperties, RFFilter, RFImage):
		self.resolutionIndex = resolutionIndex
		self.RFProperties = RFProperties
		self.RFPropertiesNormalised = ATORtf_RF.normaliseLocalRFProperties(RFProperties)
		self.RFPropertiesNormalisedWRTparent = None
		
		self.RFPropertiesNormalisedGlobal = ATORtf_RF.normaliseGlobalRFProperties(RFProperties, resolutionFactor)
		if(debugSaveRFFiltersAndImageSegments):
			self.RFFilter = RFFilter
			self.RFFilterNormalised = ATORtf_RFFilter.normaliseRFFilter(RFFilter, RFProperties)
			self.RFFilterNormalisedWRTparent = None
			self.RFImage = RFImage
			self.RFImageNormalised = ATORtf_RFFilter.normaliseRFFilter(RFImage, RFProperties)
			self.RFImageNormalisedWRTparent = None
		self.neuronComponents = []
		self.neuronComponentsWeightsList = []
		
def createRFhierarchyAccelerated(inputimagefilename):
	
	inputImage = cv2.imread(inputimagefilename)	#FUTURE: support 3D datasets
	
	#normalise image size
	inputImage = cv2.resize(inputImage, imageSizeBase)
	inputImageRGB = cv2.cvtColor(inputImage, cv2.COLOR_BGR2RGB)
	inputImageGray = cv2.cvtColor(inputImageRGB, cv2.COLOR_RGB2GRAY)
	
	inputimagefilenameTF = tf.convert_to_tensor(inputimagefilename, dtype=tf.string)
	inputimageStringTF = tf.io.read_file(inputimagefilenameTF)
	inputImageRGBTF = tf.io.decode_image(inputimageStringTF, channels=3)
	inputImageGrayTF = tf.image.rgb_to_grayscale(inputImageRGBTF)
	#print(inputImageRGBTF.shape)
	#print(inputImageGrayTF.shape)
	#inputImageRGBTF = tf.convert_to_tensor(inputImageRGB, dtype=tf.float32)
	#inputImageGrayTF = tf.convert_to_tensor(inputImageGray, dtype=tf.float32)
	
	inputImageHeight, inputImageWidth, inputImageChannels = inputImage.shape
	print("inputImageHeight = ", inputImageHeight, "inputImageWidth = ", inputImageWidth, ", inputImageChannels = ", inputImageChannels)
	blankArray = np.full((inputImageHeight, inputImageWidth, 3), 255, np.uint8)
	outputImage = blankArray
	
	ATORneuronListAllLayers = []
			
	inputImageRGBSegmentsAllRes = []	#stores subsets of input image at different resolutions, centreCoordinates, and size
	inputImageGraySegmentsAllRes = []
	RFFiltersListAllRes = []	#stores receptive field tensorflow objects (used for hardware accelerated filter detection)
	RFFiltersPropertiesListAllRes = []	#stores receptive field ellipse properties (position, size, rotation, colour etc)
	
	#generateRFFilters:
	if(debugLowIterations):
		resolutionIndexMax = 1
	else:
		resolutionIndexMax = numberOfResolutions
	
	resolutionProperties = ATORtf_operations.RFresolutionProperties(-1, resolutionIndexFirst, numberOfResolutions, imageSizeBase, debugVerbose, debugSaveRFFiltersAndImageSegments)

	for resolutionIndex in range(resolutionIndexFirst, resolutionIndexMax):
		resolutionProperties.resolutionIndex = resolutionIndex
		resolutionFactor, resolutionFactorReverse, imageSize = ATORtf_operations.getImageDimensionsR(resolutionProperties)
		inputImageRGBTF = tf.image.resize(inputImageRGBTF, imageSize)
		inputImageGrayTF = tf.image.resize(inputImageGrayTF, imageSize)
		inputImageRGBSegments, inputImageGraySegments = generateImageSegments(resolutionProperties, inputImageRGBTF, inputImageGrayTF)
		inputImageRGBSegmentsAllRes.append(inputImageRGBSegments)
		inputImageGraySegmentsAllRes.append(inputImageGraySegments)
		
	for resolutionIndex in range(resolutionIndexFirst, resolutionIndexMax):
		resolutionProperties.resolutionIndex = resolutionIndex
		RFFiltersList, RFFiltersPropertiesList = ATORtf_RF.generateRFFilters(resolutionProperties, generateRFFiltersEllipse, generateRFFiltersTri)
		RFFiltersListAllRes.append(RFFiltersList)
		RFFiltersPropertiesListAllRes.append(RFFiltersPropertiesList)
		
	#applyRFFilters:
	for resolutionIndex in range(resolutionIndexFirst, resolutionIndexMax):
		resolutionProperties.resolutionIndex = resolutionIndex
		print("resolutionIndex = ", resolutionIndex)
		inputImageRGBSegments = inputImageRGBSegmentsAllRes[resolutionIndex]
		inputImageGraySegments = inputImageGraySegmentsAllRes[resolutionIndex]
		RFFiltersList = RFFiltersListAllRes[resolutionIndex]
		RFFiltersPropertiesList = RFFiltersPropertiesListAllRes[resolutionIndex]
		applyRFFiltersList(resolutionProperties, inputImageRGBSegments, inputImageGraySegments, RFFiltersList, RFFiltersPropertiesList, ATORneuronListAllLayers)

def generateImageSegments(resolutionProperties, inputImageRGBTF, inputImageGrayTF):
	inputImageRGBSegmentsList = []
	inputImageGraySegmentsList = []
	
	resolutionFactor, imageSize, axesLengthMax, filterRadius, filterSize = ATORtf_RFFilter.getFilterDimensions(resolutionProperties)
	
	if(debugVerbose):
		print("")
		print("resolutionIndex = ", resolutionProperties.resolutionIndex)
		print("resolutionFactor = ", resolutionFactor)
		print("imageSize = ", imageSize)
		print("filterRadius = ", filterRadius)
		print("axesLengthMax = ", axesLengthMax)
		print("filterSize = ", filterSize)
			
	#print("inputImageRGBTF.shape = ", inputImageRGBTF.shape)
	#print("inputImageGrayTF.shape = ", inputImageGrayTF.shape)
	
	imageSegmentIndex = 0		
	for centerCoordinates1 in range(0, imageSize[0], ellipseCenterCoordinatesResolution):
		for centerCoordinates2 in range(0, imageSize[1], ellipseCenterCoordinatesResolution):
			centerCoordinates = (centerCoordinates1, centerCoordinates2)
			#print("imageSize = ", imageSize)
			#print("filterRadius = ", filterRadius)
			#print("filterSize = ", filterSize)
			#print("centerCoordinates = ", centerCoordinates)
			allFilterCoordinatesWithinImageResult, imageSegmentStart, imageSegmentEnd = ATORtf_RFFilter.allFilterCoordinatesWithinImage(centerCoordinates, filterRadius, imageSize)
			#if(not allFilterCoordinatesWithinImageResult): image segments and their applied filters will be discarded, but artificial (beyond bounds) image segments are still added to inputImageSegments tensor for algorithm uniformity
			inputImageRGBSegment = inputImageRGBTF[imageSegmentStart[0]:imageSegmentEnd[0], imageSegmentStart[1]:imageSegmentEnd[1], :]
			inputImageGraySegment = inputImageGrayTF[imageSegmentStart[0]:imageSegmentEnd[0], imageSegmentStart[1]:imageSegmentEnd[1], :]
			if(ATORtf_operations.storeRFFiltersValuesAsFractions):
				inputImageRGBSegment = tf.divide(inputImageRGBSegment, ATORtf_operations.rgbMaxValue)
				inputImageGraySegment = tf.divide(inputImageGraySegment, ATORtf_operations.rgbMaxValue)
			inputImageRGBSegmentsList.append(inputImageRGBSegment)
			inputImageGraySegmentsList.append(inputImageGraySegment)
			imageSegmentIndex = imageSegmentIndex+1
			
	inputImageRGBSegments = tf.stack(inputImageRGBSegmentsList)
	inputImageGraySegments = tf.stack(inputImageGraySegmentsList)
	
	#print("inputImageRGBSegments.shape = ", inputImageRGBSegments.shape)
	#print("inputImageGraySegments.shape = ", inputImageGraySegments.shape)
			
	return inputImageRGBSegments, inputImageGraySegments
	
	
def applyRFFiltersList(resolutionProperties, inputImageRGBSegments, inputImageGraySegments, RFFiltersList, RFFiltersPropertiesList, ATORneuronListAllLayers):
	
	print("\tapplyRFFiltersList: resolutionIndex = ", resolutionProperties.resolutionIndex)
	
	ATORneuronList = []	#for resolutionIndex
	
	resolutionFactor, resolutionFactorReverse, imageSize = ATORtf_operations.getImageDimensionsR(resolutionProperties)
	resolutionFactor, imageSize, axesLengthMax, filterRadius, filterSize = ATORtf_RFFilter.getFilterDimensions(resolutionProperties)

	#print("imageSize = ", imageSize)
	#print("inputImageGrayTF.shape = ", inputImageGrayTF.shape)
	#print("inputImageRGBTF.shape = ", inputImageRGBTF.shape)
	#inputImageRGBTF = tf.image.resize(inputImageRGBTF, imageSize)
	#inputImageGrayTF = tf.image.resize(inputImageGrayTF, imageSize)
	#print("inputImageGrayTF.shape = ", inputImageGrayTF.shape)
	#print("inputImageRGBTF.shape = ", inputImageRGBTF.shape)
				
	for RFlistIndex1 in range(len(RFFiltersPropertiesList)):	#or RFFiltersList
	
		print("RFlistIndex1 = ", RFlistIndex1)
		RFFiltersTensor = RFFiltersList[RFlistIndex1]
		RFFiltersPropertiesList2 = RFFiltersPropertiesList[RFlistIndex1]
		isColourFilter = RFFiltersPropertiesList2[0].isColourFilter
		numberOfDimensions = RFFiltersPropertiesList2[0].numberOfDimensions
		if(isColourFilter):
			inputImageSegments = inputImageRGBSegments
		else:
			inputImageSegments = inputImageGraySegments

		filterApplicationResultThresholdIndicesList, filterApplicationResultThresholdedList, RFPropertiesList = ATORtf_RF.applyRFFilters(resolutionProperties, inputImageSegments, RFFiltersTensor, numberOfDimensions, RFFiltersPropertiesList2)
			
		print("ATORneuronList append: len(filterApplicationResultThresholdIndicesList) = ", len(filterApplicationResultThresholdIndicesList))	
		for RFthresholdedListIndex, RFlistIndex in enumerate(filterApplicationResultThresholdIndicesList):
				
			filterApplicationResult = filterApplicationResultThresholdedList[RFthresholdedListIndex]
			RFProperties = RFPropertiesList[RFthresholdedListIndex]
			#print("type(RFProperties) = ", type(RFProperties))
			RFFilter = None
			RFImage = None
			if(debugSaveRFFiltersAndImageSegments):
				RFFilter = RFFiltersTensor[RFProperties.filterIndex]
				RFImage = inputImageSegments[RFProperties.imageSegmentIndex]

			centerCoordinates = RFProperties.centerCoordinates
			allFilterCoordinatesWithinImageResult, imageSegmentStart, imageSegmentEnd = ATORtf_RFFilter.allFilterCoordinatesWithinImage(centerCoordinates, filterRadius, imageSize)
			if(allFilterCoordinatesWithinImageResult):
			
				#create child neuron:
				neuron = ATORneuronClass(resolutionProperties.resolutionIndex, resolutionFactor, RFProperties, RFFilter, RFImage)
				ATORneuronList.append(neuron)

				#add to parent neuron:
				foundParentNeuron, parentNeuron = findParentNeuron(ATORneuronListAllLayers, resolutionProperties.resolutionIndex, neuron)
				if(foundParentNeuron):	
					#print("foundParentNeuron")
					parentNeuron.neuronComponents.append(neuron)
					ATORtf_RF.normaliseRFComponentWRTparent(resolutionProperties, neuron, parentNeuron.RFProperties)
					parentNeuron.neuronComponentsWeightsList.append(filterApplicationResult)
					
	ATORneuronListAllLayers.append(ATORneuronList)

def findParentNeuron(ATORneuronListAllLayers, resolutionIndex, neuron):
	foundParentNeuron = False
	parentNeuron = None
	if(resolutionIndex > resolutionIndexFirst):
		resolutionIndexLower = resolutionIndex-1
		ATORneuronList = ATORneuronListAllLayers[resolutionIndexLower]
		for lowerNeuron in ATORneuronList:
			#detect if RFProperties lies within RFPropertiesParent
			#CHECKTHIS: for now just use simple centroid detection algorithm
			if(ATORtf_RF.childRFoverlapsParentRF(neuron.RFPropertiesNormalisedGlobal, lowerNeuron.RFPropertiesNormalisedGlobal)):
				foundParentNeuron = True
				parentNeuron = lowerNeuron
		if(not foundParentNeuron):	
			#search for non-immediate (indirect) parent neuron:
			foundParentNeuron, neuronFound = findParentNeuron(ATORneuronListAllLayers, resolutionIndexLower, neuron)
	return foundParentNeuron, parentNeuron 




@click.command()
@click.argument('inputimagefilename')

def main(inputimagefilename):
	createRFhierarchyAccelerated(inputimagefilename)
	#ATORtf_detectEllipses.main(inputimagefilename)

if __name__ == "__main__":
	main()
	
