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

ATORtf also supports point (corner/centroid) features of the ATOR specification; 
https://www.wipo.int/patentscope/search/en/WO2011088497

# Future:
Requires upgrading to support 3DOD receptive field detection (ellipses/ellipsoids/features in 3D space)

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
import ATORtf_RFproperties
import ATORtf_RFfilter
import ATORtf_operations

np.set_printoptions(threshold=sys.maxsize)

generateRFfiltersEllipse = True
generateRFfiltersTri = False

debugLowIterations = False
debugVerbose = True
debugSaveRFfiltersAndImageSegments = True	#only RF properties are required to be saved by ATOR algorithm (not image RF pixel data)

resolutionIndexFirst = 0	#mandatory
numberOfResolutions = 4	#x; lowest res sample: 1/(2^x)


ellipseCenterCoordinatesResolution = 1	#pixels (at resolution r)

imageSizeBase = (256, 256)	#maximum image resolution used by ATORtf algorithm	#CHECKTHIS


class ATORneuronClass():
	def __init__(self, resolutionProperties, RFproperties, RFfilter, RFImage):
		self.resolutionIndex = resolutionProperties.resolutionIndex
		self.RFproperties = RFproperties
		self.RFpropertiesNormalised = ATORtf_RF.normaliseLocalRFproperties(RFproperties)
		self.RFpropertiesNormalisedWRTparent = None
		
		self.RFpropertiesNormalisedGlobal = ATORtf_RF.normaliseGlobalRFproperties(RFproperties, resolutionProperties.resolutionFactor)
		if(debugSaveRFfiltersAndImageSegments):
			self.RFfilter = RFfilter
			self.RFfilterNormalised = ATORtf_RFfilter.normaliseRFfilter(RFfilter, RFproperties)
			self.RFfilterNormalisedWRTparent = None
			self.RFImage = RFImage
			self.RFImageNormalised = ATORtf_RFfilter.normaliseRFfilter(RFImage, RFproperties)
			self.RFImageNormalisedWRTparent = None
		self.neuronComponents = []
		self.neuronComponentsWeightsList = []

def prepareRFhierarchyAccelerated():

	RFfiltersListAllRes = []	#stores receptive field tensorflow objects (used for hardware accelerated filter detection)
	RFfiltersPropertiesListAllRes = []	#stores receptive field ellipse properties (position, size, rotation, colour etc)

	#RFneuralNetworkListAllRes = []	#store mapping between receptive fields at different resolutions (for fast lookup)	
	ATORneuronListAllLayers = []

	#generateRFfilters:
	if(debugLowIterations):
		resolutionIndexMax = 1
	else:
		resolutionIndexMax = numberOfResolutions
			
	for resolutionIndex in range(resolutionIndexFirst, resolutionIndexMax):
		resolutionProperties = ATORtf_operations.RFresolutionProperties(resolutionIndex, resolutionIndexFirst, numberOfResolutions, imageSizeBase, debugVerbose, debugSaveRFfiltersAndImageSegments)
		RFfiltersList, RFfiltersPropertiesList = ATORtf_RF.generateRFfilters(resolutionProperties, generateRFfiltersEllipse, generateRFfiltersTri)
		RFfiltersListAllRes.append(RFfiltersList)
		RFfiltersPropertiesListAllRes.append(RFfiltersPropertiesList)
		
	for resolutionIndex in range(resolutionIndexFirst, resolutionIndexMax):
		resolutionProperties = ATORtf_operations.RFresolutionProperties(resolutionIndex, resolutionIndexFirst, numberOfResolutions, imageSizeBase, debugVerbose, debugSaveRFfiltersAndImageSegments)
		ATORneuronListArray = initialiseATORneuronListArray(resolutionProperties)
		ATORneuronListAllLayers.append(ATORneuronListArray)
		
	return RFfiltersListAllRes, RFfiltersPropertiesListAllRes, ATORneuronListAllLayers
				
def updateRFhierarchyAccelerated(RFfiltersListAllRes, RFfiltersPropertiesListAllRes, ATORneuronListAllLayers, inputimagefilename):
	
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
			
	inputImageRGBSegmentsAllRes = []	#stores subsets of input image at different resolutions, centreCoordinates, and size
	inputImageGraySegmentsAllRes = []

	#generateRFfilters:
	if(debugLowIterations):
		resolutionIndexMax = 1
	else:
		resolutionIndexMax = numberOfResolutions
	
	for resolutionIndex in range(resolutionIndexFirst, resolutionIndexMax):
		resolutionProperties = ATORtf_operations.RFresolutionProperties(resolutionIndex, resolutionIndexFirst, numberOfResolutions, imageSizeBase, debugVerbose, debugSaveRFfiltersAndImageSegments)
		inputImageRGBTF = tf.image.resize(inputImageRGBTF, resolutionProperties.imageSize)
		inputImageGrayTF = tf.image.resize(inputImageGrayTF, resolutionProperties.imageSize)
		inputImageRGBSegments, inputImageGraySegments = generateImageSegments(resolutionProperties, inputImageRGBTF, inputImageGrayTF)
		inputImageRGBSegmentsAllRes.append(inputImageRGBSegments)
		inputImageGraySegmentsAllRes.append(inputImageGraySegments)
		
	#applyRFfilters:
	for resolutionIndex in range(resolutionIndexFirst, resolutionIndexMax):
		resolutionProperties = ATORtf_operations.RFresolutionProperties(resolutionIndex, resolutionIndexFirst, numberOfResolutions, imageSizeBase, debugVerbose, debugSaveRFfiltersAndImageSegments)
		print("resolutionIndex = ", resolutionIndex)
		inputImageRGBSegments = inputImageRGBSegmentsAllRes[resolutionIndex]
		inputImageGraySegments = inputImageGraySegmentsAllRes[resolutionIndex]
		RFfiltersList = RFfiltersListAllRes[resolutionIndex]
		RFfiltersPropertiesList = RFfiltersPropertiesListAllRes[resolutionIndex]
		applyRFfiltersList(resolutionProperties, inputImageRGBSegments, inputImageGraySegments, RFfiltersList, RFfiltersPropertiesList, ATORneuronListAllLayers)


def initialiseATORneuronListArray(resolutionProperties):
	size = (resolutionProperties.imageSize[0], resolutionProperties.imageSize[1])
	ATORneuronListArray = ATORtf_operations.initialiseEmpty2dimensionalList(size)
	return ATORneuronListArray
		
def generateImageSegments(resolutionProperties, inputImageRGBTF, inputImageGrayTF):
	inputImageRGBSegmentsList = []
	inputImageGraySegmentsList = []
	
	axesLengthMax, filterRadius, filterSize = ATORtf_RFfilter.getFilterDimensions(resolutionProperties)
	
	if(debugVerbose):
		print("")
		print("resolutionIndex = ", resolutionProperties.resolutionIndex)
		print("resolutionFactor = ", resolutionProperties.resolutionFactor)
		print("imageSize = ", resolutionProperties.imageSize)
		print("filterRadius = ", filterRadius)
		print("axesLengthMax = ", axesLengthMax)
		print("filterSize = ", filterSize)
			
	#print("inputImageRGBTF.shape = ", inputImageRGBTF.shape)
	#print("inputImageGrayTF.shape = ", inputImageGrayTF.shape)
	
	imageSegmentIndex = 0		
	for centerCoordinates1 in range(0, resolutionProperties.imageSize[0], ellipseCenterCoordinatesResolution):
		for centerCoordinates2 in range(0, resolutionProperties.imageSize[1], ellipseCenterCoordinatesResolution):
			centerCoordinates = (centerCoordinates1, centerCoordinates2)
			#print("imageSize = ", resolutionProperties.imageSize)
			#print("filterRadius = ", filterRadius)
			#print("filterSize = ", filterSize)
			#print("centerCoordinates = ", centerCoordinates)
			allFilterCoordinatesWithinImageResult, imageSegmentStart, imageSegmentEnd = ATORtf_RFfilter.allFilterCoordinatesWithinImage(centerCoordinates, filterRadius, resolutionProperties.imageSize)
			#if(not allFilterCoordinatesWithinImageResult): image segments and their applied filters will be discarded, but artificial (beyond bounds) image segments are still added to inputImageSegments tensor for algorithm uniformity
			inputImageRGBSegment = inputImageRGBTF[imageSegmentStart[0]:imageSegmentEnd[0], imageSegmentStart[1]:imageSegmentEnd[1], :]
			inputImageGraySegment = inputImageGrayTF[imageSegmentStart[0]:imageSegmentEnd[0], imageSegmentStart[1]:imageSegmentEnd[1], :]
			if(ATORtf_operations.storeRFfiltersValuesAsFractions):
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
	
	
def applyRFfiltersList(resolutionProperties, inputImageRGBSegments, inputImageGraySegments, RFfiltersList, RFfiltersPropertiesList, ATORneuronListAllLayers):
	
	print("\tapplyRFfiltersList: resolutionIndex = ", resolutionProperties.resolutionIndex)
	
	ATORneuronListArray = ATORneuronListAllLayers[resolutionProperties.resolutionIndex]	#for resolutionIndex
	
	axesLengthMax, filterRadius, filterSize = ATORtf_RFfilter.getFilterDimensions(resolutionProperties)

	#print("imageSize = ", resolutionProperties.imageSize)
	#print("inputImageGrayTF.shape = ", inputImageGrayTF.shape)
	#print("inputImageRGBTF.shape = ", inputImageRGBTF.shape)
	#inputImageRGBTF = tf.image.resize(inputImageRGBTF, resolutionProperties.imageSize)
	#inputImageGrayTF = tf.image.resize(inputImageGrayTF, resolutionProperties.imageSize)
	#print("inputImageGrayTF.shape = ", inputImageGrayTF.shape)
	#print("inputImageRGBTF.shape = ", inputImageRGBTF.shape)
				
	for RFlistIndex1 in range(len(RFfiltersPropertiesList)):	#or RFfiltersList
	
		print("RFlistIndex1 = ", RFlistIndex1)
		RFfiltersTensor = RFfiltersList[RFlistIndex1]
		RFfiltersPropertiesList2 = RFfiltersPropertiesList[RFlistIndex1]
		isColourFilter = RFfiltersPropertiesList2[0].isColourFilter
		numberOfDimensions = RFfiltersPropertiesList2[0].numberOfDimensions
		if(isColourFilter):
			inputImageSegments = inputImageRGBSegments
		else:
			inputImageSegments = inputImageGraySegments

		filterApplicationResultThresholdIndicesList, filterApplicationResultThresholdedList, RFpropertiesList = ATORtf_RF.applyRFfilters(resolutionProperties, inputImageSegments, RFfiltersTensor, numberOfDimensions, RFfiltersPropertiesList2)
			
		print("ATORneuronList append: len(filterApplicationResultThresholdIndicesList) = ", len(filterApplicationResultThresholdIndicesList))	
		for RFthresholdedListIndex, RFlistIndex in enumerate(filterApplicationResultThresholdIndicesList):
				
			filterApplicationResult = filterApplicationResultThresholdedList[RFthresholdedListIndex]
			RFproperties = RFpropertiesList[RFthresholdedListIndex]
			#print("type(RFproperties) = ", type(RFproperties))
			RFfilter = None
			RFImage = None
			if(debugSaveRFfiltersAndImageSegments):
				RFfilter = RFfiltersTensor[RFproperties.filterIndex]
				RFImage = inputImageSegments[RFproperties.imageSegmentIndex]

			centerCoordinates = RFproperties.centerCoordinates
			allFilterCoordinatesWithinImageResult, imageSegmentStart, imageSegmentEnd = ATORtf_RFfilter.allFilterCoordinatesWithinImage(centerCoordinates, filterRadius, resolutionProperties.imageSize)
			if(allFilterCoordinatesWithinImageResult):
			
				#create child neuron:
				neuron = ATORneuronClass(resolutionProperties, RFproperties, RFfilter, RFImage)
				
				ATORneuronList = ATORneuronListArray[centerCoordinates[0]][centerCoordinates[1]]
				if(ATORneuronList is None):	#ATORneuronList has not been initialised (filled with any items)
					#initialise ATORneuronList
					ATORneuronList = []
					ATORneuronListArray[centerCoordinates[0]][centerCoordinates[1]] = ATORneuronList	
				ATORneuronList.append(neuron)

				#add to parent neuron:
				foundParentNeuron, parentNeuron = findParentNeuron(ATORneuronListAllLayers, resolutionProperties.resolutionIndex, resolutionProperties, neuron)
				if(foundParentNeuron):	
					print("foundParentNeuron")
					parentNeuron.neuronComponents.append(neuron)
					ATORtf_RF.normaliseRFComponentWRTparent(resolutionProperties, neuron, parentNeuron.RFproperties)
					parentNeuron.neuronComponentsWeightsList.append(filterApplicationResult)

def findParentNeuron(ATORneuronListAllLayers, resolutionIndexLast, resolutionPropertiesChild, neuronChild):
	foundParentNeuron = False
	parentNeuron = None
	if(resolutionIndexLast > resolutionPropertiesChild.resolutionIndexFirst):
		resolutionIndex = resolutionIndexLast-1
		ATORneuronListArray = ATORneuronListAllLayers[resolutionIndex]
		
		resolutionProperties = copy.deepcopy(resolutionPropertiesChild)
		resolutionProperties.resolutionIndex = resolutionIndex
		resolutionProperties.resolutionFactor, resolutionProperties.resolutionFactorReverse, resolutionProperties.imageSize = ATORtf_operations.getImageDimensionsR(resolutionProperties)	#reinitialiseRFProperties based on resolutionIndexLower
		axesLengthMax, filterRadius, filterSize = ATORtf_RFfilter.getFilterDimensions(resolutionProperties)	#all for lower

		#for all candidate parent RFs within rough receptive range as derived by resolution and centerCoordinates:
		RFrangeGlobalMin, RFrangeGlobalMax = getRFrangeAtResolutionGlobal(resolutionPropertiesChild, neuronChild.RFproperties.centerCoordinates)
		RFrangeLocalMin, RFrangeLocalMax = getRFrangeAtResolutionLocal(resolutionProperties, RFrangeGlobalMin, RFrangeGlobalMax)
		for centerCoordinates1 in range(RFrangeLocalMin[0], RFrangeLocalMax[0]+1, ellipseCenterCoordinatesResolution):
			for centerCoordinates2 in range(RFrangeLocalMin[1], RFrangeLocalMax[1]+1, ellipseCenterCoordinatesResolution):
				centerCoordinates = [centerCoordinates1, centerCoordinates2]
				
				allFilterCoordinatesWithinImageResult, imageSegmentStart, imageSegmentEnd = ATORtf_RFfilter.allFilterCoordinatesWithinImage(centerCoordinates, filterRadius, resolutionProperties.imageSize)
				if(allFilterCoordinatesWithinImageResult):
			
					ATORneuronList = ATORneuronListArray[centerCoordinates[0]][centerCoordinates[1]]
					if(ATORneuronList is not None):	#verify ATORneuronList has already been filled with some items 
						for neuron in ATORneuronList:
							#detect if RFproperties lies within RFpropertiesParent
							#CHECKTHIS: for now just use simple centroid detection algorithm - this needs to be upgraded with a robust method
							if(ATORtf_RF.childRFoverlapsParentRF(neuronChild.RFpropertiesNormalisedGlobal, neuron.RFpropertiesNormalisedGlobal)):
								foundParentNeuron = True
								parentNeuron = neuron
								
		if(not foundParentNeuron):	
			#search for non-immediate (indirect) parent neuron:
			foundParentNeuron, neuronFound = findParentNeuron(ATORneuronListAllLayers, resolutionIndex, resolutionPropertiesChild, neuronChild)
			
	return foundParentNeuron, parentNeuron 

	
def getRFrangeAtResolutionGlobal(resolutionProperties, centerCoordinates):
	axesLengthMax, filterRadius, filterSize = ATORtf_RFfilter.getFilterDimensions(resolutionProperties)
	
	RFrangeGlobalCentre = [int(centerCoordinates[0]*resolutionProperties.resolutionFactor), int(centerCoordinates[1]*resolutionProperties.resolutionFactor)]	#or RFrangeGlobalCentre = list(RFpropertiesNormalisedGlobal.centerCoordinates)
	RFrangeGlobalSize = [int(filterSize[0]*resolutionProperties.resolutionFactor), int(filterSize[1]*resolutionProperties.resolutionFactor)]
	RFrangeGlobalMin = [RFrangeGlobalCentre[0]-RFrangeGlobalSize[0], RFrangeGlobalCentre[1]-RFrangeGlobalSize[1]]
	RFrangeGlobalMax = [RFrangeGlobalCentre[0]+RFrangeGlobalSize[0], RFrangeGlobalCentre[1]+RFrangeGlobalSize[1]]
	return RFrangeGlobalMin, RFrangeGlobalMax
	
def getRFrangeAtResolutionLocal(resolutionProperties, RFrangeGlobalMin, RFrangeGlobalMax):
	RFrangeLocalMin = [int(RFrangeGlobalMin[0]/resolutionProperties.resolutionFactor), int(RFrangeGlobalMin[1]/resolutionProperties.resolutionFactor)]
	RFrangeLocalMax = [int(RFrangeGlobalMax[0]/resolutionProperties.resolutionFactor), int(RFrangeGlobalMax[1]/resolutionProperties.resolutionFactor)]
	return RFrangeLocalMin, RFrangeLocalMax


@click.command()
@click.argument('inputimagefilename')

def main(inputimagefilename):

	#ATORtf_detectEllipses.main(inputimagefilename)

	RFfiltersListAllRes, RFfiltersPropertiesListAllRes, ATORneuronListAllLayers = prepareRFhierarchyAccelerated()
	
	updateRFhierarchyAccelerated(RFfiltersListAllRes, RFfiltersPropertiesListAllRes, ATORneuronListAllLayers, inputimagefilename)	#trial image
	

if __name__ == "__main__":
	main()
	
