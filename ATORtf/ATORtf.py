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

generateRFfiltersEllipse = False
generateRFfiltersTri = True

debugLowIterations = False
debugVerbose = True
debugSaveRFfiltersAndImageSegments = True	#only RF properties are required to be saved by ATOR algorithm (not image RF pixel data)

resolutionIndexFirst = 0	#mandatory
numberOfResolutions = 5	#x; lowest res sample: 1/(2^x)


ellipseCenterCoordinatesResolution = 1	#pixels (at resolution r)

imageSizeBase = (256, 256)	#maximum image resolution used by ATORtf algorithm	#CHECKTHIS


class ATORneuronClass():
	def __init__(self, resolutionIndex, resolutionFactor, RFproperties, RFfilter, RFImage):
		self.resolutionIndex = resolutionIndex
		self.RFproperties = RFproperties
		self.RFpropertiesNormalised = ATORtf_RF.normaliseLocalRFproperties(RFproperties)
		self.RFpropertiesNormalisedWRTparent = None
		
		self.RFpropertiesNormalisedGlobal = ATORtf_RF.normaliseGlobalRFproperties(RFproperties, resolutionFactor)
		if(debugSaveRFfiltersAndImageSegments):
			self.RFfilter = RFfilter
			self.RFfilterNormalised = ATORtf_RFfilter.normaliseRFfilter(RFfilter, RFproperties)
			self.RFfilterNormalisedWRTparent = None
			self.RFImage = RFImage
			self.RFImageNormalised = ATORtf_RFfilter.normaliseRFfilter(RFImage, RFproperties)
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
	RFfiltersListAllRes = []	#stores receptive field tensorflow objects (used for hardware accelerated filter detection)
	RFfiltersPropertiesListAllRes = []	#stores receptive field ellipse properties (position, size, rotation, colour etc)
	
	#generateRFfilters:
	if(debugLowIterations):
		resolutionIndexMax = 1
	else:
		resolutionIndexMax = numberOfResolutions
	
	resolutionProperties = ATORtf_operations.RFresolutionProperties(-1, resolutionIndexFirst, numberOfResolutions, imageSizeBase, debugVerbose, debugSaveRFfiltersAndImageSegments)

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
		RFfiltersList, RFfiltersPropertiesList = ATORtf_RF.generateRFfilters(resolutionProperties, generateRFfiltersEllipse, generateRFfiltersTri)
		RFfiltersListAllRes.append(RFfiltersList)
		RFfiltersPropertiesListAllRes.append(RFfiltersPropertiesList)
		
	#applyRFfilters:
	for resolutionIndex in range(resolutionIndexFirst, resolutionIndexMax):
		resolutionProperties.resolutionIndex = resolutionIndex
		print("resolutionIndex = ", resolutionIndex)
		inputImageRGBSegments = inputImageRGBSegmentsAllRes[resolutionIndex]
		inputImageGraySegments = inputImageGraySegmentsAllRes[resolutionIndex]
		RFfiltersList = RFfiltersListAllRes[resolutionIndex]
		RFfiltersPropertiesList = RFfiltersPropertiesListAllRes[resolutionIndex]
		applyRFfiltersList(resolutionProperties, inputImageRGBSegments, inputImageGraySegments, RFfiltersList, RFfiltersPropertiesList, ATORneuronListAllLayers)

def generateImageSegments(resolutionProperties, inputImageRGBTF, inputImageGrayTF):
	inputImageRGBSegmentsList = []
	inputImageGraySegmentsList = []
	
	resolutionFactor, imageSize, axesLengthMax, filterRadius, filterSize = ATORtf_RFfilter.getFilterDimensions(resolutionProperties)
	
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
			allFilterCoordinatesWithinImageResult, imageSegmentStart, imageSegmentEnd = ATORtf_RFfilter.allFilterCoordinatesWithinImage(centerCoordinates, filterRadius, imageSize)
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
	
	ATORneuronList = []	#for resolutionIndex
	
	resolutionFactor, resolutionFactorReverse, imageSize = ATORtf_operations.getImageDimensionsR(resolutionProperties)
	resolutionFactor, imageSize, axesLengthMax, filterRadius, filterSize = ATORtf_RFfilter.getFilterDimensions(resolutionProperties)

	#print("imageSize = ", imageSize)
	#print("inputImageGrayTF.shape = ", inputImageGrayTF.shape)
	#print("inputImageRGBTF.shape = ", inputImageRGBTF.shape)
	#inputImageRGBTF = tf.image.resize(inputImageRGBTF, imageSize)
	#inputImageGrayTF = tf.image.resize(inputImageGrayTF, imageSize)
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
			allFilterCoordinatesWithinImageResult, imageSegmentStart, imageSegmentEnd = ATORtf_RFfilter.allFilterCoordinatesWithinImage(centerCoordinates, filterRadius, imageSize)
			if(allFilterCoordinatesWithinImageResult):
			
				#create child neuron:
				neuron = ATORneuronClass(resolutionProperties.resolutionIndex, resolutionFactor, RFproperties, RFfilter, RFImage)
				ATORneuronList.append(neuron)

				#add to parent neuron:
				foundParentNeuron, parentNeuron = findParentNeuron(ATORneuronListAllLayers, resolutionProperties.resolutionIndex, neuron)
				if(foundParentNeuron):	
					#print("foundParentNeuron")
					parentNeuron.neuronComponents.append(neuron)
					ATORtf_RF.normaliseRFComponentWRTparent(resolutionProperties, neuron, parentNeuron.RFproperties)
					parentNeuron.neuronComponentsWeightsList.append(filterApplicationResult)
					
	ATORneuronListAllLayers.append(ATORneuronList)

def findParentNeuron(ATORneuronListAllLayers, resolutionIndex, neuron):
	foundParentNeuron = False
	parentNeuron = None
	if(resolutionIndex > resolutionIndexFirst):
		resolutionIndexLower = resolutionIndex-1
		ATORneuronList = ATORneuronListAllLayers[resolutionIndexLower]
		for lowerNeuron in ATORneuronList:
			#detect if RFproperties lies within RFpropertiesParent
			#CHECKTHIS: for now just use simple centroid detection algorithm
			if(ATORtf_RF.childRFoverlapsParentRF(neuron.RFpropertiesNormalisedGlobal, lowerNeuron.RFpropertiesNormalisedGlobal)):
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
	
