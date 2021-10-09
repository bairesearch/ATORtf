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

ATORtf uses ellipsoid features (rather than point/centroid features of the ATOR specification*), and normalises them with respect to their major/minor ellipticity axis orientation. 

There are a number of advantages of using ellipsoid features over point features;
* the number of feature sets/normalised snapshots required is significantly reduced
* scene component structure can be maintained (as detected component ellipses can be represented in a hierarchical graph structure)

Ellipse features/components are detected based on simulated artificial receptive fields; RF (on/off, off/on).

* https://www.wipo.int/patentscope/search/en/WO2011088497

# Future:
Requires upgrading to support 3D receptive field detection (ellipses and ellipsoids in 3D space)

"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import click
import cv2
import copy
import sys
import ATORtf_detectEllipses
import ATORtf_ellipseProperties
import ATORtf_operations

np.set_printoptions(threshold=sys.maxsize)

debugLowIterations = False
debugVerbose = True
debugSaveRFFiltersAndImageSegments = True	#only RF properties are required to be saved by ATOR algorithm (not image RF pixel data)
if(debugSaveRFFiltersAndImageSegments):
	RFFilterImageTransformFillValue = 0.0

storeRFFiltersValuesAsFractions = True	#store RFFilters values as fractions (multipliers) rather than colours (additive)
		
resolutionIndexFirst = 0	#mandatory
numberOfResolutions = 5	#x; lowest res sample: 1/(2^x)
minimumEllipseLength = 2
ellipseCenterCoordinatesResolution = 1	#pixels (at resolution r)
ellipseAxesLengthResolution = 1	#pixels (at resolution r)
ellipseAngleResolution = 30	#degrees
ellipseColourResolution = 64	#bits
minimumFilterRequirement = 1.5	#CHECKTHIS: calibrate	#matched values fraction	#theoretical value: 0.95

rgbMaxValue = 255.0
rgbNumChannels = 3

receptiveFieldOpponencyArea = 2.0	#the radius of the opponency/negative (-1) receptive field compared to the additive (+) receptive field

imageWidthBase = 256	#maximum image resolution used by ATORtf algorithm	#CHECKTHIS
imageHeightBase = 256 	#maximum image resolution used by ATORtf algorithm	#CHECKTHIS

ellipseNormalisedAngle = 0.0
ellipseNormalisedCentreCoordinates = 0.0 
ellipseNormalisedAxesLength = 1.0


class ATORneuron():
	def __init__(self, resolutionIndex, RFProperties, RFFilter, RFImage):
		self.resolutionIndex = resolutionIndex
		self.RFProperties = RFProperties
		self.RFPropertiesNormalised = normaliseRFProperties(RFProperties)
		self.RFPropertiesNormalisedWRTparent = None
		if(debugSaveRFFiltersAndImageSegments):
			self.RFFilterNormalised = normaliseRFFilter(RFFilter, RFProperties)
			self.RFFilterNormalisedWRTparent = None
			self.RFImage = RFImage
			self.RFImageNormalised = normaliseRFFilter(RFImage, RFProperties)
			self.RFImageNormalisedWRTparent = None
		self.neuronComponents = []
		self.neuronComponentsWeightsList = []

def createRFhierarchyAccelerated(inputimagefilename):
	
	inputImage = cv2.imread(inputimagefilename)	#FUTURE: support 3D datasets
	
	#normalise image size
	inputImage = cv2.resize(inputImage, (imageWidthBase, imageHeightBase))
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
	
	for resolutionIndex in range(resolutionIndexFirst, resolutionIndexMax):
		resolutionFactor, resolutionFactorReverse, imageSize = ATORtf_operations.getImageDimensionsR(resolutionIndex, resolutionIndexFirst, numberOfResolutions, imageWidthBase, imageHeightBase)
		print("resolutionFactor = ", resolutionFactor)
		print("resolutionFactorReverse = ", resolutionFactorReverse)
		print("imageSize = ", imageSize)
		inputImageRGBTF = tf.image.resize(inputImageRGBTF, imageSize)
		inputImageGrayTF = tf.image.resize(inputImageGrayTF, imageSize)
		inputImageRGBSegments, inputImageGraySegments = generateImageSegments(resolutionIndex, inputImageRGBTF, inputImageGrayTF)
		inputImageRGBSegmentsAllRes.append(inputImageRGBSegments)
		inputImageGraySegmentsAllRes.append(inputImageGraySegments)
		
	for resolutionIndex in range(resolutionIndexFirst, resolutionIndexMax):
		RFFiltersList, RFFiltersPropertiesList = generateRFFilters(resolutionIndex)
		RFFiltersListAllRes.append(RFFiltersList)
		RFFiltersPropertiesListAllRes.append(RFFiltersPropertiesList)
		
	#applyRFFilters:
	for resolutionIndex in range(resolutionIndexFirst, resolutionIndexMax):
		print("resolutionIndex = ", resolutionIndex)
		inputImageRGBSegments = inputImageRGBSegmentsAllRes[resolutionIndex]
		inputImageGraySegments = inputImageGraySegmentsAllRes[resolutionIndex]
		RFFiltersList = RFFiltersListAllRes[resolutionIndex]
		RFFiltersPropertiesList = RFFiltersPropertiesListAllRes[resolutionIndex]
		applyRFFiltersList(resolutionIndex, inputImageRGBSegments, inputImageGraySegments, RFFiltersList, RFFiltersPropertiesList, ATORneuronListAllLayers)

def generateImageSegments(resolutionIndex, inputImageRGBTF, inputImageGrayTF):
	inputImageRGBSegmentsList = []
	inputImageGraySegmentsList = []
	
	resolutionFactor, imageSize, axesLengthMax, filterRadius, filterSize = getFilterDimensions(resolutionIndex)
	
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
			allFilterCoordinatesWithinImageResult, imageSegmentStart, imageSegmentEnd = allFilterCoordinatesWithinImage(centerCoordinates, filterRadius, imageSize)
			#if(not allFilterCoordinatesWithinImageResult): image segments and their applied filters will be discarded, but artificial (beyond bounds) image segments are still added to inputImageSegments tensor for algorithm uniformity
			inputImageRGBSegment = inputImageRGBTF[imageSegmentStart[0]:imageSegmentEnd[0], imageSegmentStart[1]:imageSegmentEnd[1], :]
			inputImageGraySegment = inputImageGrayTF[imageSegmentStart[0]:imageSegmentEnd[0], imageSegmentStart[1]:imageSegmentEnd[1], :]
			if(storeRFFiltersValuesAsFractions):
				inputImageRGBSegment = tf.divide(inputImageRGBSegment, rgbMaxValue)
				inputImageGraySegment = tf.divide(inputImageGraySegment, rgbMaxValue)
			inputImageRGBSegmentsList.append(inputImageRGBSegment)
			inputImageGraySegmentsList.append(inputImageGraySegment)
			imageSegmentIndex = imageSegmentIndex+1
			
	inputImageRGBSegments = tf.stack(inputImageRGBSegmentsList)
	inputImageGraySegments = tf.stack(inputImageGraySegmentsList)
	
	#print("inputImageRGBSegments.shape = ", inputImageRGBSegments.shape)
	#print("inputImageGraySegments.shape = ", inputImageGraySegments.shape)
			
	return inputImageRGBSegments, inputImageGraySegments
	
def generateRFFilters(resolutionIndex):

	#2D code;
	
	#filters are generated based on human magnocellular/parvocellular/koniocellular wavelength discrimination in LGN and VX (double/opponent receptive fields)
	
	RFFiltersList = []
	RFFiltersPropertiesList = []
	
	#magnocellular filters (monochromatic);
	colourH = (255, 255, 255)	#high
	colourL = (-255, -255, -255)	#low
	RFFiltersHL, RFPropertiesHL = generateRotationalInvariantRFFilters(resolutionIndex, False, colourH, colourL)
	RFFiltersLH, RFPropertiesLH = generateRotationalInvariantRFFilters(resolutionIndex, False, colourL, colourH)
	
	#parvocellular/koniocellular filters (based on 2 cardinal colour axes; ~red-~green, ~blue-~yellow);
	colourRmG = (255, -255, 0)	#red+, green-
	colourGmR = (-255, 255, 0)	#green+, red-
	colourBmY = (-127, -127, 255)	#blue+, yellow-
	colourYmB = (127, 127, -255)	#yellow+, blue-
	RFFiltersRG, RFPropertiesRG = generateRotationalInvariantRFFilters(resolutionIndex, True, colourRmG, colourGmR)
	RFFiltersGR, RFPropertiesGR = generateRotationalInvariantRFFilters(resolutionIndex, True, colourGmR, colourRmG)
	RFFiltersBY, RFPropertiesBY = generateRotationalInvariantRFFilters(resolutionIndex, True, colourBmY, colourYmB)
	RFFiltersYB, RFPropertiesYB = generateRotationalInvariantRFFilters(resolutionIndex, True, colourYmB, colourBmY)
	
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

	return RFFiltersList, RFFiltersPropertiesList

def generateRotationalInvariantRFFilters(resolutionIndex, isColourFilter, filterInsideColour, filterOutsideColour):
	
	RFFiltersList2 = []
	RFFiltersPropertiesList2 = []
	
	#FUTURE: consider storing filters in n dimensional array and finding local minima of filter matches across all dimensions

	#reduce max size of ellipse at each res
	resolutionFactor, imageSize, axesLengthMax, filterRadius, filterSize = getFilterDimensions(resolutionIndex)
	
	#print("axesLengthMax = ", axesLengthMax)
	
	for axesLength1 in range(minimumEllipseLength, axesLengthMax[0]+1, ellipseAxesLengthResolution):
		for axesLength2 in range(minimumEllipseLength, axesLengthMax[1]+1, ellipseAxesLengthResolution):
			for angle in range(0, 360, ellipseAngleResolution):	#degrees
				
				axesLengthInside = (axesLength1, axesLength2)
				axesLengthOutside = (int(axesLength1*receptiveFieldOpponencyArea), int(axesLength2*receptiveFieldOpponencyArea))

				filterCenterCoordinates = (0, 0)
				RFPropertiesInside = ATORtf_ellipseProperties.EllipseProperties(resolutionIndex, resolutionFactor, filterSize, filterCenterCoordinates, axesLengthInside, angle, filterInsideColour)
				RFPropertiesOutside = ATORtf_ellipseProperties.EllipseProperties(resolutionIndex, resolutionFactor, filterSize, filterCenterCoordinates, axesLengthOutside, angle, filterOutsideColour)
				RFPropertiesInside.isColourFilter = isColourFilter
				RFPropertiesOutside.isColourFilter = isColourFilter

				RFFilter = generateRFFilter(resolutionIndex, isColourFilter, RFPropertiesInside, RFPropertiesOutside)
				RFFiltersList2.append(RFFilter)
				RFProperties = copy.deepcopy(RFPropertiesInside)
				#RFProperties.centerCoordinates = centerCoordinates 	#centerCoordinates are set after filter is applied to imageSegment
				RFFiltersPropertiesList2.append(RFProperties)	#CHECKTHIS: use RFPropertiesInside not RFPropertiesOutside

				#debug:
				#print(RFFilter.shape)
				if(debugVerbose):
					ATORtf_ellipseProperties.printEllipseProperties(RFPropertiesInside)
					ATORtf_ellipseProperties.printEllipseProperties(RFPropertiesOutside)				
				#print("RFFilter = ", RFFilter)

	#create 3D tensor (for hardware accelerated test/application of filters)
	RFFiltersTensor = tf.stack(RFFiltersList2, axis=0)

	return RFFiltersTensor, RFFiltersPropertiesList2
	

def generateRFFilter(resolutionIndex, isColourFilter, RFPropertiesInside, RFPropertiesOutside):

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
	#resolutionFactor, resolutionFactorReverse, imageSize = ATORtf_operations.getImageDimensionsR(resolutionIndex, resolutionIndexFirst, numberOfResolutions, imageWidthBase, imageHeightBase)
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
	
	if(storeRFFiltersValuesAsFractions):
		RFFilterTF = tf.divide(RFFilterTF, rgbMaxValue)

	#print("RFFilterTF = ", RFFilterTF)
	
	if(not isColourFilter):
		RFFilterTF = tf.image.rgb_to_grayscale(RFFilterTF)
			
	return RFFilterTF
	
def applyRFFiltersList(resolutionIndex, inputImageRGBSegments, inputImageGraySegments, RFFiltersList, RFFiltersPropertiesList, ATORneuronListAllLayers):
	
	print("\tapplyRFFiltersList: resolutionIndex = ", resolutionIndex)
	
	ATORneuronList = []	#for resolutionIndex
	
	resolutionFactor, resolutionFactorReverse, imageSize = ATORtf_operations.getImageDimensionsR(resolutionIndex, resolutionIndexFirst, numberOfResolutions, imageWidthBase, imageHeightBase)
	resolutionFactor, imageSize, axesLengthMax, filterRadius, filterSize = getFilterDimensions(resolutionIndex)

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

		filterApplicationResultThresholdIndicesList, filterApplicationResultThresholdedList, RFPropertiesList = applyRFFilters(resolutionIndex, inputImageSegments, RFFiltersTensor, numberOfDimensions, RFFiltersPropertiesList2)
			
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
			allFilterCoordinatesWithinImageResult, imageSegmentStart, imageSegmentEnd = allFilterCoordinatesWithinImage(centerCoordinates, filterRadius, imageSize)
			if(allFilterCoordinatesWithinImageResult):
			
				#create child neuron:
				neuron = ATORneuron(resolutionIndex, RFProperties, RFFilter, RFImage)
				ATORneuronList.append(neuron)

				#add to parent neuron:
				foundParentNeuron, parentNeuron = findNeuron(ATORneuronListAllLayers, resolutionIndex, RFProperties)
				if(foundParentNeuron):	
					parentNeuron.neuronComponents.append(neuron)
					normaliseRFComponentWRTparent(neuron, parentNeuron.RFProperties)
					parentNeuron.neuronComponentsWeightsList.append(filterApplicationResult)
					
	ATORneuronListAllLayers.append(ATORneuronList)
	
def applyRFFilters(resolutionIndex, inputImageSegments, RFFiltersTensor, numberOfDimensions, RFFiltersPropertiesList2):
	
	#perform convolution for each filter size;
	resolutionFactor, imageSize, axesLengthMax, filterRadius, filterSize = getFilterDimensions(resolutionIndex)
		
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
	filterApplicationResultThreshold = calculateFilterApplicationResultThreshold(filterApplicationResult, minimumFilterRequirement, filterSize, isColourFilter, numberOfDimensions)
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
		#		allFilterCoordinatesWithinImageResult, imageSegmentStart, imageSegmentEnd = allFilterCoordinatesWithinImage(centerCoordinates, filterRadius, imageSize)
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
	
def calculateFilterApplicationResultThreshold(filterApplicationResult, minimumFilterRequirement, filterSize, isColourFilter, numberOfDimensions):
	
	minimumFilterRequirementLocal = minimumFilterRequirement*calculateFilterPixels(filterSize, numberOfDimensions)
	
	#if(isColourFilter):
	#	minimumFilterRequirementLocal = minimumFilterRequirementLocal*rgbNumChannels*rgbNumChannels	#CHECKTHIS	#not required as assume filter colours will be normalised to the maximum value of a single rgb channel? 
	if(not storeRFFiltersValuesAsFractions):
		minimumFilterRequirementLocal = minimumFilterRequirementLocal*(rgbMaxValue*rgbMaxValue)	#rgbMaxValue of both imageSegment and RFFilter 		

	print("minimumFilterRequirementLocal = ", minimumFilterRequirementLocal)
	print("tf.math.reduce_max(filterApplicationResult) = ", tf.math.reduce_max(filterApplicationResult))
	
	filterApplicationResultThreshold = tf.greater(filterApplicationResult, minimumFilterRequirementLocal)	
	return filterApplicationResultThreshold

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
	
def findNeuron(ATORneuronListAllLayers, resolutionIndex, RFProperties):
	result = False
	neuronFound = None
	if(resolutionIndex > resolutionIndexFirst):
		resolutionIndexParent = resolutionIndex-1
		ATORneuronList = ATORneuronListAllLayers[resolutionIndexParent]
		for neuron in ATORneuronList:
			#detect if RFProperties lies within RFPropertiesParent
			#CHECKTHIS: for now just use simple centroid detection algorithm
			ellipseCentroidOverlapsesWithParent = ATORtf_ellipseProperties.centroidOverlapsEllipse(RFProperties, neuron.RFProperties)
			if(ellipseCentroidOverlapsesWithParent):
				result = True
				neuronFound = neuron	
	return result, neuronFound 
						
def normaliseRFProperties(RFProperties):
	#normalise ellipse respect to major/minor ellipticity axis orientation (WRT self)
	RFPropertiesNormalised = copy.deepcopy(RFProperties)
	RFPropertiesNormalised.angle = ellipseNormalisedAngle	#CHECKTHIS
	RFPropertiesNormalised.centerCoordinates = (ellipseNormalisedCentreCoordinates, ellipseNormalisedCentreCoordinates, ellipseNormalisedCentreCoordinates)
	RFPropertiesNormalised.axesLength = (ellipseNormalisedAxesLength, ellipseNormalisedAxesLength)
	return RFPropertiesNormalised
	
	#if(RFProperties.axesLength[0] > RFProperties.axesLength[1]):
	#	RFPropertiesNormalised.angle = 0
	#else:
	#	RFPropertiesNormalised.angle = 90
		
def normaliseRFComponentWRTparent(neuronComponent, RFPropertiesParent):
	neuronComponent.RFPropertiesNormalisedWRTparent = generateRFTransformedProperties(neuronComponent, RFPropertiesParent)
	if(debugSaveRFFiltersAndImageSegments):
		neuronComponent.RFFilterNormalisedWRTparent = transformRFFilterTF(neuronComponent.RFFilter, RFPropertiesParent)
		neuronComponent.RFImageNormalisedWRTparent = transformRFFilterTF(neuronComponent.RFImage, RFPropertiesParent)

def generateRFTransformedProperties(neuronComponent, RFPropertiesParent):
	if(RFPropertiesParent.numberOfDimensions == 2):
		return generateRFTransformedProperties2D(neuronComponent, RFPropertiesParent)
	elif(RFPropertiesParent.numberOfDimensions == 3):
		return generateRFTransformedProperties3D(neuronComponent, RFPropertiesParent)
		
def generateRFTransformedProperties2D(neuronComponent, RFPropertiesParent):
	RFTransformedProperties = copy.copy(neuronComponent.RFProperties)
	RFTransformedProperties.centerCoordinates = transformPoint2D(neuronComponent.RFProperties.centerCoordinates, RFPropertiesParent)
	endCoordinates = calculateEndCoordinatesPosition2D(neuronComponent)
	endCoordinates = transformPoint2D(endCoordinates, RFPropertiesParent)
	RFTransformedProperties.axesLength = ATORtf_operations.calculateDistance2D(RFTransformedProperties.centerCoordinates, endCoordinates)
	RFTransformedProperties.angle = neuronComponent.RFProperties.angle-RFPropertiesParent.angle
	return RFTransformedProperties
		
def generateRFTransformedProperties3D(neuronComponent, RFPropertiesParent):
	RFTransformedProperties = copy.copy(neuronComponent.RFProperties)
	RFTransformedProperties.centerCoordinates = transformPoint3D(neuronComponent.RFProperties.centerCoordinates, RFPropertiesParent)
	endCoordinates = calculateEndCoordinatesPosition3D(neuronComponent)
	endCoordinates = transformPoint3D(endCoordinates, RFPropertiesParent)
	RFTransformedProperties.axesLength = ATORtf_operations.calculateDistance3D(RFTransformedProperties.centerCoordinates, endCoordinates)
	RFTransformedProperties.angle = ((neuronComponent.RFProperties.angle[0]-RFPropertiesParent.angle[0]), (neuronComponent.RFProperties.angle[1]-RFPropertiesParent.angle[1]))
	return RFTransformedProperties

def transformPoint2D(coordinates, RFPropertiesParent):
	coordinatesTransformed = (coordinates[0]-RFPropertiesParent.centerCoordinates[0], coordinates[1]-RFPropertiesParent.centerCoordinates[1])
	coordinatesRelativeAfterRotation = ATORtf_operations.calculateRelativePosition2D(RFPropertiesParent.angle, RFPropertiesParent.axesLength[0])
	coordinatesTransformed = (coordinatesTransformed[0]-coordinatesRelativeAfterRotation[0], coordinatesTransformed[1]-coordinatesRelativeAfterRotation[1])	#CHECKTHIS: + or -
	coordinatesTransformed = (coordinates[0]/RFPropertiesParent.axesLength[0], coordinates[1]/RFPropertiesParent.axesLength[1])
	return coordinatesTransformed
	
def transformPoint3D(coordinates, RFPropertiesParent):
	coordinatesTransformed = (coordinates[0]-RFPropertiesParent.centerCoordinates[0], coordinates[1]-RFPropertiesParent.centerCoordinates[1], coordinates[2]-RFPropertiesParent.centerCoordinates[2])
	coordinatesRelativeAfterRotation = ATORtf_operations.calculateRelativePosition3D(RFPropertiesParent.angle, RFPropertiesParent.axesLength[0])
	coordinatesTransformed = (coordinatesTransformed[0]-coordinatesRelativeAfterRotation[0], coordinatesTransformed[1]-coordinatesRelativeAfterRotation[1], coordinatesTransformed[2]-coordinatesRelativeAfterRotation[2])	#CHECKTHIS: + or -
	coordinatesTransformed = (coordinates[0]/RFPropertiesParent.axesLength[0], coordinates[1]/RFPropertiesParent.axesLength[1], coordinates[2]/RFPropertiesParent.axesLength[2])
	return coordinatesTransformed
		
def calculateEndCoordinatesPosition2D(neuronComponent):
	endCoordinatesRelativeToCentreCoordinates = ATORtf_operations.calculateRelativePosition2D(neuronComponent.RFProperties.angle, neuronComponent.RFProperties.axesLength[0])
	endCoordinates = (neuronComponent.RFProperties.centerCoordinates[0]+endCoordinatesRelativeToCentreCoordinates[0], neuronComponent.RFProperties.centerCoordinates[1]+endCoordinatesRelativeToCentreCoordinates[1])	#CHECKTHIS: + or -
	return endCoordinates
	
def calculateEndCoordinatesPosition3D(neuronComponent):
	endCoordinatesRelativeToCentreCoordinates = ATORtf_operations.calculateRelativePosition3D(neuronComponent.RFProperties.angle, neuronComponent.RFProperties.axesLength)
	endCoordinates = (neuronComponent.RFProperties.centerCoordinates[0]+endCoordinatesRelativeToCentreCoordinates[0], neuronComponent.RFProperties.centerCoordinates[1]+endCoordinatesRelativeToCentreCoordinates[1], neuronComponent.RFProperties.centerCoordinates[2]+endCoordinatesRelativeToCentreCoordinates[2])	#CHECKTHIS: + or -
	return endCoordinates

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
		


def getFilterDimensions(resolutionIndex):
	resolutionFactor, resolutionFactorReverse, imageSize = ATORtf_operations.getImageDimensionsR(resolutionIndex, resolutionIndexFirst, numberOfResolutions, imageWidthBase, imageHeightBase)
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


	
@click.command()
@click.argument('inputimagefilename')

def main(inputimagefilename):
	createRFhierarchyAccelerated(inputimagefilename)
	#ATORtf_detectEllipses.main(inputimagefilename)

if __name__ == "__main__":
	main()
	
