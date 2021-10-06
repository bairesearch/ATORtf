# -*- coding: utf-8 -*-
"""ATORtf.py

# Author:
Richard Bruce Baxter - Copyright (c) 2021 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
conda create -n ATORtf python=3
source activate ATORtf
conda install --file condaRequirements.txt
	where condaRequirements.txt contains;
		numpy
		tensorflow
		click
		opencv
		pillow
conda install -c esri tensorflow-addons
	
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

import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import click
import os
import cv2
import copy
import sys
import ATORtf_detectEllipses
import ATORtf_ellipseProperties
import ATORtf_operations

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
np.set_printoptions(threshold=sys.maxsize)

debugLowIterations = True
debugVerbose = True

resolutionIndexFirst = 1
numberOfResolutions = 5	#x; lowest res sample: 1/(2^x)
minimumEllipseLength = 2
ellipseCenterCoordinatesResolution = 1	#pixels (at resolution r)
ellipseAxesLengthResolution = 1	#pixels (at resolution r)
ellipseAngleResolution = 30	#degrees
ellipseColourResolution = 64	#bits
minimumFilterRequirement = 1.0	#CHECKTHIS: calibrate

receptiveFieldOpponencyArea = 2.0	#the radius of the opponency/negative (-1) receptive field compared to the additive (+) receptive field

imageWidthBase = 256	#maximum image resolution used by ATORtf algorithm	#CHECKTHIS
imageHeightBase = 256 	#maximum image resolution used by ATORtf algorithm	#CHECKTHIS

ellipseNormalisedAngle = 0.0
ellipseNormalisedCentreCoordinates = 0.0 
ellipseNormalisedAxesLength = 1.0

debugSaveRFFilters = True	#only RF properties are required to be saved by ATOR algorithm (not image RF pixel data)

class ATORneuron():
	def __init__(self, resolutionIndex, RFProperties, RFFilter):
		self.resolutionIndex = resolutionIndex
		self.RFProperties = RFProperties
		self.RFPropertiesNormalised = normaliseRFProperties(RFProperties)
		self.RFPropertiesNormalisedWRTparent = None
		if(debugSaveRFFilters):
			self.RFFilter = RFFilter
			self.RFFilterNormalised = normaliseRFFilter(RFFilter, RFProperties)	#not required: for debugging only
			self.RFFilterNormalisedWRTparent = None
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
		resolutionIndexMax = 2
	else:
		resolutionIndexMax = numberOfResolutions+1
	
	for resolutionIndex in range(resolutionIndexFirst, resolutionIndexMax):
		resolutionFactor, resolutionFactorReverse, imageSize = getImageDimensionsR(resolutionIndex)
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
		inputImageRGBSegments = inputImageRGBSegmentsAllRes[resolutionIndex-resolutionIndexFirst]
		inputImageGraySegments = inputImageGraySegmentsAllRes[resolutionIndex-resolutionIndexFirst]
		RFFiltersList = RFFiltersListAllRes[resolutionIndex-resolutionIndexFirst]
		RFFiltersPropertiesList = RFFiltersPropertiesListAllRes[resolutionIndex-resolutionIndexFirst]
		applyRFFiltersList(resolutionIndex, inputImageRGBSegments, inputImageGraySegments, RFFiltersList, RFFiltersPropertiesList, ATORneuronListAllLayers)

def generateImageSegments(resolutionIndex, inputImageRGBTF, inputImageGrayTF):
	inputImageRGBSegmentsList = []
	inputImageGraySegmentsList = []
	
	resolutionFactor, imageSize, axesLengthMax, filterRadius, filterSize = getFilterDimensions(resolutionIndex)
	
	#print("inputImageRGBTF.shape = ", inputImageRGBTF.shape)
	#print("inputImageGrayTF.shape = ", inputImageGrayTF.shape)
			
	for centerCoordinates1 in range(0, imageSize[0], ellipseCenterCoordinatesResolution):
		for centerCoordinates2 in range(0, imageSize[1], ellipseCenterCoordinatesResolution):
			centerCoordinates = (centerCoordinates1, centerCoordinates2)
			#print("imageSize = ", imageSize)
			#print("filterRadius = ", filterRadius)
			#print("filterSize = ", filterSize)
			#print("centerCoordinates = ", centerCoordinates)
			allFilterCoordinatesWithinImageResult, imageSegmentStart, imageSegmentEnd = allFilterCoordinatesWithinImage(centerCoordinates, filterRadius, imageSize)
			if(allFilterCoordinatesWithinImageResult):
				inputImageRGBSegment = inputImageRGBTF[imageSegmentStart[0]:imageSegmentEnd[0], imageSegmentStart[1]:imageSegmentEnd[1], :]
				inputImageGraySegment = inputImageGrayTF[imageSegmentStart[0]:imageSegmentEnd[0], imageSegmentStart[1]:imageSegmentEnd[1], :]
				inputImageRGBSegmentsList.append(inputImageRGBSegment)
				inputImageGraySegmentsList.append(inputImageGraySegment)
			
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
	colourL = (000, 000, 000)	#low
	RFFiltersHL, RFPropertiesHL = generateRotationalInvariantRFFilters(resolutionIndex, False, colourH, colourL)
	RFFiltersLH, RFPropertiesLH = generateRotationalInvariantRFFilters(resolutionIndex, False, colourL, colourH)
	
	#parvocellular/koniocellular filters (based on 2 cardinal colour axes; ~red-~green, ~blue-~yellow);
	colourR = (255, -255, 0)	#red+, green-
	colourG = (-255, 255, 0)	#green+, red-
	colourB = (-127, -127, 255)	#blue+, yellow-
	colourY = (127, 127, -255)	#yellow+, blue-
	RFFiltersRG, RFPropertiesRG = generateRotationalInvariantRFFilters(resolutionIndex, True, colourR, colourG)
	RFFiltersGR, RFPropertiesGR = generateRotationalInvariantRFFilters(resolutionIndex, True, colourG, colourR)
	RFFiltersBY, RFPropertiesBY = generateRotationalInvariantRFFilters(resolutionIndex, True, colourB, colourY)
	RFFiltersYB, RFPropertiesYB = generateRotationalInvariantRFFilters(resolutionIndex, True, colourY, colourB)
	
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
	#resolutionFactor, resolutionFactorReverse, imageSize = getImageDimensionsR(resolutionIndex)
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
	
	inputImageTF = tf.convert_to_tensor(ellipseFilterImageInside, dtype=tf.float32)	#bool
	inputImageTF = tf.greater(inputImageTF, 0.0)
	inputImageTF = tf.dtypes.cast(inputImageTF, tf.float32)
	
	outsideImageTF = tf.convert_to_tensor(ellipseFilterImageOutside, dtype=tf.float32)
	outsideImageTF = tf.greater(outsideImageTF, 0.0)
	outsideImageTF = tf.dtypes.cast(outsideImageTF, tf.float32)
	outsideImageTF = tf.math.negative(outsideImageTF)
	
	#print(inputImageTF.shape)
	#print(outsideImageTF.shape)
			
	#add colour channels;
	#inputImageTF = tf.expand_dims(inputImageTF, axis=2)
	multiples = tf.constant([1,1,3], tf.int32)	#for 2D data only
	inputImageTF = tf.tile(inputImageTF, multiples)
	#print(inputImageTF.shape)
	RFColourInside = tf.constant([RFPropertiesInside.colour[0], RFPropertiesInside.colour[1], RFPropertiesInside.colour[2]], dtype=tf.float32)
	RFColourInside = ATORtf_operations.expandDimsN(RFColourInside, RFPropertiesInside.numberOfDimensions, axis=0)
	inputImageTF = tf.multiply(inputImageTF, RFColourInside)
	
	#outsideImageTF = tf.expand_dims(outsideImageTF, axis=2)
	multiples = tf.constant([1,1,3], tf.int32)	#for 2D data only
	outsideImageTF = tf.tile(outsideImageTF, multiples)
	#print(outsideImageTF.shape)
	RFColourOutside = tf.constant([RFPropertiesOutside.colour[0], RFPropertiesOutside.colour[1], RFPropertiesOutside.colour[2]], dtype=tf.float32)
	RFColourOutside = ATORtf_operations.expandDimsN(RFColourOutside, RFPropertiesOutside.numberOfDimensions, axis=0)
	outsideImageTF = tf.multiply(outsideImageTF, RFColourOutside)

	#print(RFColourInside.shape)
	#print(RFColourOutside.shape)
	#print(inputImageTF.shape)
	#print(outsideImageTF.shape)
		
	RFFilterTF = tf.convert_to_tensor(blankArray, dtype=tf.float32)
	RFFilterTF = tf.add(RFFilterTF, inputImageTF)
	RFFilterTF = tf.add(RFFilterTF, outsideImageTF)
	
	if(not isColourFilter):
		RFFilterTF = tf.image.rgb_to_grayscale(RFFilterTF)
			
	return RFFilterTF
	
def applyRFFiltersList(resolutionIndex, inputImageRGBSegments, inputImageGraySegments, RFFiltersList, RFFiltersPropertiesList, ATORneuronListAllLayers):
	
	ATORneuronList = []	#for resolutionIndex
	
	resolutionFactor, resolutionFactorReverse, imageSize = getImageDimensionsR(resolutionIndex)
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
			filterApplicationResultList, RFPropertiesList2 = applyRFFilters(resolutionIndex, inputImageRGBSegments, RFFiltersTensor, numberOfDimensions, RFFiltersPropertiesList2)		
		else:
			filterApplicationResultList, RFPropertiesList2 = applyRFFilters(resolutionIndex, inputImageGraySegments, RFFiltersTensor, numberOfDimensions, RFFiltersPropertiesList2)
				
		for RFlistIndex2, filterApplicationResult in enumerate(filterApplicationResultList):
			if(filterApplicationResult > minimumFilterRequirement):
				
				RFProperties = RFPropertiesList2[RFlistIndex2]
				RFFilter = None
				if(debugSaveRFFilters):
					RFFilter = RFFiltersTensor[RFlistIndex2]	#not required
					
				#create child neuron:
				neuron = ATORneuron(resolutionIndex, RFProperties, RFFilter)
				ATORneuronList.append(neuron)
				
				#add to parent neuron:
				foundParentNeuron, parentNeuron = findNeuron(ATORneuronListAllLayers, resolutionIndex-1, RFProperties)
				if(foundParentNeuron):	
					parentNeuron.neuronComponents.append(neuron)
					normaliseRFComponentWRTparent(neuron, parentNeuron.RFProperties)
					parentNeuron.neuronComponentsWeightsList.append(filterApplicationResult)
					
	ATORneuronListAllLayers.append(ATORneuronList)
	
def applyRFFilters(resolutionIndex, inputImageSegments, RFFiltersTensor, numberOfDimensions, RFFiltersPropertiesList2):
	
	#perform convolution for each filter size;
	resolutionFactor, imageSize, axesLengthMax, filterRadius, filterSize = getFilterDimensions(resolutionIndex)
		
	filterApplicationResultList = []
	RFPropertiesList2 = []

	#print("RFFiltersPropertiesList2[0].isColourFilter = ", RFFiltersPropertiesList2[0].isColourFilter)
	#print("inputImageSegments.shape = ", inputImageSegments.shape)
	#print("RFFiltersTensor.shape = ", RFFiltersTensor.shape)
		
	#inputImageSegments dim: num inputImageSegments, x, y, c
	#RFFiltersTensor dim: num RFFilters, x, y, c
	inputImageSegmentsPixelsFlattened = tf.reshape(inputImageSegments, [inputImageSegments.shape[0], inputImageSegments.shape[1]*inputImageSegments.shape[2]*inputImageSegments.shape[3]])
	RFFiltersTensorPixelsFlattened = tf.reshape(RFFiltersTensor, [RFFiltersTensor.shape[0], RFFiltersTensor.shape[1]*RFFiltersTensor.shape[2]*RFFiltersTensor.shape[3]])
	
	filterApplicationResult = tf.matmul(inputImageSegmentsPixelsFlattened, tf.transpose(RFFiltersTensorPixelsFlattened))	#dim: num inputImageSegments, num RFFilters
	filterApplicationResult	= tf.reshape(filterApplicationResult, [filterApplicationResult.shape[0]*filterApplicationResult.shape[1]]) #flatten	#dim: num inputImageSegments * num RFFilters
	
	filterApplicationResultNP = filterApplicationResult.numpy()	#verify 1D
	filterApplicationResultList.extend(filterApplicationResultNP.tolist())
	
	for centerCoordinates1 in range(0, imageSize[0], ellipseCenterCoordinatesResolution):
		for centerCoordinates2 in range(0, imageSize[1], ellipseCenterCoordinatesResolution):
			centerCoordinates = (centerCoordinates1, centerCoordinates2)
			allFilterCoordinatesWithinImageResult, imageSegmentStart, imageSegmentEnd = allFilterCoordinatesWithinImage(centerCoordinates, filterRadius, imageSize)
			if(allFilterCoordinatesWithinImageResult):
				for RFFiltersProperties in RFFiltersPropertiesList2:
					RFProperties = copy.deepcopy(RFFiltersProperties)
					RFProperties.centerCoordinates = centerCoordinates
					RFPropertiesList2.append(RFProperties)

	#verify these match:
	#print("len(filterApplicationResultList) = ", len(filterApplicationResultList))
	#print("len(RFPropertiesList2) = ", len(RFPropertiesList2))

	return filterApplicationResultList, RFPropertiesList2	#filterApplicationResultThreshold
				

def findNeuron(ATORneuronAllLayers, resolutionIndex, RFProperties):
	result = False
	neuronFound = None
	if(resolutionIndex > resolutionIndexFirst):
		resolutionIndexParent = resolutionIndex-1
		for ATORneuronList in ATORneuronListAllLayers[resolutionIndexParent]:
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
	if(debugSaveRFFilters):
		neuronComponent.RFFilterNormalisedWRTparent = transformRFFilterTF(neuronComponent.RFFilter, RFPropertiesParent)	#not required: for debugging only

def generateRFTransformedProperties(neuronComponent, RFPropertiesParent):
	if(RFPropertiesParent.numberOfDimensions == 2):
		return generateRFTransformedProperties2D(neuronComponent, RFPropertiesParent)
	elif(RFPropertiesParent.numberOfDimensions == 3):
		return generateRFTransformedProperties3D(neuronComponent, RFPropertiesParent)
		
def generateRFTransformedProperties2D(neuronComponent, RFPropertiesParent):
	RFTransformedProperties = copy.copy(neuronComponent.RFProperties)
	RFTransformedProperties.centerCoordinates = transformPoint2D(neuronComponent.centerCoordinates, RFPropertiesParent)
	endCoordinates = calculateEndCoordinatesPosition2D(neuronComponent)
	endCoordinates = transformPoint2D(endCoordinates, RFPropertiesParent)
	RFTransformedProperties.axesLength = calculateDistance2D(RFTransformedProperties.centerCoordinates, endCoordinates)
	RFTransformedProperties.angle = neuronComponent.angle-RFPropertiesParent.angle
	return RFTransformedProperties
		
def generateRFTransformedProperties3D(neuronComponent, RFPropertiesParent):
	RFTransformedProperties = copy.copy(neuronComponent.RFProperties)
	RFTransformedProperties.centerCoordinates = transformPoint3D(neuronComponent.centerCoordinates, RFPropertiesParent)
	endCoordinates = calculateEndCoordinatesPosition3D(neuronComponent)
	endCoordinates = transformPoint3D(endCoordinates, RFPropertiesParent)
	RFTransformedProperties.axesLength = calculateDistance3D(RFTransformedProperties.centerCoordinates, endCoordinates)
	RFTransformedProperties.angle = ((neuronComponent.angle[0]-RFPropertiesParent.angle[0]), (neuronComponent.angle[1]-RFPropertiesParent.angle[1]))
	return RFTransformedProperties

def transformPoint2D(coordinates, RFPropertiesParent):
	coordinatesTransformed = (coordinates[0]-RFPropertiesParent.centerCoordinates[0], coordinates[1]-RFPropertiesParent.centerCoordinates[1])
	coordinatesRelativeAfterRotation = ATORtf_operations.calculateRelativePosition2D(RFPropertiesParent.angle, RFPropertiesParent.axisLength[0])
	coordinatesTransformed = (coordinatesTransformed[0]-coordinatesRelativeAfterRotation[0], coordinatesTransformed[1]-coordinatesRelativeAfterRotation[1])	#CHECKTHIS: + or -
	coordinatesTransformed = (coordinates[0]/RFPropertiesParent.axesLength[0], coordinates[1]/RFPropertiesParent.axesLength[1])
	return coordinatesTransformed
	
def transformPoint3D(coordinates, RFPropertiesParent):
	coordinatesTransformed = (coordinates[0]-RFPropertiesParent.centerCoordinates[0], coordinates[1]-RFPropertiesParent.centerCoordinates[1], coordinates[2]-RFPropertiesParent.centerCoordinates[2])
	coordinatesRelativeAfterRotation = ATORtf_operations.calculateRelativePosition3D(RFPropertiesParent.angle, RFPropertiesParent.axisLength[0])
	coordinatesTransformed = (coordinatesTransformed[0]-coordinatesRelativeAfterRotation[0], coordinatesTransformed[1]-coordinatesRelativeAfterRotation[1], coordinatesTransformed[2]-coordinatesRelativeAfterRotation[2])	#CHECKTHIS: + or -
	coordinatesTransformed = (coordinates[0]/RFPropertiesParent.axesLength[0], coordinates[1]/RFPropertiesParent.axesLength[1], coordinates[2]/RFPropertiesParent.axesLength[2])
	return coordinatesTransformed
		
def calculateEndCoordinatesPosition2D(neuronComponent):
	endCoordinatesRelativeToCentreCoordinates = ATORtf_operations.calculateRelativePosition2D(neuronComponent.angle, neuronComponent.axisLength[0])
	endCoordinates = (neuronComponent.centerCoordinates[0]+endCoordinatesRelativeToCentreCoordinates[0], neuronComponent.centerCoordinates[1]+endCoordinatesRelativeToCentreCoordinates[1])	#CHECKTHIS: + or -
	return endCoordinates
	
def calculateEndCoordinatesPosition3D(neuronComponent):
	endCoordinatesRelativeToCentreCoordinates = ATORtf_operations.calculateRelativePosition3D(neuronComponent.angle, neuronComponent.axisLength)
	endCoordinates = (neuronComponent.centerCoordinates[0]+endCoordinatesRelativeToCentreCoordinates[0], neuronComponent.centerCoordinates[1]+endCoordinatesRelativeToCentreCoordinates[1], neuronComponent.centerCoordinates[2]+endCoordinatesRelativeToCentreCoordinates[2])	#CHECKTHIS: + or -
	return endCoordinates

def normaliseRFFilter(RFFilter, RFProperties):
	#normalise ellipse respect to major/minor ellipticity axis orientation (WRT self)
	RFFilterNormalised = transformRFFilterTF(RFFilter, RFProperties) 
	#RFFilterNormalised = RFFilter
	return RFFilterNormalised
	
def transformRFFilterTF(RFFilter, RFPropertiesParent):
	if(RFPropertiesParent.numberOfDimensions == 2):
		centerCoordinates = [-RFPropertiesParent.centerCoordinates[0], -RFPropertiesParent.centerCoordinates[1]]
		axesLength = [1.0/RFPropertiesParent.axesLength[0], 1.0/RFPropertiesParent.axesLength[1]]
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
	print("RFFilter.shape = ", RFFilter.shape)
	angleRadians =  ATORtf_operations.convertDegreesToRadians(angle)
	print("angleRadians = ", angleRadians)
	print("RFFilterTransformed.shape = ", RFFilterTransformed.shape)
	RFFilterTransformed = tfa.image.rotate(RFFilterTransformed, angleRadians)		#https://www.tensorflow.org/addons/api_docs/python/tfa/image/rotate
	RFFilterTransformed = tfa.image.translate(RFFilterTransformed, centerCoordinates)		#https://www.tensorflow.org/addons/api_docs/python/tfa/image/translate
	RFFilterTransformed = tf.image.resize(RFFilterTransformed, axesLength)	#https://www.tensorflow.org/api_docs/python/tf/image/resize
	RFFilterTransformed = tf.squeeze(RFFilterTransformed)
	return RFFilterTransformed
		
def rotateRFFilterTF(RFFilter, RFProperties):
	return rotateRFFilterTF(-RFProperties.angle)
def rotateRFFilterTF(RFFilter, angle):
	RFFilter = tf.expand_dims(RFFilter, 0)	#add extra dimension for num_images
	return RFFilterNormalised
		


def getFilterDimensions(resolutionIndex):
	resolutionFactor, resolutionFactorReverse, imageSize = getImageDimensionsR(resolutionIndex)
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
	
def getImageDimensionsR(resolutionIndex):

	resolutionIndexReverse = numberOfResolutions-resolutionIndex+1
	resolutionFactor = 2**resolutionIndexReverse
	resolutionFactorReverse = 2**resolutionIndex
	resolutionFactorInverse = 1.0/(resolutionFactor)
	#print("resolutionIndex = ", resolutionIndex, ", resolutionFactor = ", resolutionFactor)

	imageSize = (int(imageWidthBase*resolutionFactorInverse), int(imageHeightBase*resolutionFactorInverse))
	
	return resolutionFactor, resolutionFactorReverse, imageSize

def allFilterCoordinatesWithinImage(centerCoordinates, filterRadius, imageSize):
	result = False
	imageSegmentStart = (centerCoordinates[0]-filterRadius, centerCoordinates[1]-filterRadius)
	imageSegmentEnd = (centerCoordinates[0]+filterRadius, centerCoordinates[1]+filterRadius)
	if(imageSegmentStart[0]>=0 and imageSegmentStart[1]>=0 and imageSegmentEnd[0]<imageSize[0] and imageSegmentEnd[1]<imageSize[1]):
		result = True
	return result, imageSegmentStart, imageSegmentEnd


	
@click.command()
@click.argument('inputimagefilename')

def main(inputimagefilename):
	createRFhierarchyAccelerated(inputimagefilename)
	#ATORtf_detectEllipses.main(inputimagefilename)

if __name__ == "__main__":
	main()
	
