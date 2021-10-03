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

resolutionIndexFirst = 1
numberOfResolutions = 6	#x; lowest res sample: 1/(2^x)
minimumEllipseLength = 2
ellipseCenterCoordinatesResolution = 1	#pixels (at resolution r)
ellipseAxesLengthResolution = 1	#pixels (at resolution r)
ellipseAngleResolution = 10	#degrees
ellipseColourResolution = 64	#bits
minimumFilterRequirement = 1.0	#CHECKTHIS: calibrate

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
	inputImageRGBTF = tf.convert_to_tensor(inputImageRGB, dtype=tf.float32)
	inputImageGrayTF = tf.convert_to_tensor(inputImageGray, dtype=tf.float32)
	
	inputImageHeight, inputImageWidth, inputImageChannels = inputImage.shape
	print("inputImageHeight = ", inputImageHeight, "inputImageWidth = ", inputImageWidth, ", inputImageChannels = ", inputImageChannels)
	blankArray = np.full((inputImageHeight, inputImageWidth, 3), 255, np.uint8)
	outputImage = blankArray
	
	ATORneuronListAllLayers = []
			
	RFFiltersListAllRes = []	#stores receptive field tensorflow objects (used for hardware accelerated filter detection)
	RFPropertiesListAllRes = []	#stores receptive field ellipse properties (position, size, rotation, colour etc)
	
	#generateRFFilters:
	for resolutionIndex in range(resolutionIndexFirst, numberOfResolutions):
		RFFiltersList, RFPropertiesList = generateRFFilters(resolutionIndex)
		RFFiltersListAllRes.append(RFFiltersList)
		RFPropertiesListAllRes.append(RFPropertiesList)
		
	#applyRFFilters:
	for resolutionIndex in range(resolutionIndexFirst, numberOfResolutions):
		RFFiltersList = RFFiltersListAllRes[resolutionIndex]
		RFPropertiesList = RFPropertiesListAllRes[resolutionIndex]
		applyRFFiltersList(inputImageRGBTF, inputImageGrayTF, resolutionIndex, RFFiltersList, RFPropertiesList)
		
def normaliseRFFilter(RFFilter, RFProperties):
	#normalise ellipse respect to major/minor ellipticity axis orientation (WRT self)
	RFFilterNormalised = transformRFFilterTF(RFFilter, RFProperties) 
	return RFFilterNormalised

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
	RFTransformedProperties, RFTransformationProperties = generateRFTransformedPropertiesWRTparent(neuronComponent, RFPropertiesParent)
	if(debugSaveRFFilters):
		neuronComponent.RFFilterNormalisedWRTparent = transformRFFilterTF(neuronComponent.RFFilter, RFTransformationProperties)	#not required: for debugging only
	neuronComponent.RFPropertiesWRTparent = RFTransformedProperties

#CHECKTHIS: requires coding;
def generateRFTransformedProperties(neuronComponent, RFPropertiesParent):
	
	RFTransformedProperties = None
	RFTransformationProperties = None
	
	#RFTransformationProperties = copy.copy(neuronComponent.RFProperties)	#used for simplicity (not all properties are required)
	#RFTransformationProperties.centerCoordinates = (neuronComponent.centerCoordinates[0]-RFPropertiesParent.centerCoordinates[0], neuronComponent.centerCoordinates[1]-RFPropertiesParent.centerCoordinates[1])	#2D code only
	#RFTransformationProperties.axesLength = (neuronComponent.axesLength[0]/RFPropertiesParent.axesLength[0], neuronComponent.axesLength[1]/RFPropertiesParent.axesLength[1])	#2D code only
	#RFTransformationProperties.angle = 

	#RFTransformedProperties = copy.copy(neuronComponent.RFProperties)
	#RFTransformedProperties.centerCoordinates = 
	#RFTransformedProperties.axesLength = 
	#RFTransformedProperties.angle = 
		
	return RFTransformedProperties, RFTransformationProperties
	
def transformRFFilterTF(RFFilter, RFProperties):
	centerCoordinates = [-RFProperties.centerCoordinates[0], -RFProperties.centerCoordinates[1]]
	axesLength = [1.0/RFProperties.axesLength[0], 1.0/RFProperties.axesLength[1]]
	angle = -RFProperties.angle
	return transformRFFilterTF(RFFilter, centerCoordinates, axesLength, angle)

def transformRFFilterTF(RFFilter, centerCoordinates, axesLength, angle):
	#CHECKTHIS: 2D code only;
	RFFilterTransformed = tf.expand_dims(RFFilterTransformed, 0)	#add extra dimension for num_images
	RFFilterTransformed = tfa.image.rotate(RFFilterTransformed, ATORtf_operations.convertDegreesToRadians(angle))		#https://www.tensorflow.org/addons/api_docs/python/tfa/image/rotate
	RFFilterTransformed = tfa.image.translate(RFFilterTransformed, centerCoordinates)		#https://www.tensorflow.org/addons/api_docs/python/tfa/image/translate
	RFFilterTransformed = tf.image.resize(RFFilterTransformed, axesLength)	#https://www.tensorflow.org/api_docs/python/tf/image/resize
	RFFilterTransformed = tf.squeeze(RFFilterTransformed)
	return RFFilterTransformed
		
def rotateRFFilterTF(RFFilter, RFProperties):
	return rotateRFFilterTF(-RFProperties.angle)
def rotateRFFilterTF(RFFilter, angle):
	RFFilter = tf.expand_dims(RFFilter, 0)	#add extra dimension for num_images
	return RFFilterNormalised
		

def applyRFFiltersList(inputImageRGBTF, inputImageGrayTF, resolutionIndex, RFFiltersList, RFPropertiesList, ATORneuronListAllLayers):
	
	ATORneuronList = []	#for resolutionIndex
	
	for RFlistIndex1 in range(len(RFPropertiesList)):	#or RFFiltersList
	
		RFFiltersTensor = RFFiltersList[RFlistIndex1]
		RFPropertiesList2 = RFPropertiesList[RFlistIndex1]
		isColourFilter = RFPropertiesList2[0].isColourFilter
		numberOfDimensions = RFPropertiesList2[0].numberOfDimensions
		if(isColourFilter):
			filterApplicationResultThreshold = applyRFFilters(inputImageRGBTF, RFFiltersTensor, numberOfDimensions, RFPropertiesList2)		
		else:
			filterApplicationResultThreshold = applyRFFilters(inputImageGrayTF, RFFiltersTensor, numberOfDimensions, RFPropertiesList2)
		
		filterApplicationResultNP = tf.make_ndarray(filterApplicationResult)
		filterApplicationResultList = filterApplicationResultNP.tolist()
		for filterApplicationResult, RFlistIndex2 in enumerate(filterApplicationResultList):
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
						
def applyRFFilters(inputImageTF, RFFiltersTensor, numberOfDimensions):
	#perform convolution
	inputImageTF = tf.expand_dims(inputImageTF, axis=0)	#add filter dimension
	filterApplicationResult = tf.multiply(inputImageRGBTF, RFFiltersTensor)
	if(numberOfDimensions == 2):
		imageDataAxes = [1, 2, 3]	#x, y, c	
	else:
		imageDataAxes = [1, 2, 3, 4]	#x, y, d/z, c	
	filterApplicationResult = tf.math.reduce_sum(filterApplicationResult, axis=imageDataAxes)	
	#filterApplicationResultThreshold = tf.greater(filterApplicationResult, minimumFilterRequirement)
	return filterApplicationResult	#filterApplicationResultThreshold
	
def generateRFFilters(resolutionIndex):

	#filters are generated based on human magnocellular/parvocellular pathway wavelength discrimination
	
	RFFiltersList = []
	RFPropertiesList = []
	
	#magnocellular filters (monochromatic);
	colourH = (255, 255, 255)	#high
	colourL = (000, 000, 000)	#low
	RFFiltersHL, RFPropertiesHL = generateRotationalInvariantRFFilters(resolutionIndex, False, colourH, colourL)
	RFFiltersLH, RFPropertiesLH = generateRotationalInvariantRFFilters(resolutionIndex, False, colourL, colourH)
	
	#parvocellular filters (based on 2 cardinal colour axes; ~red-~green, ~blue-~yellow);
	colourR = (255, 000, 000)	#red
	colourG = (000, 255, 000)	#green
	colourB = (000, 000, 255)	#blue
	colourY = (255, 255, 000)	#yellow
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

	RFPropertiesList.append(RFPropertiesHL)
	RFPropertiesList.append(RFPropertiesLH)
	RFPropertiesList.append(RFPropertiesRG)
	RFPropertiesList.append(RFPropertiesGR)
	RFPropertiesList.append(RFPropertiesBY)
	RFPropertiesList.append(RFPropertiesYB)
		
	return RFFiltersList, RFPropertiesList

def generateRotationalInvariantRFFilters(resolutionIndex, isColourFilter, filterInsideColour, filterOutsideColour):
	
	RFFiltersList2 = []
	RFPropertiesList2 = []
	
	#FUTURE: consider storing filters in n dimensional array and finding local minima of filter matches across all dimensions

	resolutionFactor, resolutionFactorReverse, imageWidthR, imageHeightR = getImageDimensionsR(resolutionIndex)
	
	#reduce max size of ellipse at each res
	#axesLengthMax1 = imageWidthR
	#axesLengthMax2 = imageHeightR
	axesLengthMax1 = imageWidthR//resolutionFactorReverse * 4	#CHECKTHIS
	axesLengthMax2 = imageHeightR//resolutionFactorReverse * 4	#CHECKTHIS
	print("axesLengthMax1 = ", axesLengthMax1, ", axesLengthMax2 = ", axesLengthMax2)
			
	for centerCoordinates1 in range(0, imageWidthR, ellipseCenterCoordinatesResolution):
		for centerCoordinates2 in range(0, imageHeightR, ellipseCenterCoordinatesResolution):
			for axesLength1 in range(minimumEllipseLength, axesLengthMax1, ellipseAxesLengthResolution):
				for axesLength2 in range(minimumEllipseLength, axesLengthMax2, ellipseAxesLengthResolution):
					for angle in range(0, 360, ellipseAngleResolution):	#degrees
					
						centerCoordinates = (centerCoordinates1, centerCoordinates2)
						axesLength = (axesLength1, axesLength2)
						
						RFPropertiesInside = ATORtf_ellipseProperties.EllipseProperties(resolutionIndex, resolutionFactor, imageWidthR, imageHeightR, centerCoordinates, axesLength, angle, filterInsideColour)
						RFPropertiesOutside = ATORtf_ellipseProperties.EllipseProperties(resolutionIndex, resolutionFactor, imageWidthR, imageHeightR, centerCoordinates, axesLength, angle, filterOutsideColour)
						RFPropertiesInside.isColourFilter = isColourFilter
						RFPropertiesOutside. isColourFilter = isColourFilter
						
						RFFilter = generateRFFilter(resolutionIndex, isColourFilter, RFPropertiesInside, RFPropertiesOutside)
						RFFiltersList2.append(RFFilter)
						RFPropertiesList2.append(RFPropertiesInside)	#CHECKTHIS: use RFPropertiesInside not RFPropertiesOutside
	
	#create 3D tensor (for hardware accelerated test/application of filters)
	RFFiltersTensor = tf.stack(RFFiltersList2, axis=0)

	return RFFiltersTensor, RFPropertiesList2
						
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
	resolutionFactor, resolutionFactorReverse, imageHeightR, imageWidthR = getImageDimensionsR(resolutionIndex)
	blankArray = np.full((imageHeightR, imageWidthR, 1), 0, np.uint8)	#grayscale (or black/white)	#0: black
	
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
		
	#add colour channels;
	inputImageTF = tf.expand_dims(inputImageTF, axis=2)
	multiples = tf.constant([1,1,1,3], tf.int32)	#for 2D data only
	inputImageTF = tf.tile(inputImageTF, multiples)
	RFColourInside = tf.constant([RFPropertiesInside.colour[0], RFPropertiesInside.colour[1], RFPropertiesInside.colour[2]], dtype=tf.float32)
	RFColourInside = ATORtf_operations.expandDimsN(RFColourInside, RFPropertiesInside.numberOfDimensions, axis=0)
	inputImageTF = tf.multiply(inputImageTF, RFColourInside)
	
	outsideImageTF = tf.expand_dims(outsideImageTF, axis=2)
	multiples = tf.constant([1,1,1,3], tf.int32)	#for 2D data only
	outsideImageTF = tf.tile(outsideImageTF, multiples)
	RFColourOutside = tf.constant([RFPropertiesOutside.colour[0], RFPropertiesOutside.colour[1], RFPropertiesOutside.colour[2]], dtype=tf.float32)
	RFColourOutside = ATORtf_operations.expandDimsN(RFColourOutside, RFPropertiesOutside.numberOfDimensions, axis=0)
	outsideImageTF = tf.multiply(outsideImageTF, RFColourOutside)
	
	RFFilterTF = tf.convert_to_tensor(blankArray, dtype=tf.float32)
	RFFilterTF = tf.add(RFFilterTF, inputImageTF)
	RFFilterTF = tf.add(RFFilterTF, outsideImageTF)
		
	return RFFilterTF
	
def getImageDimensionsR(resolutionIndex):

	resolutionIndexReverse = numberOfResolutions-resolutionIndex+1
	resolutionFactor = 2**resolutionIndexReverse
	resolutionFactorReverse = 2**resolutionIndex
	resolutionFactorInverse = 1.0/(resolutionFactor)
	#print("resolutionIndex = ", resolutionIndex, ", resolutionFactor = ", resolutionFactor)

	imageWidthR = int(imageWidthBase*resolutionFactorInverse)
	imageHeightR = int(imageHeightBase*resolutionFactorInverse) 
	
	return resolutionFactor, resolutionFactorReverse, imageWidthR, imageHeightR 

	
@click.command()
@click.argument('inputimagefilename')

def main(inputimagefilename):
	createRFhierarchyAccelerated(inputimagefilename)
	#ATORtf_detectEllipses.main(inputimagefilename)

if __name__ == "__main__":
	main()
	
