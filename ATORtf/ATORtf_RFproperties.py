"""ATORtf_RFproperties.py

# Author:
Richard Bruce Baxter - Copyright (c) 2021 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
See ATORtf.tf

# Usage:
See ATORtf.tf

# Description:
ATORtf RF Properties - RF Properties transformations (primitive space: ellipse or tri/artificial ellipse)

"""

import tensorflow as tf
import numpy as np
import cv2
import copy

import ATORtf_ellipseProperties
import ATORtf_operations


RFtypeEllipse = 1
RFtypeTri = 2

RFfeatureTypeEllipse = 1
RFfeatureTypeCircle = 2
RFfeatureTypePoint = 3
RFfeatureTypeCorner = 4

maximumAxisLengthMultiplierDefault = 4

minimumEllipseAxisLength = 1	#1
receptiveFieldOpponencyAreaFactorEllipse = 2.0	#the radius of the opponency/negative (-1) receptive field compared to the additive (+) receptive field

lowResultFilterPosition = True

#supportFractionalRFdrawSize = False	#floats unsupported by opencv ellipse draw - requires large draw, then resize down (interpolation)


class RFpropertiesClass(ATORtf_ellipseProperties.EllipsePropertiesClass):
	def __init__(self, resolutionIndex, resolutionFactor, imageSize, RFtype, centerCoordinates, axesLength, angle, colour):
		
		self.resolutionIndex = resolutionIndex
		self.resolutionFactor = resolutionFactor
		self.imageSize = imageSize
		
		self.RFtype = RFtype
		super().__init__(centerCoordinates, axesLength, angle, colour)
		
		if(RFtype == RFtypeTri):
			self.vertexCoordinatesRelative = deriveTriVertexCoordinatesFromArtificialEllipseProperties(axesLength, angle)
		
		self.isColourFilter = True
		self.numberOfDimensions = 2	#currently only support 2D data (not ellipses in 3D space or ellipsoids in 3D space)
		self.filterIndex = None
		self.imageSegmentIndex = None

def deriveTriVertexCoordinatesFromArtificialEllipseProperties(axesLength, angle):	
	#https://docs.opencv.org/4.5.3/d6/d6e/group__imgproc__draw.html#ga28b2267d35786f5f890ca167236cbc69
		#tri is oriented towards the first axis, at angle
		#axesLength	Half of the size of the ellipse main axes. 
		#angle	Ellipse rotation angle in degrees. 
	angle1 = angle
	angle2 = angle+90
	angle3 = angle-90
	vertexCoordinatesRelative1 = ATORtf_operations.calculateRelativePositionGivenAngleAndLength(angle1, axesLength[0])
	vertexCoordinatesRelative2 = ATORtf_operations.calculateRelativePositionGivenAngleAndLength(angle2, axesLength[1])
	vertexCoordinatesRelative3 = ATORtf_operations.calculateRelativePositionGivenAngleAndLength(angle3, axesLength[1])
	#print("vertexCoordinatesRelative1 = ", vertexCoordinatesRelative1)
	#print("vertexCoordinatesRelative2 = ", vertexCoordinatesRelative2)
	#print("vertexCoordinatesRelative3 = ", vertexCoordinatesRelative3)
	vertexCoordinatesRelative = [vertexCoordinatesRelative1, vertexCoordinatesRelative2, vertexCoordinatesRelative3]
	return vertexCoordinatesRelative

#def deriveArtificialEllipsePropertiesFromTriVertexCoordinates(vertexCoordinatesRelative):
#	#axesLength = 
#	#angle = 
#	#colour = 
#	return centerCoordinates, axesLength, angle, colour

def printRFproperties(RFproperties):
	print("printRFproperties: numberOfDimensions = ", RFproperties.numberOfDimensions, ", resolutionIndex = ", RFproperties.resolutionIndex, ", isColourFilter = ", RFproperties.isColourFilter, ", imageSize = ", RFproperties.imageSize)
	if(RFproperties.RFtype == RFtypeEllipse):
		ATORtf_ellipseProperties.printEllipseProperties(RFproperties)
	elif(RFproperties.RFtype == RFtypeTri):
		ATORtf_ellipseProperties.printEllipseProperties(RFproperties)
		#ATORtf_RFtri.printTriProperties(RFproperties.triProperties)
		print("vertexCoordinatesRelative = ", RFproperties.vertexCoordinatesRelative)
	
#this function is used by both ATORtf_RFellipse/ATORtf_RFtri
def drawRF(RFfilterTF, RFpropertiesInside, RFpropertiesOutside, drawFeatureType, drawFeatureOverlay):

	blankArray = np.full((RFpropertiesInside.imageSize[1], RFpropertiesInside.imageSize[0], 1), 0, np.uint8)	#grayscale (or black/white)	#0: black	#or filterSize

	ellipseFilterImageInside = copy.deepcopy(blankArray)
	ellipseFilterImageOutside = copy.deepcopy(blankArray)
 
	RFpropertiesInsideWhite = copy.deepcopy(RFpropertiesInside)
	RFpropertiesInsideWhite.colour = (255, 255, 255)
	
	RFpropertiesOutsideWhite = copy.deepcopy(RFpropertiesOutside)
	RFpropertiesOutsideWhite.colour = (255, 255, 255)
	RFpropertiesInsideBlack = copy.deepcopy(RFpropertiesInside)
	RFpropertiesInsideBlack.colour = (000, 000, 000)
	
	if(drawFeatureType == RFfeatureTypeEllipse):
		print("drawEllipse")
		ATORtf_ellipseProperties.drawEllipse(ellipseFilterImageInside, RFpropertiesInsideWhite, True)

		ATORtf_ellipseProperties.drawEllipse(ellipseFilterImageOutside, RFpropertiesOutsideWhite, True)
		ATORtf_ellipseProperties.drawEllipse(ellipseFilterImageOutside, RFpropertiesInsideBlack, True)	
	elif(drawFeatureType == RFfeatureTypeCircle):	#NOTUSED
		ATORtf_ellipseProperties.drawCircle(ellipseFilterImageInside, RFpropertiesInsideWhite, True)

		ATORtf_ellipseProperties.drawCircle(ellipseFilterImageOutside, RFpropertiesOutsideWhite, True)
		ATORtf_ellipseProperties.drawCircle(ellipseFilterImageOutside, RFpropertiesInsideBlack, True)
	elif(drawFeatureType == RFfeatureTypePoint):
		ATORtf_ellipseProperties.drawPoint(ellipseFilterImageInside, RFpropertiesInsideWhite, True)

		ATORtf_ellipseProperties.drawCircle(ellipseFilterImageOutside, RFpropertiesOutsideWhite, True)
		ATORtf_ellipseProperties.drawPoint(ellipseFilterImageOutside, RFpropertiesInsideBlack, True)	
	elif(drawFeatureType == RFfeatureTypeCorner):	#INCOMPLETE
		ATORtf_ellipseProperties.drawPoint(ellipseFilterImageInside, RFpropertiesInsideWhite, True)

		ATORtf_ellipseProperties.drawRectangle(ellipseFilterImageOutside, RFpropertiesOutsideWhite, True)
		ATORtf_ellipseProperties.drawPoint(ellipseFilterImageOutside, RFpropertiesInsideBlack, True)
	
	#ATORtf_operations.displayImage(ellipseFilterImageInside)
	#ATORtf_operations.displayImage(ellipseFilterImageOutside)
	
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
	RFcolourInside = tf.Variable([RFpropertiesInside.colour[0], RFpropertiesInside.colour[1], RFpropertiesInside.colour[2]], dtype=tf.float32)
	RFcolourInside = ATORtf_operations.expandDimsN(RFcolourInside, RFpropertiesInside.numberOfDimensions, axis=0)
	insideImageTF = tf.multiply(insideImageTF, RFcolourInside)
	
	#outsideImageTF = tf.expand_dims(outsideImageTF, axis=2)
	multiples = tf.constant([1,1,3], tf.int32)	#for 2D data only
	outsideImageTF = tf.tile(outsideImageTF, multiples)
	#print(outsideImageTF.shape)
	RFcolourOutside = tf.Variable([RFpropertiesOutside.colour[0], RFpropertiesOutside.colour[1], RFpropertiesOutside.colour[2]], dtype=tf.float32)
	RFcolourOutside = ATORtf_operations.expandDimsN(RFcolourOutside, RFpropertiesOutside.numberOfDimensions, axis=0)
	outsideImageTF = tf.multiply(outsideImageTF, RFcolourOutside)
	
	#print("RFcolourInside = ", RFcolourInside)
	#print("RFcolourOutside = ", RFcolourOutside)
	#print("insideImageTF = ", insideImageTF)
	#print("outsideImageTF = ", outsideImageTF)
	
	#print(RFcolourInside.shape)
	#print(RFcolourOutside.shape)
	#print(insideImageTF.shape)
	#print(outsideImageTF.shape)
		
	if(drawFeatureOverlay):
		#TODO: require a more efficient mask function
		RFfilterTFon = tf.not_equal(RFfilterTF, 0.0)
		insideImageTFon = tf.not_equal(insideImageTF, 0.0)
		outsideImageTFon = tf.not_equal(outsideImageTF, 0.0)
		insideImageTFonOverlap = tf.logical_and(RFfilterTFon, insideImageTFon)
		outsideImageTFonOverlap = tf.logical_and(RFfilterTFon, outsideImageTFon)
		insideImageTFonMask = tf.cast(tf.logical_not(insideImageTFonOverlap), tf.float32)
		outsideImageTFonMask = tf.cast(tf.logical_not(outsideImageTFonOverlap), tf.float32)
		insideImageTF = tf.multiply(insideImageTF, insideImageTFonMask)
		outsideImageTF = tf.multiply(outsideImageTF, outsideImageTFonMask)
		RFfilterTF = tf.add(RFfilterTF, insideImageTF)
		RFfilterTF = tf.add(RFfilterTF, outsideImageTF)
		#RFfilterTF = tf.maximum(RFfilterTF, insideImageTF)
		#RFfilterTF = tf.minimum(RFfilterTF, outsideImageTF)	
	else:
		RFfilterTF = tf.add(RFfilterTF, insideImageTF)
		RFfilterTF = tf.add(RFfilterTF, outsideImageTF)
	
	return RFfilterTF

def generateRFtransformedProperties(neuronComponent, RFpropertiesParent):
	if(RFpropertiesParent.numberOfDimensions == 2):
		return generateRFtransformedProperties2D(neuronComponent, RFpropertiesParent)
	elif(RFpropertiesParent.numberOfDimensions == 3):
		return generateRFtransformedProperties3D(neuronComponent, RFpropertiesParent)
		
def generateRFtransformedProperties2D(neuronComponent, RFpropertiesParent):
	RFtransformedProperties = copy.copy(neuronComponent.RFproperties)
	RFtransformedProperties.centerCoordinates = transformPoint2D(neuronComponent.RFproperties.centerCoordinates, RFpropertiesParent)
	endCoordinates = calculateEndCoordinatesPosition2D(neuronComponent)
	endCoordinates = transformPoint2D(endCoordinates, RFpropertiesParent)
	RFtransformedProperties.axesLength = ATORtf_operations.calculateDistance2D(RFtransformedProperties.centerCoordinates, endCoordinates)
	RFtransformedProperties.angle = neuronComponent.RFproperties.angle-RFpropertiesParent.angle
	return RFtransformedProperties
		
def generateRFtransformedProperties3D(neuronComponent, RFpropertiesParent):
	RFtransformedProperties = copy.copy(neuronComponent.RFproperties)
	RFtransformedProperties.centerCoordinates = transformPoint3D(neuronComponent.RFproperties.centerCoordinates, RFpropertiesParent)
	endCoordinates = calculateEndCoordinatesPosition3D(neuronComponent)
	endCoordinates = transformPoint3D(endCoordinates, RFpropertiesParent)
	RFtransformedProperties.axesLength = ATORtf_operations.calculateDistance3D(RFtransformedProperties.centerCoordinates, endCoordinates)
	RFtransformedProperties.angle = ((neuronComponent.RFproperties.angle[0]-RFpropertiesParent.angle[0]), (neuronComponent.RFproperties.angle[1]-RFpropertiesParent.angle[1]))
	return RFtransformedProperties

def transformPoint2D(coordinates, RFpropertiesParent):
	coordinatesTransformed = (coordinates[0]-RFpropertiesParent.centerCoordinates[0], coordinates[1]-RFpropertiesParent.centerCoordinates[1])
	coordinatesRelativeAfterRotation = ATORtf_operations.calculateRelativePosition2D(RFpropertiesParent.angle, RFpropertiesParent.axesLength[0])
	coordinatesTransformed = (coordinatesTransformed[0]-coordinatesRelativeAfterRotation[0], coordinatesTransformed[1]-coordinatesRelativeAfterRotation[1])	#CHECKTHIS: + or -
	coordinatesTransformed = (coordinates[0]/RFpropertiesParent.axesLength[0], coordinates[1]/RFpropertiesParent.axesLength[1])
	return coordinatesTransformed
	
def transformPoint3D(coordinates, RFpropertiesParent):
	coordinatesTransformed = (coordinates[0]-RFpropertiesParent.centerCoordinates[0], coordinates[1]-RFpropertiesParent.centerCoordinates[1], coordinates[2]-RFpropertiesParent.centerCoordinates[2])
	coordinatesRelativeAfterRotation = ATORtf_operations.calculateRelativePosition3D(RFpropertiesParent.angle, RFpropertiesParent.axesLength[0])
	coordinatesTransformed = (coordinatesTransformed[0]-coordinatesRelativeAfterRotation[0], coordinatesTransformed[1]-coordinatesRelativeAfterRotation[1], coordinatesTransformed[2]-coordinatesRelativeAfterRotation[2])	#CHECKTHIS: + or -
	coordinatesTransformed = (coordinates[0]/RFpropertiesParent.axesLength[0], coordinates[1]/RFpropertiesParent.axesLength[1], coordinates[2]/RFpropertiesParent.axesLength[2])
	return coordinatesTransformed
		
def calculateEndCoordinatesPosition2D(neuronComponent):
	endCoordinatesRelativeToCentreCoordinates = ATORtf_operations.calculateRelativePosition2D(neuronComponent.RFproperties.angle, neuronComponent.RFproperties.axesLength[0])
	endCoordinates = (neuronComponent.RFproperties.centerCoordinates[0]+endCoordinatesRelativeToCentreCoordinates[0], neuronComponent.RFproperties.centerCoordinates[1]+endCoordinatesRelativeToCentreCoordinates[1])	#CHECKTHIS: + or -
	return endCoordinates
	
def calculateEndCoordinatesPosition3D(neuronComponent):
	endCoordinatesRelativeToCentreCoordinates = ATORtf_operations.calculateRelativePosition3D(neuronComponent.RFproperties.angle, neuronComponent.RFproperties.axesLength)
	endCoordinates = (neuronComponent.RFproperties.centerCoordinates[0]+endCoordinatesRelativeToCentreCoordinates[0], neuronComponent.RFproperties.centerCoordinates[1]+endCoordinatesRelativeToCentreCoordinates[1], neuronComponent.RFproperties.centerCoordinates[2]+endCoordinatesRelativeToCentreCoordinates[2])	#CHECKTHIS: + or -
	return endCoordinates

def saveRFFilterImage(RFfilter, RFfilterImageFilename):
	#print(RFfilter)
	RFfilterMask =  tf.cast(tf.not_equal(RFfilter, 0.0), tf.float32)
	RFfilterImage = tf.add(RFfilter, 3.0)
	RFfilterImage = tf.divide(RFfilterImage, 4.0)
	RFfilterImage = tf.multiply(RFfilterImage, RFfilterMask)
	RFfilterImage = tf.multiply(RFfilterImage, ATORtf_operations.rgbMaxValue)
	RFfilterUint8 = tf.cast(RFfilterImage, tf.uint8)
	RFfilterNP = RFfilterUint8.numpy()	#FORMAT: np.full((inputImageHeight, inputImageWidth, 3), 255, np.uint8)
	ATORtf_operations.saveImage(RFfilterImageFilename, RFfilterNP)

#use common filter dimensions across all filter types (for ellipse/tri)
def getFilterDimensions(resolutionProperties, maximumAxisLengthMultiplier=maximumAxisLengthMultiplierDefault, receptiveFieldOpponencyAreaFactor=receptiveFieldOpponencyAreaFactorEllipse):
	#reduce max size of ellipse at each res
	axesLengthMax1 = minimumEllipseAxisLength*maximumAxisLengthMultiplier
	axesLengthMax2 = minimumEllipseAxisLength*maximumAxisLengthMultiplier
	filterRadius = int(max(axesLengthMax1*receptiveFieldOpponencyAreaFactor, axesLengthMax2*receptiveFieldOpponencyAreaFactor))
	#axesLengthMax1 = int(resolutionProperties.imageSize[0]//resolutionProperties.resolutionFactorReverse * 1 / 2)	#CHECKTHIS
	#axesLengthMax2 = int(resolutionProperties.imageSize[1]//resolutionProperties.resolutionFactorReverse * 1 / 2)	#CHECKTHIS
	#filterRadius = int(max(axesLengthMax1*receptiveFieldOpponencyAreaFactor, axesLengthMax2*receptiveFieldOpponencyAreaFactor)/2)
	filterSize = (int(filterRadius*2), int(filterRadius*2))	#x/y dimensions are identical
	axesLengthMax = (axesLengthMax1, axesLengthMax2)
	
	#print("resolutionFactorReverse = ", resolutionProperties.resolutionFactorReverse)
	#print("resolutionFactor = ", resolutionProperties.imageSize)
	#print("axesLengthMax = ", axesLengthMax)
	
	return axesLengthMax, filterRadius, filterSize	
	
	
