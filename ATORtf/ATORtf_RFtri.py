"""ATORtf_RFtri.py

# Author:
Richard Bruce Baxter - Copyright (c) 2021 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
See ATORtf.tf

# Usage:
See ATORtf.tf

# Description:
ATORtf RF Tri - generate tri (as represented by 3 feature points) receptive fields

"""

import tensorflow as tf
import numpy as np
import cv2
import copy

import ATORtf_RFproperties
import ATORtf_operations

#RF Tri can be applied to luminosity or contrast maps

pointFeatureRFinsideRadius = 1 	#floats unsupported by opencv ellipse draw: 0.5	#CHECKTHIS - requires calibration
pointFeatureRFopponencyArea = 2	#floats unsupported by opencv ellipse draw: 0.5	#CHECKTHIS - requires calibration

generatePointFeatureCorners = True
if(generatePointFeatureCorners):
	minimumCornerOpponencyPosition = -pointFeatureRFinsideRadius	#CHECKTHIS
	maximumCornerOpponencyPosition = pointFeatureRFinsideRadius		#CHECKTHIS
	cornerOpponencyPositionResolution = 1	#CHECKTHIS
else:
	#only generate simple point features;
	minimumCornerOpponencyPosition = 0
	maximumCornerOpponencyPosition = 0
	cornerOpponencyPositionResolution = 1

#match ATORtf_RFellipse algorithm;
matchRFellipseAlgorithm = True	#else create equilateral triangle snapshot

#match ATORtf_RFellipse algorithm;
RFnormaliseLocalEquilateralTriangle = True

#match ATORtf_RFellipse algorithm;
ellipseNormalisedAngle = 0.0
ellipseNormalisedCentreCoordinates = 0.0 
ellipseNormalisedAxesLength = 1.0

#match ATORtf_RFellipse algorithm;
minimumEllipseLength = 2
ellipseAxesLengthResolution = 1	#pixels (at resolution r)
ellipseAngleResolution = 30	#degrees
ellipseColourResolution = 64	#bits

#match ATORtf_RFellipse algorithm;
if(matchRFellipseAlgorithm):
	filterSnapshotArea = 2.0	#temporarily set to ATORtf_RFellipse.receptiveFieldOpponencyArea
else:
	filterSnapshotArea = 1.0	#could be set to 1.0 to match ATOR specification
	
def printTriProperties(triProperties):
	print("vertexCoordinatesRelative = ", triProperties.vertexCoordinatesRelative)

#~match ATORtf_RFellipse algorithm;
def normaliseGlobalTriProperties(ellipseProperties, resolutionFactor):
	resolutionFactor = ellipseProperties.resolutionFactor
	ellipsePropertiesNormalised = copy.deepcopy(ellipseProperties) 
	ellipsePropertiesNormalised.centerCoordinates = (ellipsePropertiesNormalised.centerCoordinates[0]*resolutionFactor, ellipsePropertiesNormalised.centerCoordinates[1]*resolutionFactor)
	ellipsePropertiesNormalised.axesLength = (ellipsePropertiesNormalised.axesLength[0]*resolutionFactor, ellipsePropertiesNormalised.axesLength[1]*resolutionFactor)
	return ellipsePropertiesNormalised

#match ATORtf_RFellipse algorithm;	
def normaliseLocalTriProperties(RFproperties):
	#normalise ellipse respect to major/minor ellipticity axis orientation (WRT self)
	RFpropertiesNormalised = copy.deepcopy(RFproperties)
	RFpropertiesNormalised.angle = ellipseNormalisedAngle	#CHECKTHIS
	RFpropertiesNormalised.centerCoordinates = (ellipseNormalisedCentreCoordinates, ellipseNormalisedCentreCoordinates, ellipseNormalisedCentreCoordinates)
	if(RFnormaliseLocalEquilateralTriangle):
		RFpropertiesNormalised.axesLength = ATORtf_operations.getEquilateralTriangleAxesLength(ellipseNormalisedAxesLength)
	else:
		RFpropertiesNormalised.axesLength = (ellipseNormalisedAxesLength, ellipseNormalisedAxesLength)
		
	return RFpropertiesNormalised

#match ATORtf_RFellipse algorithm;	
def calculateFilterPixels(filterSize, numberOfDimensions):
	internalFilterSize = getInternalFilterSize(filterSize, numberOfDimensions)	#CHECKTHIS: only consider contribution of positive (additive) pixels
	#print("internalFilterSize = ", internalFilterSize)
	
	if(numberOfDimensions == 2):
		numberOfFilterPixels = internalFilterSize[0]*internalFilterSize[1]
	elif(numberOfDimensions == 3):
		numberOfFilterPixels = internalFilterSize[0]*internalFilterSize[1]	#CHECKTHIS
	#print("numberOfFilterPixels = ", numberOfFilterPixels)
	return numberOfFilterPixels

#match ATORtf_RFellipse algorithm;	
def getInternalFilterSize(filterSize, numberOfDimensions):
	if(numberOfDimensions == 2):
		internalFilterSize = (pointFeatureRFinsideRadius*2, pointFeatureRFinsideRadius*2)
	elif(numberOfDimensions == 3):
		internalFilterSize = (pointFeatureRFinsideRadius*2, pointFeatureRFinsideRadius*2)	#CHECKTHIS
	return internalFilterSize
	
		
def generateRFfiltersTri(resolutionProperties, RFfiltersList, RFfiltersPropertiesList):
	#generate filter types
	
	#2D code;
	
	#filters are generated based on human magnocellular/parvocellular/koniocellular wavelength discrimination in LGN and VX (double/opponent receptive fields)
	
	#magnocellular filters (monochromatic);
	colourH = (255, 255, 255)	#high
	colourL = (-255, -255, -255)	#low
	RFfiltersHL, RFpropertiesHL = generateRotationalInvariantRFfilters(resolutionProperties, False, colourH, colourL)
	
	RFfiltersList.append(RFfiltersHL)

	RFfiltersPropertiesList.append(RFpropertiesHL)

def generateRotationalInvariantRFfilters(resolutionProperties, isColourFilter, filterInsideColour, filterOutsideColour):
	
	RFfiltersList2 = []
	RFfiltersPropertiesList2 = []
	
	#FUTURE: consider storing filters in n dimensional array and finding local minima of filter matches across all dimensions

	#reduce max size of ellipse at each res
	axesLengthMax, filterRadius, filterSize = getFilterDimensions(resolutionProperties)
	
	#print("axesLengthMax = ", axesLengthMax)
	
	for axesLength1 in range(minimumEllipseLength, axesLengthMax[0]+1, ellipseAxesLengthResolution):
		for axesLength2 in range(minimumEllipseLength, axesLengthMax[1]+1, ellipseAxesLengthResolution):
			for angle in range(0, 360, ellipseAngleResolution):	#degrees
				for corner1OpponencyPosition1 in range(minimumCornerOpponencyPosition, maximumCornerOpponencyPosition+1, cornerOpponencyPositionResolution):
					for corner1OpponencyPosition2 in range(minimumCornerOpponencyPosition, maximumCornerOpponencyPosition+1, cornerOpponencyPositionResolution):
						for corner2OpponencyPosition1 in range(minimumCornerOpponencyPosition, maximumCornerOpponencyPosition+1, cornerOpponencyPositionResolution):
							for corner2OpponencyPosition2 in range(minimumCornerOpponencyPosition, maximumCornerOpponencyPosition+1, cornerOpponencyPositionResolution):
								for corner3OpponencyPosition1 in range(minimumCornerOpponencyPosition, maximumCornerOpponencyPosition+1, cornerOpponencyPositionResolution):
									for corner3OpponencyPosition2 in range(minimumCornerOpponencyPosition, maximumCornerOpponencyPosition+1, cornerOpponencyPositionResolution):
									
										axesLength = (axesLength1, axesLength2)
 
										axesLengthInside = (pointFeatureRFinsideRadius, pointFeatureRFinsideRadius)	#estimation
										axesLengthOutside = (pointFeatureRFinsideRadius*pointFeatureRFopponencyArea, pointFeatureRFinsideRadius*pointFeatureRFopponencyArea)	#estimation
										angleInside = 0.0	#circular	
										angleOutside = 0.0	#circular
										filterCenterCoordinates = (0, 0)

										RFpropertiesInside = ATORtf_RFproperties.RFpropertiesClass(resolutionProperties.resolutionIndex, resolutionProperties.resolutionFactor, filterSize, ATORtf_RFproperties.RFtypeEllipse, filterCenterCoordinates, axesLengthInside, angleInside, filterInsideColour)
										RFpropertiesOutside = ATORtf_RFproperties.RFpropertiesClass(resolutionProperties.resolutionIndex, resolutionProperties.resolutionFactor, filterSize, ATORtf_RFproperties.RFtypeEllipse, filterCenterCoordinates, axesLengthOutside, angleOutside, filterOutsideColour)
										RFpropertiesInside.isColourFilter = isColourFilter
										RFpropertiesOutside.isColourFilter = isColourFilter

										#number of corner filters;
										#3^6 = 729 filters (or 9^3)
										
										#example corner point filter;
										#++-0
										#++--
										#----
										#0--0
										
										vertexCoordinatesRelative = ATORtf_RFproperties.deriveTriVertexCoordinatesFromArtificialEllipseProperties(axesLength, angle)
										vertexCoordinatesRelativeInside = copy.deepcopy(vertexCoordinatesRelative)
										vertexCoordinatesRelativeOutside = copy.deepcopy(vertexCoordinatesRelative)
										vertexCoordinatesRelativeInside[0][0] = vertexCoordinatesRelativeInside[0][0] + corner1OpponencyPosition1
										vertexCoordinatesRelativeInside[0][1] = vertexCoordinatesRelativeInside[0][1] + corner1OpponencyPosition2
										vertexCoordinatesRelativeInside[1][0] = vertexCoordinatesRelativeInside[1][0] + corner2OpponencyPosition1
										vertexCoordinatesRelativeInside[1][1] = vertexCoordinatesRelativeInside[1][1] + corner2OpponencyPosition2
										vertexCoordinatesRelativeInside[2][0] = vertexCoordinatesRelativeInside[2][0] + corner3OpponencyPosition1
										vertexCoordinatesRelativeInside[2][1] = vertexCoordinatesRelativeInside[2][1] + corner3OpponencyPosition2
										
										RFfilter = generateRFfilter(resolutionProperties.resolutionIndex, isColourFilter, RFpropertiesInside, RFpropertiesOutside, vertexCoordinatesRelativeInside, vertexCoordinatesRelativeOutside)
										RFfiltersList2.append(RFfilter)

										RFproperties = ATORtf_RFproperties.RFpropertiesClass(resolutionProperties.resolutionIndex, resolutionFactor, filterSize, ATORtf_RFproperties.RFtypeTri, filterCenterCoordinates, axesLength, angle, filterInsideColour)
										#RFproperties.centerCoordinates = centerCoordinates 	#centerCoordinates are set after filter is applied to imageSegment
										RFproperties.isColourFilter = isColourFilter
										RFfiltersPropertiesList2.append(RFproperties)	#CHECKTHIS: use RFpropertiesInside not RFpropertiesOutside

										#debug:
										#print(RFfilter.shape)
										if(resolutionProperties.debugVerbose):
											ATORtf_RFproperties.printRFproperties(RFproperties)
										#print("RFfilter = ", RFfilter)

	#create 3D tensor (for hardware accelerated test/application of filters)
	RFfiltersTensor = tf.stack(RFfiltersList2, axis=0)

	return RFfiltersTensor, RFfiltersPropertiesList2
	

def generateRFfilter(resolutionProperties, isColourFilter, RFpropertiesInside, RFpropertiesOutside, vertexCoordinatesRelativeInside, vertexCoordinatesRelativeOutside):

	# RF filter example (RFfilterTF):
	#
	# 0 0 0 0 0 0 0 0 0 0 0 0
	# 0 0 - 0 0 0 0 - 0 0 0 0
	# 0 - + - 0 0 - + - 0 0 0
	# 0 0 - 0 0 0 0 - 0 0 0 0
	# 0 0 0 0 0 0 0 0 0 0 0 0
	# 0 0 0 0 0 0 0 0 0 0 0 0 
	# 0 0 0 0 - 0 0 0 0 0 0 0
	# 0 0 0 - + - 0 0 0 0 0 0
	# 0 0 0 0 - 0 0 0 0 0 0 0
	# 0 0 0 0 0 0 0 0 0 0 0 0 
	#
	# where "-" = -RFcolourOutside [R G B], "+" = +RFcolourInside [R G B], and "0" = [0, 0, 0]
	
	#generate ellipse on blank canvas
	blankArray = np.full((RFpropertiesInside.imageSize[1], RFpropertiesInside.imageSize[0], ATORtf_operations.rgbNumChannels), 0, np.uint8)	#rgb
	RFfilterTF = tf.convert_to_tensor(blankArray, dtype=tf.float32)
	
	RFpropertiesInside1 = copy.deepcopy(RFpropertiesInside)
	RFpropertiesOutside1 = copy.deepcopy(RFpropertiesOutside)
	RFpropertiesInside2 = copy.deepcopy(RFpropertiesInside)
	RFpropertiesOutside2 = copy.deepcopy(RFpropertiesOutside)
	RFpropertiesInside3 = copy.deepcopy(RFpropertiesInside)
	RFpropertiesOutside3 = copy.deepcopy(RFpropertiesOutside)
		
	RFpropertiesInside1.centerCoordinates = (RFpropertiesInside1.centerCoordinates[0]+vertexCoordinatesRelativeInside[0][0], RFpropertiesInside1.centerCoordinates[1]+vertexCoordinatesRelativeInside[0][1])
	RFpropertiesOutside1.centerCoordinates = (RFpropertiesOutside1.centerCoordinates[0]+vertexCoordinatesRelativeOutside[0][0], RFpropertiesOutside1.centerCoordinates[1]+vertexCoordinatesRelativeOutside[0][1])
	RFpropertiesInside2.centerCoordinates = (RFpropertiesInside2.centerCoordinates[0]+vertexCoordinatesRelativeInside[1][0], RFpropertiesInside2.centerCoordinates[1]+vertexCoordinatesRelativeInside[1][1])
	RFpropertiesOutside2.centerCoordinates = (RFpropertiesOutside2.centerCoordinates[0]+vertexCoordinatesRelativeOutside[1][0], RFpropertiesOutside2.centerCoordinates[1]+vertexCoordinatesRelativeOutside[1][1])
	RFpropertiesInside3.centerCoordinates = (RFpropertiesInside3.centerCoordinates[0]+vertexCoordinatesRelativeInside[2][0], RFpropertiesInside3.centerCoordinates[1]+vertexCoordinatesRelativeInside[2][1])
	RFpropertiesOutside3.centerCoordinates = (RFpropertiesOutside3.centerCoordinates[0]+vertexCoordinatesRelativeOutside[2][0], RFpropertiesOutside3.centerCoordinates[1]+vertexCoordinatesRelativeOutside[2][1])
	
	#print("RFpropertiesInside1.centerCoordinates = ", RFpropertiesInside1.centerCoordinates)
	#print("RFpropertiesOutside1.centerCoordinates = ", RFpropertiesOutside1.centerCoordinates)
	
	RFfilterTF = ATORtf_RFproperties.drawRF(blankArray, RFfilterTF, RFpropertiesInside1, RFpropertiesOutside1, False)
	RFfilterTF = ATORtf_RFproperties.drawRF(blankArray, RFfilterTF, RFpropertiesInside2, RFpropertiesOutside2, False)
	RFfilterTF = ATORtf_RFproperties.drawRF(blankArray, RFfilterTF, RFpropertiesInside3, RFpropertiesOutside3, False)
	
	#print("RFfilterTF = ", RFfilterTF)

	if(ATORtf_operations.storeRFfiltersValuesAsFractions):
		RFfilterTF = tf.divide(RFfilterTF, ATORtf_operations.rgbMaxValue)
			
	if(not isColourFilter):
		RFfilterTF = tf.image.rgb_to_grayscale(RFfilterTF)
	
	#print("RFfilterTF.shape = ", RFfilterTF.shape)	
	#print("RFfilterTF = ", RFfilterTF)

	return RFfilterTF
	

def getFilterDimensions(resolutionProperties):
	#reduce max size of ellipse at each res
	axesLengthMax1 = int(resolutionProperties.imageSize[0]//resolutionProperties.resolutionFactorReverse * 1 / 2)	#CHECKTHIS
	axesLengthMax2 = int(resolutionProperties.imageSize[1]//resolutionProperties.resolutionFactorReverse * 1 / 2)	#CHECKTHIS
	filterRadius = int(max(axesLengthMax1*filterSnapshotArea, axesLengthMax2*filterSnapshotArea)/2)	
	filterSize = (int(filterRadius*2), int(filterRadius*2))	#x/y dimensions are identical
	axesLengthMax = (axesLengthMax1, axesLengthMax2)
	
	#print("resolutionFactorReverse = ", resolutionProperties.resolutionFactorReverse)
	#print("resolutionFactor = ", resolutionProperties.imageSize)
	#print("axesLengthMax = ", axesLengthMax)
	
	return resolutionFactor, imageSize, axesLengthMax, filterRadius, filterSize	
	
	
