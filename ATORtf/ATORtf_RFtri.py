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

debugSmallIterations = False

pointFeatureRFinsideRadius = 0.5 	
pointFeatureRFopponencyAreaFactor = 2	#floats unsupported by opencv ellipse draw

generatePointFeatureCorners = True
if(generatePointFeatureCorners):
	minimumCornerOpponencyPosition = -1	#CHECKTHIS
	maximumCornerOpponencyPosition = 1		#CHECKTHIS
	cornerOpponencyPositionResolution = 1	#CHECKTHIS
else:
	#only generate simple point features;
	minimumCornerOpponencyPosition = 0
	maximumCornerOpponencyPosition = 0
	cornerOpponencyPositionResolution = 1

#match ATORtf_RFellipse algorithm;
matchRFellipseAlgorithm = False

#match ATORtf_RFellipse algorithm;
RFnormaliseLocalEquilateralTriangle = True

#match ATORtf_RFellipse algorithm;
ellipseNormalisedAngle = 0.0
ellipseNormalisedCentreCoordinates = 0.0 
ellipseNormalisedAxesLength = 1.0

#match ATORtf_RFellipse algorithm;
# ellipse axesLength definition (based on https://docs.opencv.org/4.5.3/d6/d6e/group__imgproc__draw.html#ga28b2267d35786f5f890ca167236cbc69)
#  ___
# / | \ minimumEllipseAxisLength1 | (axis1: axes width)
# | --| minimumEllipseAxisLength2 - (axis2: axis height)
# \___/
#
if(matchRFellipseAlgorithm):
	receptiveFieldOpponencyAreaFactorTri = ATORtf_RFproperties.receptiveFieldOpponencyAreaFactorEllipse	#2.0
	maximumAxisLengthMultiplierTri = 1
	maximumAxisLengthMultiplier = ATORtf_RFproperties.maximumAxisLengthMultiplierDefault
	minimumEllipseAxisLength1 = ATORtf_RFproperties.minimumEllipseAxisLength*2	#minimum elongation is required	#can be set to 1 as (axesLength1 > axesLength2) condition is enforced for RFellipse creation
	minimumEllipseAxisLength2 = ATORtf_RFproperties.minimumEllipseAxisLength
else:
	receptiveFieldOpponencyAreaFactorTri = 1.0
	maximumAxisLengthMultiplierTri = 2	#tri RF support 2x larger coordinates than ellipse RF
	maximumAxisLengthMultiplier = ATORtf_RFproperties.maximumAxisLengthMultiplierDefault*maximumAxisLengthMultiplierTri
	minimumEllipseAxisLength1 = ATORtf_RFproperties.minimumEllipseAxisLength*2*maximumAxisLengthMultiplierTri	#4	#minimum elongation is required	#can be set to 1 as (axesLength1 > axesLength2) condition is enforced for RFellipse creation
	minimumEllipseAxisLength2 = ATORtf_RFproperties.minimumEllipseAxisLength*maximumAxisLengthMultiplierTri	#2
	
if(ATORtf_RFproperties.lowResultFilterPosition):
	ellipseAxesLengthResolution = 1*maximumAxisLengthMultiplierTri	#2*maximumAxisLengthMultiplierTri	#2*	#pixels (at resolution r)	#use course grain resolution to decrease number of filters #OLD: 1
	if(debugSmallIterations):
		ellipseAxesLengthResolution = ellipseAxesLengthResolution*2	#increase resolution higher than is allowed to be drawn
else:
	ellipseAxesLengthResolution = 1*maximumAxisLengthMultiplierTri	#pixels (at resolution r)
ellipseAngleResolution = 30	#degrees
ellipseColourResolution = 64	#bits
	
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
		numberOfFilterPixels = internalFilterSize[0]*internalFilterSize[1] * 3	#3 features
	elif(numberOfDimensions == 3):
		numberOfFilterPixels = internalFilterSize[0]*internalFilterSize[1] * 3	#3 features	#CHECKTHIS
	#print("numberOfFilterPixels = ", numberOfFilterPixels)
	return numberOfFilterPixels

#match ATORtf_RFellipse algorithm;	
def getInternalFilterSize(filterSize, numberOfDimensions):
	if(numberOfDimensions == 2):
		internalFilterSize = (int(pointFeatureRFinsideRadius*pointFeatureRFopponencyAreaFactor), int(pointFeatureRFinsideRadius*pointFeatureRFopponencyAreaFactor))
	elif(numberOfDimensions == 3):
		internalFilterSize = (int(pointFeatureRFinsideRadius*pointFeatureRFopponencyAreaFactor), int(pointFeatureRFinsideRadius*pointFeatureRFopponencyAreaFactor))	#CHECKTHIS
	return internalFilterSize
	
		
def generateRFfiltersTri(resolutionProperties, RFfiltersList, RFfiltersPropertiesList):
	#generate filter types
	
	#2D code;
	
	#filters are generated based on human magnocellular/parvocellular/koniocellular wavelength discrimination in LGN and VX (double/opponent receptive fields)
	filterTypeIndex = 0
	
	#magnocellular filters (monochromatic);
	colourH = (255, 255, 255)	#high
	colourL = (-255, -255, -255)	#low
	RFfiltersHL, RFpropertiesHL = generateRotationalInvariantRFfilters(resolutionProperties, False, colourH, colourL, filterTypeIndex)
	filterTypeIndex+=1
	
	RFfiltersList.append(RFfiltersHL)

	RFfiltersPropertiesList.append(RFpropertiesHL)

def generateRotationalInvariantRFfilters(resolutionProperties, isColourFilter, filterInsideColour, filterOutsideColour, filterTypeIndex):
	
	RFfiltersList2 = []
	RFfiltersPropertiesList2 = []
	
	#FUTURE: consider storing filters in n dimensional array and finding local minima of filter matches across all dimensions

	#reduce max size of ellipse at each res
	axesLengthMax, filterRadius, filterSize = ATORtf_RFproperties.getFilterDimensions(resolutionProperties, maximumAxisLengthMultiplier, receptiveFieldOpponencyAreaFactorTri)
	
	#print("axesLengthMax = ", axesLengthMax)
	
	for axesLength1 in range(minimumEllipseAxisLength1, axesLengthMax[0]+1, ellipseAxesLengthResolution):
		for axesLength2 in range(minimumEllipseAxisLength2, axesLengthMax[1]+1, ellipseAxesLengthResolution):
			if(axesLength1 > axesLength2):	#ensure that ellipse is always alongated towards axis1 (required for consistent normalisation)
				if((axesLength1 < filterRadius) and (axesLength2 < filterRadius)):
					for angle in range(0, 360, ellipseAngleResolution):	#degrees
						for corner1OpponencyPosition1 in range(minimumCornerOpponencyPosition, maximumCornerOpponencyPosition+1, cornerOpponencyPositionResolution):
							for corner1OpponencyPosition2 in range(minimumCornerOpponencyPosition, maximumCornerOpponencyPosition+1, cornerOpponencyPositionResolution):
								for corner2OpponencyPosition1 in range(minimumCornerOpponencyPosition, maximumCornerOpponencyPosition+1, cornerOpponencyPositionResolution):
									for corner2OpponencyPosition2 in range(minimumCornerOpponencyPosition, maximumCornerOpponencyPosition+1, cornerOpponencyPositionResolution):
										for corner3OpponencyPosition1 in range(minimumCornerOpponencyPosition, maximumCornerOpponencyPosition+1, cornerOpponencyPositionResolution):
											for corner3OpponencyPosition2 in range(minimumCornerOpponencyPosition, maximumCornerOpponencyPosition+1, cornerOpponencyPositionResolution):

												axesLength = (axesLength1, axesLength2)

												axesLengthInside = (pointFeatureRFinsideRadius, pointFeatureRFinsideRadius)	#estimation
												axesLengthOutside = (int(pointFeatureRFinsideRadius*pointFeatureRFopponencyAreaFactor), int(pointFeatureRFinsideRadius*pointFeatureRFopponencyAreaFactor))	#estimation
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
												#print("vertexCoordinatesRelative = ", vertexCoordinatesRelative)
												vertexCoordinatesRelativeInside = copy.deepcopy(vertexCoordinatesRelative)
												vertexCoordinatesRelativeOutside = copy.deepcopy(vertexCoordinatesRelative)
												vertexCoordinatesRelativeOutside[0][0] = vertexCoordinatesRelativeOutside[0][0] + corner1OpponencyPosition1
												vertexCoordinatesRelativeOutside[0][1] = vertexCoordinatesRelativeOutside[0][1] + corner1OpponencyPosition2
												vertexCoordinatesRelativeOutside[1][0] = vertexCoordinatesRelativeOutside[1][0] + corner2OpponencyPosition1
												vertexCoordinatesRelativeOutside[1][1] = vertexCoordinatesRelativeOutside[1][1] + corner2OpponencyPosition2
												vertexCoordinatesRelativeOutside[2][0] = vertexCoordinatesRelativeOutside[2][0] + corner3OpponencyPosition1
												vertexCoordinatesRelativeOutside[2][1] = vertexCoordinatesRelativeOutside[2][1] + corner3OpponencyPosition2
												
												#vertexCoordinatesRelativeInside[0][0] = vertexCoordinatesRelativeInside[0][0] + corner1OpponencyPosition1
												#vertexCoordinatesRelativeInside[0][1] = vertexCoordinatesRelativeInside[0][1] + corner1OpponencyPosition2
												#vertexCoordinatesRelativeInside[1][0] = vertexCoordinatesRelativeInside[1][0] + corner2OpponencyPosition1
												#vertexCoordinatesRelativeInside[1][1] = vertexCoordinatesRelativeInside[1][1] + corner2OpponencyPosition2
												#vertexCoordinatesRelativeInside[2][0] = vertexCoordinatesRelativeInside[2][0] + corner3OpponencyPosition1
												#vertexCoordinatesRelativeInside[2][1] = vertexCoordinatesRelativeInside[2][1] + corner3OpponencyPosition2

												RFfilter = generateRFfilter(resolutionProperties.resolutionIndex, isColourFilter, RFpropertiesInside, RFpropertiesOutside, vertexCoordinatesRelativeInside, vertexCoordinatesRelativeOutside)
												RFfiltersList2.append(RFfilter)

												RFproperties = ATORtf_RFproperties.RFpropertiesClass(resolutionProperties.resolutionIndex, resolutionProperties.resolutionFactor, filterSize, ATORtf_RFproperties.RFtypeTri, filterCenterCoordinates, axesLength, angle, filterInsideColour)
												#RFproperties.centerCoordinates = centerCoordinates 	#centerCoordinates are set after filter is applied to imageSegment
												RFproperties.isColourFilter = isColourFilter
												RFfiltersPropertiesList2.append(RFproperties)	#CHECKTHIS: use RFpropertiesInside not RFpropertiesOutside

												#debug:
												#print(RFfilter.shape)
												if(resolutionProperties.debugVerbose):
													ATORtf_RFproperties.printRFproperties(RFproperties)
												#print("RFfilter = ", RFfilter)

												RFfilterImageFilename = "RFfilterResolutionIndex" + str(resolutionProperties.resolutionIndex) + "filterTypeIndex" + str(filterTypeIndex) + "axesLength1" + str(axesLength1) + "axesLength2" + str(axesLength2) + "angle" + str(angle) + "corner1OpponencyPosition1" + str(corner1OpponencyPosition1) + "corner1OpponencyPosition2" + str(corner1OpponencyPosition2) + "corner2OpponencyPosition1" + str(corner2OpponencyPosition1) + "corner2OpponencyPosition2" + str(corner2OpponencyPosition2) + "corner3OpponencyPosition1" + str(corner3OpponencyPosition1) + "corner3OpponencyPosition2" + str(corner3OpponencyPosition2) + ".png"
												ATORtf_RFproperties.saveRFFilterImage(RFfilter, RFfilterImageFilename)

	#create 3D tensor (for hardware accelerated test/application of filters)
	RFfiltersTensor = tf.stack(RFfiltersList2, axis=0)

	return RFfiltersTensor, RFfiltersPropertiesList2
	

def generateRFfilter(resolutionProperties, isColourFilter, RFpropertiesInside, RFpropertiesOutside, vertexCoordinatesRelativeInside, vertexCoordinatesRelativeOutside):

	# RF filter example (RFfilterTF) - generatePointFeatureCorners=False:
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
	
	if(generatePointFeatureCorners):
		drawFeatureType = ATORtf_RFproperties.RFfeatureTypeCorner
	else:
		drawFeatureType = ATORtf_RFproperties.RFfeatureTypePoint		
		
	RFfilterTF = ATORtf_RFproperties.drawRF(RFfilterTF, RFpropertiesInside1, RFpropertiesOutside1, drawFeatureType, True)
	RFfilterTF = ATORtf_RFproperties.drawRF(RFfilterTF, RFpropertiesInside2, RFpropertiesOutside2, drawFeatureType, True)
	RFfilterTF = ATORtf_RFproperties.drawRF(RFfilterTF, RFpropertiesInside3, RFpropertiesOutside3, drawFeatureType, True)
	
	#print("RFfilterTF = ", RFfilterTF)

	if(ATORtf_operations.storeRFfiltersValuesAsFractions):
		RFfilterTF = tf.divide(RFfilterTF, ATORtf_operations.rgbMaxValue)
			
	if(not isColourFilter):
		RFfilterTF = tf.image.rgb_to_grayscale(RFfilterTF)
	
	#print("RFfilterTF.shape = ", RFfilterTF.shape)	
	#print("RFfilterTF = ", RFfilterTF)

	return RFfilterTF
	
