# -*- coding: utf-8 -*-
"""ATORtf_ellipseProperties.py

# Author:
Richard Bruce Baxter - Copyright (c) 2021 Baxter AI (baxterai.com)

# License:
MIT License

# Requirements:
See ATORtf.tf

# Usage:
See ATORtf.tf

# Description:
ATORtf Ellipse (or Ellipsoid) Properties

"""

import cv2
import copy
import numpy as np

ellipseAngleResolution = 10	#degrees
minimumEllipseFitErrorRequirement = 1500.0	#calibrate

class EllipseProperties():	#or EllipsoidProperties
	def __init__(self, resolutionIndex,  resolutionFactor, imageSize, centerCoordinates, axesLength, angle, colour):
		self.resolutionIndex = resolutionIndex
		self.resolutionFactor = resolutionFactor
		self.imageSize = imageSize
		self.centerCoordinates = centerCoordinates
		self.axesLength = axesLength
		self.angle = angle
		self.colour = colour
		self.isColourFilter = True
		self.numberOfDimensions = 2	#currently only support 2D data (not ellipses in 3D space or ellipsoids in 3D space)
		self.filterIndex = None
		#self.centerCoordinates = (-1, -1)
		#self.axesLength = (-1, -1)
		#self.angle = -1
		#self.colour = -1

def drawEllipse(outputImage, ellipseProperties):
	#https://docs.opencv.org/4.5.3/d6/d6e/group__imgproc__draw.html#ga28b2267d35786f5f890ca167236cbc69
	outputImageMod = cv2.ellipse(outputImage, ellipseProperties.centerCoordinates, ellipseProperties.axesLength, ellipseProperties.angle, 0, 360, ellipseProperties.colour, -1)
	return outputImageMod

def normaliseEllipseProperties(ellipsePropertiesOptimum):
	resolutionFactor = ellipsePropertiesOptimum.resolutionFactor
	ellipsePropertiesOptimumNormalised = copy.deepcopy(ellipsePropertiesOptimum) 
	ellipsePropertiesOptimumNormalised.centerCoordinates = (ellipsePropertiesOptimumNormalised.centerCoordinates[0]*resolutionFactor, ellipsePropertiesOptimumNormalised.centerCoordinates[1]*resolutionFactor)
	ellipsePropertiesOptimumNormalised.axesLength = (ellipsePropertiesOptimumNormalised.axesLength[0]*resolutionFactor, ellipsePropertiesOptimumNormalised.axesLength[1]*resolutionFactor)
	return ellipsePropertiesOptimumNormalised
	 
def calculateEllipseFitError(inputImage, inputImageMod):
	meanSquaredError = (np.sqrt((pow(np.subtract(inputImage, inputImageMod, dtype=np.int32), 2))).sum())	#currently use mean squared error
	ellipseFitError = meanSquaredError
	return ellipseFitError

def testEllipseApproximation(inputImageR, ellipseProperties):
	inputImageRmod = copy.deepcopy(inputImageR)
	cv2.ellipse(inputImageRmod, ellipseProperties.centerCoordinates, ellipseProperties.axesLength, ellipseProperties.angle, 0, 360, ellipseProperties.colour, -1)
	ellipseFitError = calculateEllipseFitError(inputImageR, inputImageRmod)
	return inputImageRmod, ellipseFitError

def centroidOverlapsEllipseWrapper(ellipseFitError, ellipseProperties, ellipsePropertiesOptimumLast):
	result = True
	if(ellipseFitError < minimumEllipseFitErrorRequirement):
		if ellipsePropertiesOptimumLast is None:
			result = False
		else:
			result = centroidOverlapsEllipse(ellipseProperties, ellipsePropertiesOptimumLast)
	return result					

def centroidOverlapsEllipse(ellipseProperties, ellipsePropertiesOptimumLast):
	result = True
	#minimumDistance = max(ellipseProperties.axesLength[0], ellipseProperties.axesLength[1], ellipsePropertiesOptimumLast.axesLength[0], ellipsePropertiesOptimumLast.axesLength[1])	#CHECKTHIS
	minimumDistance = max((ellipseProperties.axesLength[0] + ellipseProperties.axesLength[1])/2, (ellipsePropertiesOptimumLast.axesLength[0] + ellipsePropertiesOptimumLast.axesLength[1])/2)	#CHECKTHIS
	if((ellipsePropertiesOptimumLast.centerCoordinates[0] - ellipseProperties.centerCoordinates[0])**2 + (ellipsePropertiesOptimumLast.centerCoordinates[1] - ellipseProperties.centerCoordinates[1])**2) > minimumDistance**2:
		result = False
	return result		
	
def printEllipseProperties(ellipseProperties):
	if(ellipseProperties.numberOfDimensions == 2):
		print("printEllipseProperties: numberOfDimensions = ", ellipseProperties.numberOfDimensions, ", resolutionIndex = ", ellipseProperties.resolutionIndex, ", isColourFilter = ", ellipseProperties.isColourFilter, ", imageSize = ", ellipseProperties.imageSize[0], ",", ellipseProperties.imageSize[1], ", centerCoordinates = ", ellipseProperties.centerCoordinates[0], ",", ellipseProperties.centerCoordinates[1], ", axesLength = ", ellipseProperties.axesLength[0], ",", ellipseProperties.axesLength[1], " angle = ", ellipseProperties.angle, ", colour = ", ellipseProperties.colour[0], ",", ellipseProperties.colour[1], ",", ellipseProperties.colour[2])
	elif(ellipseProperties.numberOfDimensions == 3):
		print("printEllipseProperties: numberOfDimensions = ", ellipseProperties.numberOfDimensions, ", resolutionIndex = ", ellipseProperties.resolutionIndex, ", isColourFilter = ", ellipseProperties.isColourFilter, ", imageSize = ", ellipseProperties.imageSize[0], ",", ellipseProperties.imageSize[1], ",", ellipseProperties.imageSize[2], ", centerCoordinates = ", ellipseProperties.centerCoordinates[0], ",", ellipseProperties.centerCoordinates[1], ",", ellipseProperties.centerCoordinates[2], ", axesLength = ", ellipseProperties.axesLength[0], ",", ellipseProperties.axesLength[1], ",", ellipseProperties.axesLength[2], " angle = ", ellipseProperties.angle, ", colour = ", ellipseProperties.colour[0], ",", ellipseProperties.colour[1], ",", ellipseProperties.colour[2])	

