# -*- coding: utf-8 -*-
"""ATORtf_RFProperties.py

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

import numpy as np
import copy

import ATORtf_ellipseProperties
import ATORtf_RFellipse
import ATORtf_RFtri
import ATORtf_operations


RFtypeEllipse = 1
RFtypeTri = 2


class RFPropertiesClass(ATORtf_ellipseProperties.EllipsePropertiesClass):
	def __init__(self, resolutionIndex, resolutionFactor, imageSize, RFtype, centerCoordinates, axesLength, angle, colour):
		
		self.resolutionIndex = resolutionIndex
		self.resolutionFactor = resolutionFactor
		self.imageSize = imageSize
		
		self.RFtype = RFtype
		super().__init__(centerCoordinates, axesLength, angle, colour)
		
		self.isColourFilter = True
		self.numberOfDimensions = 2	#currently only support 2D data (not ellipses in 3D space or ellipsoids in 3D space)
		self.filterIndex = None
		self.imageSegmentIndex = None

def printRFProperties(RFProperties):
	print("printRFProperties: numberOfDimensions = ", RFProperties.numberOfDimensions, ", resolutionIndex = ", RFProperties.resolutionIndex, ", isColourFilter = ", RFProperties.isColourFilter, ", imageSize = ", RFProperties.imageSize, "centerCoordinates = ", RFProperties.centerCoordinates)
	if(RFProperties.RFtype == RFtypeEllipse):
		ATORtf_ellipseProperties.printEllipseProperties(RFProperties)
	elif(RFProperties.RFtype == RFtypeTri):
		#ATORtf_ellipseProperties.printEllipseProperties(RFProperties)
		#ATORtf_RFtri.printTriProperties(RFProperties.triProperties)
		print("vertexCoordinatesRelative = ", RFProperties.vertexCoordinatesRelative)
	

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


