# -*- coding: utf-8 -*-
"""ATORtf_operations.py

# Author:
Richard Bruce Baxter - Copyright (c) 2021 Baxter AI (baxterai.com)

# License:
MIT License

# Requirements:
See ATORtf.tf

# Usage:
See ATORtf.tf

# Description:
ATORtf Operations

"""

import tensorflow as tf
import os
import cv2
import math
from PIL import Image

opencvVersion = 3	#or 4

def modifyTuple(t, index, value):
	lst = list(t)
	lst[index] = value
	tNew = tuple(lst)
	return tNew

def displayImage(outputImage):
	#for debugging purposes only - no way of closing window
	image = cv2.cvtColor(outputImage, cv2.COLOR_BGR2RGB)
	Image.fromarray(image).show()  

def saveImage(inputimagefilename, imageObject):
	#outputImageFolder = os.getcwd()	#current folder
	inputImageFolder, inputImageFileName = os.path.split(inputimagefilename)
	outputImageFileName = inputImageFileName
	cv2.imwrite(outputImageFileName, imageObject)
			
def convertDegreesToRadians(degrees):
	radians = degrees * math.pi / 180
	return radians

def expandDimsN(tensor, numberOfDimensions, axis):
	tensorExpanded = tensor
	for index in range(numberOfDimensions):
		tensorExpanded = tf.expand_dims(tensorExpanded, axis=axis)
	return tensorExpanded

def rotatePoint2D(origin, point, angle):

	origin1 = (origin[0], origin[1]) 
	point1 = (point[0], point[1])
	angle1 = angle[0]
	qx1, qy1 = rotatePointNP2D(origin1, point1, angle1)
	
	pointRotated = (qx1, qy1)	
	return pointRotated

def rotatePoint3D(origin, point, angle):

	origin1 = (origin[0], origin[1]) 
	point1 = (point[0], point[1])
	angle1 = angle[0]
	qx1, qy1 = rotatePointNP2D(origin1, point1, angle1)
	
	origin2 = (origin1[1], origin[2])
	point2 = (point1[1], point[2])
	angle2 = angle[1]
	qx2, qy2 = rotatePointNP2D(origin2, point2, angle2)
	
	pointRotated = (qx1, qx2, qy2)
	return pointRotated

def rotatePointNP2D(origin, point, angle):

	theta = convertDegreesToRadians(angle)

	ox, oy = origin
	px, py = point

	qx = ox + math.cos(theta) * (px - ox) - math.sin(theta) * (py - oy)
	qy = oy + math.sin(theta) * (px - ox) + math.cos(theta) * (py - oy)

	return qx, qy

def calculateDistance2D(point1, point2):
	point_a = np.array((point1[0],point1[1]))
	point_b = np.array((point2[0],point2[1]))
	return calculateDistanceNP(point1, point2)
	
def calculateDistance3D(point1, point2):
	point_a = np.array((point1[0],point1[1],point1[2]))
	point_b = np.array((point2[0],point2[1],point2[2]))
	return calculateDistanceNP(point1, point2)
	
def calculateDistanceNP(point1, point2):
	distance = np.linalg.norm(point1 - point2)
	return distance

def calculateRelativePosition3D(angle, axisLength):
	print("error calculateRelativePosition3D: RFPropertiesParent.numberOfDimensions == 3 not yet coded")
	quit()
	
def calculateRelativePosition2D(angle, hyp):
	#cos(theta) = adj/hyp
	#sin(theta) = opp/hyp
	#opp (x) = sin(theta)*hyp
	#adj (y) = cos(theta)*hyp 
	
	hyp = axisLength[0]
	theta = convertDegreesToRadians(angle)
	relativePosition2D = (math.sin(theta)*hyp, math.cos(theta)*hyp)
	return relativePosition2D
		
