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
	
