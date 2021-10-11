# -*- coding: utf-8 -*-
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

#class TriProperties():
#	def __init__(self, vertexCoordinatesRelative):
#		self.vertexCoordinatesRelative = vertexCoordinatesRelative
#		#alternatively, could be defined by an isosceles triangle where each vertix corresponds to the vertices of an ellipse;
#		#self.axesLength = axesLength
#		#self.angle = angle
		

def printTriProperties(triProperties):
	print("vertexCoordinatesRelative = ", triProperties.vertexCoordinatesRelative)
		

def deriveArtificialEllipsePropertiesFromTriVertexCoordinates(centerCoordinates, vertexCoordinatesRelative):
	#TODO:
	#centerCoordinates
	#axesLength
	#angle
	#colour
	return centerCoordinates, axesLength, angle, colour
	
