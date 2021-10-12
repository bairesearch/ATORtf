# ATORtf

### Author

Richard Bruce Baxter - Copyright (c) 2021 Baxter AI (baxterai.com)

### Description

ATORtf is a hardware accelerated version of BAI ATOR (Axis Transformation Object Recognition) for TensorFlow.

ATORtf supports ellipsoid features, and normalises them with respect to their major/minor ellipticity axis orientation. 

There are a number of advantages of using ellipsoid features over point features;
* the number of feature sets/normalised snapshots required is significantly reduced
* scene component structure can be maintained (as detected component ellipses can be represented in a hierarchical graph structure)

Ellipse features/components are detected based on simulated artificial receptive fields; RF (on/off, off/on).

ATORtf also supports point (corner/centroid) features of the ATOR specification; 
https://www.wipo.int/patentscope/search/en/WO2011088497

### License

MIT License

### Installation
```
conda create -n ATORtf python=3.9
source activate ATORtf
pip install tensorflow-gpu==2.6
conda install --file condaRequirements.txt
	where condaRequirements.txt contains;
		numpy
		click
		pillow
pip install tensorflow-addons
pip install opencv-python opencv-contrib-python

```

### Execution
```
source activate ATORtf
python ATORtf.py images/leaf1.png
```
