# ATORtf

### Author

Richard Bruce Baxter - Copyright (c) 2021 Baxter AI (baxterai.com)

### Description

ATORtf is a hardware accelerated version of BAI ATOR (Axis Transformation Object Recognition).

ATORtf uses ellipsoid features (rather than point/centroid features of the ATOR specification*), and normalises them with respect to their major/minor ellipticity axis orientation. 

There are a number of core advantages of using ellipsoid features over point features;
a) the number of feature sets/normalised snapshots requires significantly reduced
b) scene component structure can be maintained (as detected component ellipses can be represented in a hierarchical graph structure)

Ellipse features/components are detected based on simulated artificial receptive fields (on/off, off/on).

* https://www.wipo.int/patentscope/search/en/WO2011088497

### License

MIT License

### Installation
```
conda create -n ATORtf python=3
source activate ATORtf
conda install --file condaRequirements.txt
	where requirements.txt contains;
		numpy
		tensorflow
		click
		opencv
		pillow
conda install -c esri tensorflow-addons

```

### Execution
```
source activate ATORtf
python ATORtf.py images/leaf1.png
```
