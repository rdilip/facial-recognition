# facial-recognition

A Python facial recognition program based on Principal Component Analysis. Still very much a work in progress. This program uses Principal Component Analysis to essentially create a basis for a "face subspace" in 150 by 150 dimensional space. See my blog for further details on how this works mathematically. 

Images should be stored in an img folder, which I have not included to a) preserve privacy, and b) because it's a very big file size. PCA is based on linear transforms, but pure translations are not linear, so PCA is very sensitive to factors like misalignment and background noise. There are ways to address this, some of which I will be inserting into the code. 
