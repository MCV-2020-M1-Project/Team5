# Team5

Instructions to run our code.
It is assumed that folders of the queries and DDBB are at the same path level than the files.

## Q1_W3.py

It performs the retrieval of the corresponding image. To do so, it computes the text and color descriptor ( and also texture , but it is not default). Also, it removes the text in order to match the paintings.

Creates a folder where the .txt with the names of the authors are stored. 

Change path to query: (default 'qsd1_W3') - Line 9

Change path to BBDD - Line 10

Number of best mappings (k): 10 (for MAP@5, k = 5) - Line 11


## Q2_W3.py

It performs the retrieval of the corresponding image, but this time, there can be 2 paintings per
image, so a background subtration must be performed, along with the differentiation of the 2 paintings.
It computes the multiresolution, the text removal, and all the descriptors (text, color, texture). 

Change path to query: (default 'qsd2_W3') - Line 9

Change path to BBDD - Line 10

Number of best mappings (k): 10 (for MAP@5, k = 5) - Line 11


## Files description

### histograms.py

Functions to extract different type of histograms from the images.

### masks.py

Functions that compute the background subtraction, store the masks, and cropps the images.

### evaluation.py

It computes the Precision, Recall, and F1-measure.

### calculateDistances.py

Functions to compute the different type of metrics:
- Hellinger
- Chi-Square
- Euclidean
- Correlation
- Histogram intersection

### utils.py

Useful generic functions. (For instance: load images, load ground truth masks, etc.)

### multiresolution.py

Divide input image into k subimages and compute the k histograms. Also computes the distance between the subimages, and subsequently, between the original images. 

### textDetection.py

It finds the Bbox of the text using morphological filters. 

### textDetection.py

Finds the coordinates of the text bbox for each painting. 

### textReader.py

With the bbox coordinates, it crops the text region and performs OCR in order to read the text in the image.

### descriptors.py

This file contains the functions to compute the 3 different descriptors: text, texture and color

### feature_extraction.py

Implements the texture detection algorithms, LBP in our case. 



