# Team5

Instructions to run our code.
It is assumed that folders of the queries and DDBB are at the same path level than the files.

## Q1_W5.py

It performs the retrieval of the corresponding image. To do so, we perform a background subtraction, since we can have up to 3 paintings per image. Also it rotates the paintings so they are parallel to the screen. 
Then it can compute the text and ORB. Also, it removes the text in order to match the paintings.

Creates a folder where the .txt with the names of the authors are stored. 

Change path to query: (default 'qsd1_5') - Line 12

Change path to BBDD - Line 13

Number of best mappings (k): 10 (for MAP@5, k = 5) - Line 14

Choose descriptor (ORB) - Line 15


## clustering.py

It performs the clustering of the paintings using k-means. K-means is implemented to cluster paintings given a certain set of feature vectors. 

Change path to BBDD - Line 205

K - Line 252

## Files description


### masks.py

Functions that compute the background subtraction, store the masks, and cropp the images.


### utils.py

Useful generic functions. (For instance: load images, load ground truth masks, etc.)


### textDetection.py

It finds the Bbox of the text using morphological filters. 


### textReader.py

With the bbox coordinates, it crops the text region and performs OCR in order to read the text in the image.


### descriptors.py

This file contains the functions to compute the 3 different descriptors: text, ORB and SIFT


### rotation.py

Contains necessary functions to perform the rotation of the image. 


### keypoints.py 

Calculate keypoints (and descriptors) with SIFT and ORB.







