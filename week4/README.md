# Team5

Instructions to run our code.
It is assumed that folders of the queries and DDBB are at the same path level than the files.

## Q1_W4.py

It performs the retrieval of the corresponding image. To do so, we perform a background subtraction, since we can have up to 3 paintings per image. 
Then it can compute the text, ORB and SIFT descriptor. Also, it removes the text in order to match the paintings.

Creates a folder where the .txt with the names of the authors are stored. 

Change path to query: (default 'qsd1_W4') - Line 10

Change path to BBDD - Line 11

Number of best mappings (k): 10 (for MAP@5, k = 5) - Line 12

Choose descriptor (ORB or SIFT) - Line 13



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





