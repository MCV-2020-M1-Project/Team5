# Team5

Instructions to run our code.
It is assumed that folders of the queries and DDBB are at the same path level than the files.

## Q1_W2.py

It performs the retrieval of the corresponding image. To do so, it computes the multiresolution and the text removal. 

Change path to query - Line 10 (default 'qsd1_W2')
Change path to BBDD - Line 11
Number of best mappings (k): 10 (for MAP@5, k = 5) - Line 12

## Q2_W2.py

It performs the retrieval of the corresponding image, but this time, there can be 2 paintings per
image, so a background subtration must be performed, along with the differentiation of the 2 paintings.
It computes the multiresolution and the text removal. 

Change path to query - Line 10 (default 'qsd1_W2')
Change path to BBDD - Line 11
Number of best mappings (k): 10 (for MAP@5, k = 5) - Line 12


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



