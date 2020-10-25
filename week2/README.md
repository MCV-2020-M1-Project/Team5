# Team5

Instructions to run our code.

## Q1_W2.py

It performs a simple mapping between query images and museum images (no bg subtraction). 

Change path to query - Line 12
Change path to BBDD - Line 13
Change the metric - Line 11
- Options: "hell", "chi", "euclid", "corr", "histInt"
Change function to call histogram - Line 15-16
- For example: labHist()

## Q2_W2.py

It performs a simple mapping between query images and museum images (with bg subtraction). 

Change path to query - Line 17
Change path to BBDD - Line 49
Change the metric - Line 47
- Options: "hell", "chi", "euclid", "corr", "histInt"
Change function to call histogram - Line 52-53
- Case line 52: labHist()
- Case line 53: labHist2() --> (the 2 is compulsory)


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

### utils

Useful generic functions. (For instance: load images, load ground truth masks, etc.)