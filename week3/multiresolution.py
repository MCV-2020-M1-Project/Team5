from histograms import *
from utils import *
from calculateDistances import *
import pickle
import ml_metrics as metrics


#Guiding dictionaries
histTypes = {'gray','rgb','hsv','lab','labrgb','ycrcb','lbp','lbp_sci','hog'}
metricTypes = {'euclid','chi','corr','histInt','hell'}

#Params to change
qPath = 'qsd1_w3/non_augmented'
mPath = 'BBDD'
typeHist = 'hog'
typeMetric = 'hell'
k = 4

listQImgs , idsQImgs = load_images(qPath)
listMImgs , idsMImgs = load_images(mPath)

nBins = 64
listQHists = []
listMHists = []
finalIds = []
#For each one of the queries, compute the distance(s) between the query and each of the museum images
for i in range(len(listQImgs)):
    print(i)
    sumKdists = []
    # Divide image in k parts
    dividedImage = divideImg(listQImgs[i],k)
    #Calculate histograme for each of the subimages
    if typeHist == 'gray':
        listsubQHist = grayHist(dividedImage,nBins)
    elif typeHist == 'rgb':
        listsubQHist = rgbHist(dividedImage,nBins)
    elif typeHist == 'hsv':
        listsubQHist = hsvHist(dividedImage,nBins)
    elif typeHist == 'lab':
        listsubQHist = labHist(dividedImage,nBins)
    elif typeHist == 'labrgb':
        listsubQHist = labrgbHist(dividedImage,nBins)
    elif typeHist == 'ycrcb':
        listsubQHist = ycrcbHist(dividedImage, nBins)
    elif typeHist == 'hog':
        listsubQHist = hogHist(dividedImage, nBins)
    elif typeHist == 'lbp':
        listsubQHist = lbpHist(dividedImage, nBins)
    elif typeHist == 'lbp_sci':
        listsubQHist = lbpHistSci(dividedImage, nBins)
    else:
        print('YOU MUST INTRODUCE A SUITABLE HISTOGRAM NAME!!')
        break
    listQHists.append(listsubQHist)

    #For each of the museum images, compute the k distances between the query and the museum images
    for j in range(len(listMImgs)):
        Kdists = []
        # Divide image in k parts
        dividedImage = divideImg(listMImgs[j], k)
        # Calculate histograme for each of the subimages
        if typeHist == 'gray':
            listsubMHist = grayHist(dividedImage, nBins)
        elif typeHist == 'rgb':
            listsubMHist = rgbHist(dividedImage, nBins)
        elif typeHist == 'hsv':
            listsubMHist = hsvHist(dividedImage, nBins)
        elif typeHist == 'lab':
            listsubMHist = labHist(dividedImage, nBins)
        elif typeHist == 'labrgb':
            listsubMHist = labrgbHist(dividedImage, nBins)
        elif typeHist == 'ycrcb':
            listsubMHist = ycrcbHist(dividedImage, nBins)
        elif typeHist == 'hog':
            listsubMHist = hogHist(dividedImage, nBins)
        elif typeHist == 'lbp':
            listsubMHist = lbpHist(dividedImage, nBins)
        elif typeHist == 'lbp_sci':
            listsubMHist = lbpHistSci(dividedImage, nBins)
        else:
            print('YOU MUST INTRODUCE A SUITABLE HISTOGRAM NAME!!')
            break
        listMHists.append(listsubMHist)
        for m in range(k):
            if typeMetric == 'euclid':
                dist = euclid_dist(listQHists[i][m][:], listMHists[j][m][:])
            elif typeMetric == 'chi':
                dist = chiSquare_dist(listQHists[i][m][:], listMHists[j][m][:])
            elif typeMetric == 'corr':
                dist = corrDist(listQHists[i][m][:], listMHists[j][m][:])
            elif typeMetric == 'histInt':
                dist = Hist_intersection(listQHists[i][m][:], listMHists[j][m][:])
            elif typeMetric == 'hell':
                dist = Hellinger_kernel(listQHists[i][m][:], listMHists[j][m][:])
            Kdists.append(dist)
        sumKdists.append(sum(Kdists))
    allDist, idsSorted = zip(*sorted(zip(sumKdists, idsMImgs)))
    finalIds.append(list(idsSorted))


filename = 'result.pkl'
outfile = open(filename, 'wb')
pickle.dump(finalIds, outfile)
outfile.close()





