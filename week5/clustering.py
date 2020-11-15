

import cv2 as cv
from utils import *
import os
import matplotlib.pyplot as plt


# Euclidean distance
def euclid_dist(vec1, vec2):
    d = np.linalg.norm(vec1-vec2)
    return d

def descriptor_mean_HSV(painting):

    hsv = cv2.cvtColor(painting, cv2.COLOR_BGR2HSV)
    x,y,_ = painting.shape

    mh = np.mean(hsv[:,:,0])
    ms = np.mean(hsv[:,:,1])
    mv = np.mean(hsv[:,:,2])

    descriptor = [mh,ms,mv]

    return descriptor

def descriptor_mean_BGR(painting):

    bgr = painting
    x,y,_ = painting.shape

    mb = np.mean(bgr[:,:,0])
    mg = np.mean(bgr[:,:,1])
    mr = np.mean(bgr[:,:,2])

    descriptor = [mb,mg,mr]

    return descriptor

def descriptor_HSV_H6(painting):

    hsv = cv2.cvtColor(painting, cv2.COLOR_BGR2HSV)
    x, y, _ = painting.shape

    mh = np.mean(hsv[:, :, 0])
    ms = np.mean(hsv[:, :, 1])
    mv = np.mean(hsv[:, :, 2])

    s = np.array(hsv[:,:,1]/43,np.uint8)
    ms = np.median(s).astype(np.uint8) * 43
    ms = ms.astype(np.float32)

    v = np.array(hsv[:,:,2]/43,np.uint8)
    mv = np.median(v).astype(np.uint8) * 43
    mv = mv.astype(np.float32)

    descriptor = [mh, ms, mv]

    return descriptor


def descriptor_HSV_LapGrad(painting):

    hsv = cv2.cvtColor(painting, cv2.COLOR_BGR2HSV)
    x, y, _ = painting.shape

    mh = np.mean(hsv[:, :, 0])
    ms = np.mean(hsv[:, :, 1])

    l = cv2.Laplacian(painting, cv.CV_64F)
    lap = (np.mean(l) - (np.min(l)) / (1e-7 + (np.max(l) - np.min(l)))) * 255
    lap = lap.astype(np.float32)

    descriptor = [mh, ms, lap]

    return descriptor

def descriptor_Gradients(painting):

    gray = cv2.cvtColor(painting, cv2.COLOR_BGR2GRAY)
    x, y, _ = painting.shape

    l = cv2.Laplacian(gray, cv.CV_64F)
    lap = (np.mean(l) - (np.min(l)) / (1e-7 + (np.max(l) - np.min(l)))) * 255
    lap = lap.astype(np.float32)

    sobelx = cv.Sobel(gray, cv.CV_64F, 1, 0, ksize=5)
    sx = (np.mean(sobelx) - (np.min(sobelx)) / (1e-7 + (np.max(sobelx) - np.min(sobelx)))) * 255
    sx = sx.astype(np.float32)

    sobely = cv.Sobel(gray, cv.CV_64F, 0, 1, ksize=5)
    sy = (np.mean(sobely) - (np.min(sobely)) / (1e-7 + (np.max(sobely) - np.min(sobely)))) * 255
    sy = sy.astype(np.float32)


    descriptor = [sy, sx, lap]

    return descriptor


def descriptor_labhist(painting):

    nBins = 8

    lab = cv2.cvtColor(painting, cv2.COLOR_BGR2LAB)
    l = np.histogram(lab[:, :, 0], nBins, [0, 255])[0]
    a = np.histogram(lab[:, :, 1], nBins, [0, 255])[0]
    b = np.histogram(lab[:, :, 2], nBins, [0, 255])[0]

    feat = np.concatenate((l, a, b), axis=None)

    return feat

def desc_rgb_lapl(painting):

    bgr = painting
    x,y,_ = painting.shape

    mb = np.mean(bgr[:,:,0])
    mg = np.mean(bgr[:,:,1])
    mr = np.mean(bgr[:,:,2])

    l = cv2.Laplacian(bgr, cv.CV_64F)
    lap = (np.mean(l) - (np.min(l)) / (1e-7 + (np.max(l) - np.min(l)))) * 255
    lap = lap.astype(np.float32)

    descriptor = [mb,mg,mr,lap]

    return descriptor

def desc_rgb_lapl_sat(painting):

    bgr = painting
    hsv = cv2.cvtColor(painting, cv2.COLOR_BGR2HSV)
    x,y,_ = painting.shape

    mb = np.mean(bgr[:,:,0])
    mg = np.mean(bgr[:,:,1])
    mr = np.mean(bgr[:,:,2])
    mh = np.mean(hsv[:, :, 0])
    ms = np.mean(hsv[:,:,1])

    l = cv2.Laplacian(bgr, cv.CV_64F)
    lap = (np.mean(l) - (np.min(l)) / (1e-7 + (np.max(l) - np.min(l)))) * 255
    lap = lap.astype(np.float32)

    descriptor = [mb,mg,mr,mh, ms, lap]

    return descriptor

def desc_rgb_hue_sat(painting):

    bgr = painting
    hsv = cv2.cvtColor(painting, cv2.COLOR_BGR2HSV)
    x,y,_ = painting.shape

    mb = np.mean(bgr[:,:,0])
    mg = np.mean(bgr[:,:,1])
    mr = np.mean(bgr[:,:,2])
    mm = np.mean([mb,mg,mr])
    mh = np.mean(hsv[:, :, 0])
    ms = np.mean(hsv[:,:,1])

    l = cv2.Laplacian(bgr, cv.CV_64F)
    lap = (np.mean(l) - (np.min(l)) / (1e-7 + (np.max(l) - np.min(l)))) * 255
    lap = lap.astype(np.float32)

    descriptor = [mb,mg,mr,lap, ms]

    return descriptor


# Input painting, output descriptor list
def desc_over128(painting):

    descriptor = []

    if len(painting.shape) == 3:
        for i in range(painting.shape[2]):
            # chansum = np.float32(sum(painting[:,:,i].reshape((painting.shape[0]*painting.shape[1]),1)>128))
            bpaint = painting[:, :, i].reshape((painting.shape[0] * painting.shape[1]), 1) > 128
            chansum = np.float32(np.sum(bpaint))
            descriptor.append(chansum)

    else:
        bpaint = painting.reshape((painting.shape[0] * painting.shape[1]), 1) > 128
        descriptor = [np.float32(np.sum(bpaint))]

    return descriptor

def denoise_img(paint):

    denoised_img = np.zeros((np.size(paint, 0), np.size(paint, 1), np.size(paint, 2)))
    denoised_img[:, :, 0] = cv.medianBlur(painting[:, :, 0],3)  # Median is the best effective for salt and pepper noise
    denoised_img[:, :, 1] = cv.medianBlur(painting[:, :, 1],3)  # Median is the best effective for salt and pepper noise
    denoised_img[:, :, 2] = cv.medianBlur(painting[:, :, 2],3)  # Median is the best effective for salt and pepper noise
    denoised_img = denoised_img.astype(np.uint8)

    return denoised_img


# ------------------ Clustering --------------------
# --------------------------------------------------

folder = "D:\MCV\M1\P1\W5\BBDD"

print('loading images..')
ordImgs, ordIds = load_images(folder)
print('images loaded')


# -------- Compute image descriptors for clustering --------

image_descriptors = []
for i in range(0, len(ordImgs)):

    print('processing painting: ' + str(i))

    painting = ordImgs[i]
    # painting = denoise_img(painting)

    ## Extract descriptor - Change to the wanted one
    # descriptor = desc_over128(paint[:,:,1])
    # descriptor = descriptor_mean_HSV(painting)
    # descriptor = descriptor_mean_BGR(painting)
    # descriptor = descriptor_HSV_H6(painting)
    # descriptor = descriptor_HSV_LapGrad(painting)
    # descriptor = descriptor_Gradients(painting)
    # descriptor = descriptor_labhist(painting)
    # descriptor = desc_rgb_lapl(painting)
    descriptor = desc_rgb_lapl_sat(painting)
    # descriptor = desc_rgb_hue_sat(painting)


    image_descriptors.append(descriptor)

# Transform descriptors to float (preprocess for kmeans)
image_descriptors = np.array(image_descriptors,np.float32)

# # Normalize features
# for i in range(image_descriptors.shape[1]):
#     max_feat = np.max(image_descriptors[:,i])
#     image_descriptors[:,i] = image_descriptors[:,i]/max_feat
    # if i==0:
    #     image_descriptors[:, i] = image_descriptors[:, i] * 1.25


# ----------------- Compute clusters with K-means ------------------------------

# define criteria, number of clusters(K) and apply kmeans()
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K = 10
ret,label,center=cv.kmeans(image_descriptors,K,None,criteria,10,cv.KMEANS_RANDOM_CENTERS)


# ----------------- Save paintings in folders by room ------------------------------

# Create a folder to store the clustered paintings
dir = 'cluster'
if not os.path.exists(dir):
    os.mkdir(dir)
subdirs = []
for i in range(K):
    subdir = os.path.join(dir,"Room" + str(i))
    if not os.path.exists(subdir):
        os.mkdir(subdir)
    subdirs.append(subdir)

# label = label.reshape(len(label))

# Save images with room information
for i in range(len(label)):
    cv2.imwrite(dir + "\R" +  str([label[i]])  + "P000" + str(i) + ".png", ordImgs[i])


# Calculate distances and sort images by distance to the correspondent cluster centroid
distances = []
im_ids = []
for i in range(K):
    cluster_dist = []
    cluster_img_ids = []
    for j in range(len(image_descriptors)):
        if label[j] == i:
            dist = euclid_dist(image_descriptors[j], center[label[j]])
            imgid = j
            cluster_dist.append(dist)
            cluster_img_ids.append(imgid)
    #ordenenate dists and ids
    cluster_dist, cluster_img_ids = zip(*sorted(zip(cluster_dist, cluster_img_ids)))
    distances.append(cluster_dist)
    im_ids.append(cluster_img_ids)

# Show best 5  images for each cluster
fig = plt.figure()
for i in range(K):
    for j in range(5):
        if j < len(im_ids[i]):
            m_rgb = cv2.cvtColor(ordImgs[im_ids[i][j]], cv2.COLOR_BGR2RGB)
            plt.subplot(5,10,(i+1+j*10)), plt.imshow(m_rgb)
            plt.axis('off')
plt.show()


# Representation of the feature dimension
t = ['gray','red','chocolate','orange','gold','green','cyan','blue','violet','slateblue']
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.set_xlabel('RGB')
ax.set_ylabel('Gradient \"Laplacian\"')
ax.set_zlabel('Saturation')
x,y,z = zip(*image_descriptors)
for i in range(K):
    x_sub = [l for (l, v) in zip(x,label) if v==i]
    y_sub = [l for (l, v) in zip(y,label) if v==i]
    z_sub = [l for (l, v) in zip(z,label) if v==i]
    ax.scatter(x_sub, y_sub, z_sub, marker='o', c=t[i])
    ax.scatter(center[i][0], center[i][1], center[i][2], marker='^', c=t[i])
plt.show()
