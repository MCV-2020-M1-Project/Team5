
import cv2
from utils import *
from masks import *
from textDetection import *
import matplotlib.pyplot as plt
from multiresolution import *

# Import images and sort them

# pathQ2 = 'qsd2_w2'  # Path to the images
# q2Gt_mask, q2IdsMs = load_images(pathQ2)
# # Compute binary masks
# q2Im = []
# name = []
# idsq2 = []
# for file in os.listdir(pathQ2):
#     if file.endswith('.jpg'):
#         im = cv2.imread(os.path.join(pathQ2, file))
#         q2Im.append(im)
#         name = file.split('.')[0]
#         idsq2.append(int(name))
# idsq2Ord, q2ImOrd = zip(*sorted(zip(idsq2, q2Im)))

pathQ2 = 'qsd2_w2'  # Path to the query images
listQImgs , idsQImgs = load_images(pathQ2)
pathM2 = 'BBDD'
listMImgs,idsMImgs = load_images(pathM2)
k = 10 # Number of best predictions

pkl_path = "result.pkl"
results = []

# Crop images for each painting
crped_images = crop_imgarray(listQImgs)


# For each cropped image, a textbox is seacrhed and removed and a multiresolution is computed
for img in crped_images:

    results_paiting = []

    for painting in img:

        txtbox = calculate_txtbox(painting)
        x, y, w, h = txtbox
        mask = np.ones(painting.shape[:2], np.uint8)
        mask[y:y + h, x:x + w] = -1

        d3mask = np.zeros_like(painting)
        d3mask[:,:,0] = mask
        d3mask[:,:,1] = mask
        d3mask[:,:,2] = mask


        painting_tbx = painting * d3mask

        # Returns ids of Museum images sorted by similarity
        klist = compute_multiresolution(painting_tbx,listMImgs,idsMImgs,4)
        klist = klist[:k]

        results_paiting.append(klist)

    results.append(results_paiting)



outfile = open(pkl_path, 'wb')
pickle.dump(results, outfile)
outfile.close()



# Pickle test
infile = open(pkl_path,'rb')
new_dict = pickle.load(infile)
infile.close()
print(new_dict)