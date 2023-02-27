import numpy as np
import json
import cv2
import pycocotools.mask as mask_utils
from skimage.draw import polygon2mask

f = open("./json/polygon/train_poly_coco.json", 'r')
data = json.load(f)
# print(data)

image_dir = r"C:\Users\vinnr\Downloads\Manganji\image\All\train_images"
mask_dir = r"C:\Users\vinnr\Downloads\Manganji\mask\All\polygon\train_masks"

images = data["images"]
print(len(images))
annots = data["annotations"]

for x, y in zip(images, annots):
    filename = x["file_name"]
    h = x["height"]
    w = x["width"]
    
    mask = np.zeros((h,w))
    
    seg = y["segmentation"]
    
    for i in range(len(seg)):
        poly = np.array(seg[i]).reshape((len(seg[i])//2, 2))
        binary_mask = polygon2mask((h,w), poly)
        rle = mask_utils.encode(np.array(binary_mask, order='F', dtype=np.uint8))
        mask += mask_utils.decode(rle) * (i+1)
            
    cv2.imwrite(f'{mask_dir}/{filename}', mask)


