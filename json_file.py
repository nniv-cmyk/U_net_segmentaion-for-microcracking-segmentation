import numpy as np
import json
import cv2

f = open("./json/polygon/train_poly_coco6.json", 'r')
data = json.load(f)
print(data)

image_dir = r"C:\Users\vinnr\Downloads\Manganji\image\All\train_images"
mask_dir = r"C:\Users\vinnr\Downloads\Manganji\mask\All\polygon\train_masks"

images = data["images"]
annots = data["annotations"]

for x, y in zip(images, annots):
    filename = x["file_name"]
#     print(x)
    h = x["height"]
    w = x["width"]
    
    mask = np.zeros((h,w))
    
    seg = y["segmentation"]
    
    for points in seg:
        contours = []
        for i in range(0, len(points), 2):
            contours.append((points[i], points[i+1]))
            
        contours = np.array(contours, dtype=np.int32)

        cv2.drawContours(mask, [contours], -1, 255, 1)

        cv2.imwrite(f'{mask_dir}/{filename}', mask)