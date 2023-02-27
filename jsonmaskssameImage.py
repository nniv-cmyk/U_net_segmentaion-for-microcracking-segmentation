import numpy as np
import json
import cv2
import os

# Load the COCO format json file
f = open("./json/polygon/train_nozoom50_coco.json", 'r')
data = json.load(f)

# Set the directory paths for images and masks
image_dir = r"C:\Users\vinnr\Downloads\Manganji\image\All\train_images"
mask_dir = r"C:\Users\vinnr\Downloads\Manganji\mask\All\polygon\train_masks"

# Get the list of images and annotations from the COCO json file
images = data["images"]
annots = data["annotations"]
print(len(images))
# Create a dictionary to store the masks for each image
image_masks = {}

# Loop through each image and its annotations
for x in images:
    # Get the file name, height, and width of the image
    filename = x["file_name"]
    h = x["height"]
    w = x["width"]

    # If this is the first annotation for the image, create an empty mask
    if filename not in image_masks:
        image_masks[filename] = np.zeros((h, w), dtype=np.uint8)

    # Loop through each annotation for the image
    for y in annots:
        if y['image_id'] == x['id']:
            # Get the list of segmentations for the annotation
            seg = y["segmentation"]

            # Loop through each segmentation and draw it on the mask
            for points in seg:
                contours = []
                for i in range(0, len(points), 2):
                    contours.append((points[i], points[i+1]))

                contours = np.array(contours, dtype=np.int32)

                # Draw the segmentation on the mask
                cv2.drawContours(image_masks[filename], [contours], -1, 255, -1)

    # Invert the mask to get black foreground and white background
    # image_masks[filename] = cv2.bitwise_not(image_masks[filename])

# Save the masks for each image
for filename, mask in image_masks.items():
    # Create the mask directory if it doesn't exist
    os.makedirs(mask_dir, exist_ok=True)
    # Save the mask for the current image
    cv2.imwrite(os.path.join(mask_dir, filename), mask)
