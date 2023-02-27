import numpy as np
import cv2

# Load the image and convert it to grayscale
image = cv2.imread("./image/C_6.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Threshold the image to create a binary mask
thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]


# Set the background pixels to zero
image[thresh == 255] = 0

# Display the image with the background set to zero
cv2.imwrite("Image_C.png", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
