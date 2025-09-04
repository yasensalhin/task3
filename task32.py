import numpy as np
import matplotlib.pyplot as plt
import cv2

img =cv2.imread('shapes.jpg')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
out =img_rgb.copy()

lower_red = np.array([100, 0, 0])
higher_red = np.array([255, 80, 80])

lower_blue=np.array([0,0,100])
higher_blue=np.array([100,100,255])

lower_black = np.array([0, 0, 0])
higher_black = np.array([50, 50, 50])

mask_blue =cv2.inRange(img_rgb,lower_blue,higher_blue)

mask_red=cv2.inRange(img_rgb,lower_red,higher_red)

mask_black=cv2.inRange(img_rgb,lower_black,higher_black)
print(mask_blue.shape)


out[mask_blue>0]=[0,0,0]
out[mask_red>0]=[0,0,255]
out[mask_black>0]=[255,0,0]


fig, axes = plt.subplots(1, 2)
axes[0].imshow(img_rgb)
axes[0].set_title('Original Image')
axes[0].axis('off')

axes[1].imshow(out)
axes[1].set_title('Processed Image')
axes[1].axis('off')

plt.show()
