import numpy as np
import matplotlib.pyplot as plt
import cv2

img =cv2.imread('shapes.jpg')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
out =img_rgb.copy()


red_mask   = (out[...,0] > 200) & (out[...,1] < 80)  & (out[...,2] < 80)
blue_mask  = (out[...,2] > 200) & (out[...,0] < 80)  & (out[...,1] < 80)
black_mask = (out[...,0] < 60)  & (out[...,1] < 60)  & (out[...,2] < 60)


out[red_mask]=[0,0,255]
out[blue_mask]=[0,0,0]
out[black_mask]=[255,0,0]


fig, axes = plt.subplots(1, 2)
axes[0].imshow(img_rgb)
axes[0].set_title('Original Image')
axes[0].axis('off')

axes[1].imshow(out)
axes[1].set_title('Processed Image')
axes[1].axis('off')

plt.show()

