import numpy as np
import matplotlib.pyplot as plt
import cv2
#working for rgb and grayscale
def convolve(image, kernel):
    if isinstance(image, str):
        image = plt.imread(image)
    if image.ndim == 2:  
        H, W = image.shape
        C = 1
        image = np.expand_dims(image, axis=-1)  
    else:
        H, W, C = image.shape
    kH, kW = kernel.shape
    padded = np.pad(image, ((2,2),(2,2),(0,0)), mode='constant')
    out_H = H - kH + 2*2 + 1
    out_W = W - kW + 2*2 + 1
    output = np.zeros((out_H, out_W, C))
    for c in range(C):
        for i in range(out_H):
            for j in range(out_W):
                region = padded[i:i+kH, j:j+kW, c]
                output[i, j, c] = np.sum(region * kernel)
    if C == 1:
        output = output[:, :, 0]
    return output.astype(np.float32)

img = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)
fig, axes = plt.subplots(2, 2, figsize=(8, 8))

axes[0, 0].imshow(img, cmap='gray')
axes[0, 0].set_title('Original Image')
axes[0, 0].axis('off')

axes[0, 1].imshow(convolve(img, np.ones((5, 5)) / 25), cmap='gray')
axes[0, 1].set_title('Box Filter')
axes[0, 1].axis('off')

axes[1, 0].imshow(convolve(img, np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])), cmap='gray')
axes[1, 0].set_title('Horizontal Sobel Filter')
axes[1, 0].axis('off')

axes[1, 1].imshow(convolve(img, np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])), cmap='gray')
axes[1, 1].set_title('Vertical Sobel Filter')
axes[1, 1].axis('off')

plt.show()
