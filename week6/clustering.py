from skimage.io import imread
from skimage import img_as_float
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt

image = imread('parrots.jpg')
img = img_as_float(image)

r = np.array([img[:,:, 0].ravel()]).T
g = np.array([img[:,:, 1].ravel()]).T
b = np.array([img[:,:, 2].ravel()]).T
result = np.hstack((r, g, b))

clustersNumber = 11
kmeans = KMeans(n_clusters=clustersNumber, init="k-means++", random_state=241)
kmeans.fit(result)

def recreate_image(codebook, labels, w, h):
    """Recreate the (compressed) image from the code book & labels"""
    d = codebook.shape[1]
    image = np.zeros((w, h, d))
    label_idx = 0
    for i in range(w):
        for j in range(h):
            image[i][j] = codebook[labels[label_idx]]
            label_idx += 1
    return image

reduced_image = recreate_image(kmeans.cluster_centers_, kmeans.labels_, image.shape[0], image.shape[1])

mse = np.mean((img - reduced_image) ** 2)
psnr = 10 * np.log10(1.0 / mse)
print("PSNR =", psnr, "> 20; optimal number of clusters is", clustersNumber)

plt.imshow(reduced_image)
plt.show()
