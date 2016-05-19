# Date Created: 18 May 2016
# Author: Rohit Dilip
# Facial recognition software in Python for Princeton students, using PCA
# Faces must be stored in jpg format in a subfolder called img
# I crop my images to 150 by 150. You can use whatever you want.

import numpy as np
import cv2
import glob, os
from sklearn.preprocessing import normalize


# Creating relevant directories. We could store this all in memory, but
# that's a little clunky.
os.mkdir("grayimg")
os.mkdir("adjusted")
# Converting all images to grayscale, and getting average image

problems = []
num_images = len(glob("img/*.jpg"))
image_num = range(num_images * 2)
mu_image = np.zeros((150, 150))
num_images = 0

for img in glob("img/*.jpg"):
    try:
        image = cv2.imread(img, 0)[:150, :150]
        pre, ext = os.path.splitext(img)
        cv2.imwrite("grayimg/{0}.png".format(image_num.pop(0)), image)
        mu_image += image
        num_images += 1
    except:
        problems.append(img)

mu_image = mu_image / float(num_images)

# Here, we save the grayscale images minus the average image to a folder
# titled adjusted. We need to mess around with the type of the image's 
# bits (changing to float) to avoid int versus float subtraction problems
for img in xrange(1, num_images):
    image = cv2.imread("grayimg/{0}.png".format(img))
    new_image = image.astype(np.float64)
    new_image -= mu_image
    cv2.imwrite('adjusted/{0}.png'.format(img), new_image)

# Now we find the eigenvalues and vectors of the covariance matrix. This would clearly be a difficult problem, so we use a few mathematical tricks, which I describe elsewhere. 
vectors = []
for img in glob("adjusted/*.png"):
    image = cv2.imread(img, 0)
    vectors.append(image.reshape((150 * 150,)))
V = np.array(vectors)
A = np.transpose(V)
L = np.dot(np.transpose(A), A)
eigenValues, eigenVectors = np.linalg.eig(L)

# This is a fancy trick just to get the eigenvalues and vectors nicely ordered in decreasing form.
idx = eigenValues.argsort()[::-1]   
eigenValues = eigenValues[idx]
eigenVectors = eigenVectors[:,idx]

eigenVectors_ = np.dot(A, eigenVectors)

# U has columns of eigenvalues.

U = normalize(eigenVectors_, norm = 'l1', axis = 0)

# %load 101
smallest_dist = float("inf")
smallest_img = ""
U_r = U[:,:50]
omega = np.dot(np.transpose(U_r), np.reshape(cv2.imread("george.jpg", 0)[:150,:150], (150*150, 1)) - mu)
for img in glob("img/*.jpg"):
    omega_i = np.dot(np.transpose(U_r), np.reshape(cv2.imread(img, 0)[:150,:150], (150*150, 1)) - mu)
    d = np.linalg.norm(omega - omega_i)
    if d  < smallest_dist:
        smallest_dist = d
        smallest_img = img
