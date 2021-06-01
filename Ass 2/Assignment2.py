#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CS 4442 - Assignment 2
@author: Shaan Verma
Student #: 250804514
"""

# Adding required packges
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np

# Loading data file
file = np.loadtxt('faces.dat')

'''
Part 3A
'''
# Reshaping and displaying 100th image
pic = file[99].reshape((64,64))
imgplot = plt.imshow(pic)
valueMean = file.mean(axis=1)
print("Part 3A ... Complete\n")

'''
Part 3B
'''
plt.figure()
file = file - valueMean[:, np.newaxis]
pic2 = file[99].reshape((64,64))
imgplot = plt.imshow(pic2)
print("Part 3B ... Complete\n")


'''
Part 3C
'''
covMat = PCA(n_components = 400)
covMat.fit(file)
var = covMat.explained_variance_ratio_ 

arraySort = np.sort(var)[::-1]

plt.figure()
plt.ylabel('Rev Sorted Variance')
plt.xlabel('Number of Features')
plt.title('PCA Analysis Graph')
plt.plot(arraySort)
print("Part 3C ... Complete\n")


'''
Part 3D

- 400th feature has no more valuable info for the pca. 
- Thus, the last eigenvalue is a min. No variance is captured by the feature 
'''
print("Part 3D ... Answer in comments\n")

'''
Part 3E

- The variance will be printed in descending order.
- Variance does not drop too much anymore after feature 34.
- Majority of the variance is contained within the first 34 features
- Since there is a large dropoff in the eigenvalues at the 34th feature point, this would be a good cutoff.
'''

print("\n--------- Part 3E ---------\n")
print(arraySort[:100])
print("#######################################")

for z in range(1, arraySort.shape[0]):
  if abs(arraySort[z] - arraySort[z-1]) <= 0.00005:
      print("# Values stop increasing at index", z-1, " #")
      break
print("#######################################")

print("\nPart 3E Complete ... Answer in comments\n")

'''
Part 3F
'''

print("\n--------- Part 3F ---------")
print('\nComponents: ', covMat.components_.shape)

# plot data - sklearn orders the components
for i in range(5):
  vector =covMat.components_[i,:]
  component = vector.reshape((64,64))
  plt.figure()
  imgplot = plt.imshow(component)
  plt.show()
print("Part 3F ... Complete\n")

'''
Part 3G
'''
vals = [9,99,199,398]
for k in vals:
  comp = covMat.components_[k,:].reshape((1,4096))
  image_col = file[99].reshape((1,4096))
  reconstructed = np.dot(comp,np.dot(comp.T, image_col))
  image_col = reconstructed.reshape((64,64))
  plt.figure()
  imgplot = plt.imshow(image_col)
  plt.show()
print("Part 3G ... Complete\n")




