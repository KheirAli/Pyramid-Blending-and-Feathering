#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  5 13:54:33 2022

@author: alirezakheirandish
"""

import cv2 
import numpy as np
import matplotlib.pyplot as plt


img_apple = plt.imread('res08.jpeg')
img_apple = np.asarray(img_apple/255,np.float64)
img_orange = plt.imread('res09.jpeg')
img_orange = np.asarray(img_orange/255,np.float64)
img_orange = cv2.resize(img_orange,(img_apple.shape[1],img_apple.shape[0]))

# img_orange.shape,img_apple.shape
# plt.imshow(img_orange)
# img_orange = plt.imread('res09.jpeg')
# img_orange[0,0]

mask = np.zeros_like(img_apple,dtype = np.float)
mask = mask[:,:,0]
mask[:,:img_apple.shape[1]//2-100]= 1
counter = 0
for i in range(-100,20):
    counter += 1
    mask[:,img_apple.shape[1]//2+i]= 1-1/120*counter
    

# plt.imshow(mask)

G1_apple = cv2.GaussianBlur(img_apple, (9, 9), 0)
# plt.imshow(G1_apple)

L0_apple = img_apple-G1_apple
# plt.imshow((L0_apple))

# L0_apple

G2_apple = cv2.GaussianBlur(G1_apple, (9, 9), 0)
# plt.imshow(G2_apple)

L1_apple = G1_apple-G2_apple
# plt.imshow(L1_apple)

G3_apple = cv2.GaussianBlur(G2_apple, (9, 9), 0)
# plt.imshow(G3_apple)
# plt.show()
L2_apple = G2_apple-G3_apple
# plt.imshow(L2_apple)

G4_apple = cv2.GaussianBlur(G3_apple, (9, 9), 0)
# plt.imshow(G4_apple)
# plt.show()
L3_apple = G3_apple-G4_apple
# plt.imshow(L3_apple)

# plt.imshow(L3_apple*255)



G1_orange = cv2.GaussianBlur(img_orange, (9, 9), 0)
# plt.imshow(G1_orange)

L0_orange = img_orange-G1_orange
# plt.imshow((L0_orange))

# L0_apple

G2_orange = cv2.GaussianBlur(G1_orange, (9, 9), 0)
# plt.imshow(G2_orange)

L1_orange = G1_orange-G2_orange
# plt.imshow(L1_orange)

G3_orange = cv2.GaussianBlur(G2_orange, (9, 9), 0)
# plt.imshow(G3_orange)
# plt.show()
L2_orange = G2_orange-G3_orange
# plt.imshow(L2_orange)

G4_orange = cv2.GaussianBlur(G3_orange, (9, 9), 0)
# plt.imshow(G4_orange)
# plt.show()
L3_orange = G3_orange-G4_orange
# plt.imshow(L3_orange)

M = cv2.merge([mask,mask,mask])
G4 = G4_apple*M+G4_orange*(1-M)
# plt.imshow(G4)

G3 = G4+L3_apple*M+L3_orange*(1-M)
# plt.imshow(G3)

G2 = G3+L2_apple*M+L2_orange*(1-M)
# plt.imshow(G2)

G1 = G2+L1_apple*M+L1_orange*(1-M)
# plt.imshow(G1)

G0 = G1+L0_apple*M+L0_orange*(1-M)
# plt.imshow(np.uint8(G0*255))

# G0[0,0]

plt.imsave('res10.jpeg',np.uint8(G0*255))