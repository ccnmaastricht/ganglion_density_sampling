from GDS import GanglionDensitySampling
import numpy as np
import cv2
import matplotlib.pyplot as plt

image = 'C:/Users/Danny/PycharmProjects/GDS/test.jpg'

GDS1 = GanglionDensitySampling()
im = GDS1.load_image(image)
GDS1.show_image(im)
im2, msk = GDS1.resample_image(image=im)
im3 = GDS1.mask(im2,msk)
GDS1.show_image(im3)