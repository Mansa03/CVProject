import os
import math

import cv2 as cv
import numpy as np
from Dataset import Dataset
from Items.Items import Item, Itemtype
ImageDirectory = r'../Images'

def DataPipelineAndEnhancement() -> list:
    dataset:list = []
    for  file in os.listdir(ImageDirectory):
        # Remove shadows from image using thresholding to find shadows and fill in
        if file.endswith('.jpg'):
            image = cv.imread(os.path.join(ImageDirectory, file),)
            image = cv.resize(image, (640, 480))
            hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
            _, _, v = cv.split(hsv)
            shadow_mask = v < 256
            enhanced = cv.inpaint(image, shadow_mask.astype(np.uint8), 3, cv.INPAINT_TELEA)

            # Adding a Gaussian High-Pass Filter to reduce noise and sharpen edges
            Gray = cv.cvtColor(enhanced, cv.COLOR_BGR2GRAY)

            dataset.append(Gray)

    return dataset

def main():
    #Put Images into dataset with labels
    dataset:list = DataPipelineAndEnhancement()
    #Segmentation using Canny Edge Detector
    image = dataset[9]
    image2 = dataset[6]
    cv.imshow('og', image)
    cv.waitKey(0)
    edges = cv.Canny(image, 100, 200)
    dilatedEdges = cv.dilate(edges, np.ones((9, 9), np.uint8), iterations=1)
    openEdges = cv.erode(dilatedEdges, np.ones((9, 9), np.uint8), iterations=1)
    cv.imshow('edges', openEdges)
    cv.waitKey(0)

    #Finding components using contours
    contours, hierarchy = cv.findContours(openEdges, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    edgeContours = cv.drawContours(image, contours, -1, (0, 255, 0), 3)
    cv.imshow('contours',edgeContours)
    cv.waitKey(0)
    output_image = edgeContours.copy()
    for contour in contours:
        # Draw contour
        cv.drawContours(output_image, [contour], -1, (0, 255, 0), 2)

        # Draw bounding box
        x, y, w, h = cv.boundingRect(contour)
        cv.rectangle(output_image, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # Draw centroid
        moments = cv.moments(contour)
        if moments['m00'] != 0:
            centroid = (int(moments['m10'] / moments['m00']), int(moments['m01'] / moments['m00']))
            cv.circle(output_image, centroid, 5, (255, 0, 0), -1)

    # Display the resulting image
    cv.imshow('Contours and Features', output_image)
    cv.waitKey(0)
main()

