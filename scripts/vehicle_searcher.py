import numpy as np
import cv2

from scipy.ndimage.measurements import label
from sliding_window import SlidingWindow

class VehicleSearcher():
    def __init__(self, sliders):
        self.sliders = sliders
        self.threshold = 1

    def add_heat(self, bboxes, heatmap):
        # Iterate through list of bboxes
        for box in bboxes:
            # Add += 1 for all pixels inside each bbox
            # Assuming each "box" takes the form ((x1, y1), (x2, y2))
            heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

        # Return updated heatmap
        return heatmap# Iterate through list of bboxes
        
    def apply_threshold(self, heatmap, threshold):
        # Zero out pixels below the threshold
        heatmap[heatmap <= threshold] = 0
        # Return thresholded map
        return heatmap

    def draw_labeled_bboxes(self, img, labels):
        # Iterate through all detected cars
        carboxes = []
        for car_number in range(1, labels[1]+1):
            # Find pixels with each car_number label value
            nonzero = (labels[0] == car_number).nonzero()
            # Identify x and y values of those pixels
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])
            # Define a bounding box based on min/max x and y
            bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
            carboxes.append(bbox)
            # Draw the box on the image
            cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
        # Return the image
        return carboxes, img

    def annotate_image(self, img, bboxes, threshold):
        heat = np.zeros_like(img[:,:,0]).astype(np.float)
        heat = self.add_heat(bboxes, heat)
        heat = self.apply_threshold(heat, threshold)
        heatmap = np.clip(heat, 0, 255)
        labels = label(heatmap)
        carboxes, draw_img = self.draw_labeled_bboxes(np.copy(img), labels)
        return carboxes, heatmap, draw_img

    def detect(self, img):
        bboxes = []
        for slider in self.sliders:
            windows = slider.detect(img)
            bboxes.extend(windows)

        heat = np.zeros_like(img[:,:,0]).astype(np.float)
        heat = self.add_heat(bboxes, heat)
        heat = self.apply_threshold(heat, self.threshold)
        heatmap = np.clip(heat, 0, 255)
        labels = label(heatmap)
        carboxes, draw_img = self.draw_labeled_bboxes(np.copy(img), labels)
        return carboxes, heatmap, draw_img

    def detect_bboxes(self, img):
        bboxes = []
        for slider in self.sliders:
            windows = slider.detect(img)
            bboxes.extend(windows)

        return bboxes