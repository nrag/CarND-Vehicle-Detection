import cv2
import matplotlib.image as mpimg
import numpy as np
from skimage.feature import hog

class ImageFeatures:
    def __init__(self,
                 color_space='RGB', 
                 bin_size=(32,32), 
                 hist_bins=32, 
                 hist_bin_range=(0,32),
                 hog_channel=3,
                 hog_orient=9, 
                 hog_pix_per_cell=8, 
                 hog_cell_per_block=2):
        self.color_space = color_space
        self.bin_size = bin_size
        self.hist_bins = hist_bins
        self.hist_bin_range = hist_bin_range
        self.hog_channel=hog_channel
        self.hog_orient = hog_orient
        self.hog_pix_per_cell = hog_pix_per_cell
        self.hog_cell_per_block = hog_cell_per_block

    def convert_image(self, img):
        if self.color_space == 'HSV':
            return cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif self.color_space == 'LUV':
            return cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        elif self.color_space == 'HLS':
            return cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        elif self.color_space == 'YUV':
            return cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        elif self.color_space == 'YCrCb':
            return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
        else:
            return np.copy(img)

    def getChannel(self, img):
        feature_image = self.convert_image(img)
        if self.hog_channel < 3:
            return feature_image[:, :, self.hog_channel]
        else:
            return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    def bin_spatial(self, img):
        features = cv2.resize(img, self.bin_size).ravel()

        # Return the feature vector
        return features

    # Define a function to compute color histogram features  
    def color_hist(self, img):
        # Compute the histogram of the color channels separately
        channel1_hist = np.histogram(img[:,:,0], bins=self.hist_bins, range=self.hist_bin_range)
        channel2_hist = np.histogram(img[:,:,1], bins=self.hist_bins, range=self.hist_bin_range)
        channel3_hist = np.histogram(img[:,:,2], bins=self.hist_bins, range=self.hist_bin_range)
        # Concatenate the histograms into a single feature vector
        hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
        # Return the individual histograms, bin_centers and feature vector
        return hist_features

    # Define a function to return HOG features and visualization
    def get_hog_features(self, img, vis=False):
        gray = self.getChannel(img)
        # Call with two outputs if vis==True
        if vis == True:
            features, hog_image = hog(gray, orientations=self.hog_orient, pixels_per_cell=(self.hog_pix_per_cell, self.hog_pix_per_cell),
                                      cells_per_block=(self.hog_cell_per_block, self.hog_cell_per_block), transform_sqrt=True, 
                                      visualise=vis, feature_vector=True)
            return features, hog_image
        # Otherwise call with one output
        else:      
            features = hog(gray, orientations=self.hog_orient, pixels_per_cell=(self.hog_pix_per_cell, self.hog_pix_per_cell),
                           cells_per_block=(self.hog_cell_per_block, self.hog_cell_per_block), transform_sqrt=True, 
                           visualise=vis, feature_vector=True)
            return features

    def extract_feature(self, img):
        feature_image = self.convert_image(img)
        spatial_features = self.bin_spatial(feature_image)
        hist_features = self.color_hist(feature_image)
        hog_features = self.get_hog_features(img)
        return np.concatenate((spatial_features, hist_features, hog_features))

    def featurize(self, img_paths):
        # Create a list to append feature vectors to
        features = []
        # Iterate through the list of images
            # Read in each one by one
            # apply color conversion if other than 'RGB'
            # Apply bin_spatial() to get spatial color features
            # Apply color_hist() to get color histogram features
            # Append the new feature vector to the features list
        # Return list of feature vectors
        for img_path in img_paths:
            img = mpimg.imread(img_path)
            img_features = self.extract_feature(img)
            features.append(img_features)
        return features