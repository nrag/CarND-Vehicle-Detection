import numpy as np
import pickle
import cv2
import glob
import pickle 
import os

from tqdm import tqdm
from image_features import ImageFeatures
from vehicle_classifier import VehicleClassifier

def get_best_parameters():
    if os.path.isfile('./best_param.p'):
        f = open('best_param.p', 'rb')
        return pickle.load(f)

    car_images = glob.glob('../vehicles/**/*.png')
    noncar_images = glob.glob('../non-vehicles/**/*.png')

    best_score = 0
    best_params = {}
    best_classifier = None

    bin_size = (32, 32)
    hist_bins = 32 
    hist_bin_range = 32
    with tqdm(total=1350) as pbar:
        for cspace in ['HSV', 'LUV', 'HLS', 'YUV', 'YCrCb', 'RGB']:
            for hog_channel in [0, 1, 2, 3]:
                for hog_orient in [6, 9, 10, 11, 12]:
                    for hog_pix_per_cell in [4, 6, 8, 10, 16]:
                        for hog_cell_per_block in [1, 2, 3]:
                            featurizer = ImageFeatures(color_space=cspace, 
                                             bin_size=bin_size, 
                                             hist_bins=hist_bins, 
                                             hist_bin_range=(0,hist_bin_range), 
                                             hog_channel=hog_channel,
                                             hog_orient=hog_orient, 
                                             hog_pix_per_cell=hog_pix_per_cell, 
                                             hog_cell_per_block=hog_cell_per_block)
                            car_features = featurizer.featurize(car_images)
                            noncar_features = featurizer.featurize(noncar_images)
                            classifier = VehicleClassifier(car_features, noncar_features)
                            score = classifier.fit()
                            if (score > best_score):
                                print('Classification accuracy improved to: ', score)
                                best_score = score
                                best_classifier = classifier
                                best_params['cspace'] = cspace
                                best_params['bin_size'] = bin_size
                                best_params['hist_bins'] = hist_bins
                                best_params['hist_bin_range'] = hist_bin_range
                                best_params['hog_channel'] = hog_channel
                                best_params['hog_orient'] = hog_orient
                                best_params['hog_pix_per_cell'] = hog_pix_per_cell
                                best_params['hog_cell_per_block'] = hog_cell_per_block
                            
                            pbar.update(1)

    print('Best accuracy = ', best_accuracy)
    f = open('best_params.p', 'wb')
    pickle.dump(best_params, f)
    return best_params

best_params = get_best_parameters()
print("Best parameters are: Color Space =  %s, bin_size = %s, hist_bins = %d, hist_bin_range = (0, %d), hog_orient = %d, hog_pix_per_cell = %d, hog_pix_per_block = %d" % (best_params['cspace'], best_params['bin_size'], best_params['hist_bins'], best_params['hist_bin_range'], best_params['hog_orient'], best_params['hog_pix_per_cell'], best_params['hog_cell_per_block']))
