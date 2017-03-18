##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/car_not_car.png
[image2]: ./examples/HOG_example.png
[image3]: ./examples/sliding_windows.jpg
[image4]: ./examples/sliding_window.jpg
[image5]: ./examples/bboxes_and_heat1.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[image8]: ./examples/test_images_result.png
[video1]: ./project_video_out.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points

###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it! All the files can be found in `./scripts/*.py`. You can also look at `./scripts/VehicleDetection.ipynb`.

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the first code cell of the IPython notebook (or in lines # through # of the file called `some_file.py`).  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image2]

####2. Explain how you settled on your final choice of HOG parameters.

I did an exhaustive search across all the parameters (see `scripts/feature_search.py` lines 29-63). I iterated through all the parameters training a Linear SVM for each parameter combination and choosing the parameter combination with the best accuracy. I found that the following parameter produced the best accuracy on the test set:
      
      * Color Space =  LUV      
      * Spatial bin size = (32, 32)   
      * Histogram bins = 32
      * Histogram bin range = (0, 32)
      * HOG channel = 'V' channel     
      * HOG orientations = 11      
      * HOG pixels_per_cell = 8      
      * HOG cells_per_block = 3
    
I got a test accuracy of 0.9879 for these parameters.

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

As I described above, I did an exhaustive search across all the parameters (see `scripts/feature_search.py` lines 29-63) by training a classifier for each of the combinations. I used the code in `./scripts/vehicle_classifier.py` to train my classifier. The class `VehicleClassifier` receives the spatial binning, color histogram and HOG features for each of the car/not-car images, splits them randomly into training and test sets, trains a Linear SVM and calculates the test accuracy.

      * Color Space =  LUV      
      * Spatial bin size = (32, 32)   
      * Histogram bins = 32
      * Histogram bin range = (0, 32)
      * HOG channel = 'V' channel     
      * HOG orientations = 11      
      * HOG pixels_per_cell = 8      
      * HOG cells_per_block = 3
    
I got a test accuracy of 0.9879 for these parameters.

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

My sliding window search is implemented in the two files `./scripts/sliding_window.py` and `./scripts/vehicle_searcher.py`. VehicleSearcher class receives a list of SlidingWindow objects. SlidingWindow objects slides a window from start to stop and returns the set of windows where it detects the cars. VehicleSearcher slides through all the sliding window configurations and accumulates all the windows together. Then it creates a heatmap and the final car positions are calculated on the thresholded heatmaps.

I decided to use the following six configurations for the sliding windows:
      
      * Slider1 = x_start_stop=[550,900], y_start_stop=[370, 430], xy_window=[32,32], xy_step=(5, 5)
      * Slider2 = x_start_stop=[480,1200], y_start_stop=[400, 470], xy_window=[48,48], xy_step=(5, 5)
      * Slider3 = x_start_stop=[450,None], y_start_stop=[420, 500], xy_window=[64,64], xy_step=(5, 5)
      * Slider4 = x_start_stop=[400,None], y_start_stop=[400, 530], xy_window=[96,96], xy_step=(5, 5)
      * Slider5 = x_start_stop=[350,None], y_start_stop=[430, 560], xy_window=[128,128], xy_step=(5, 5)
      * Slider6 = x_start_stop=[350,None], y_start_stop=[500, 690], xy_window=[192,192], xy_step=(5, 5)

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

I searched on 6 scales using YUV 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image8]
---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)

Here's a [link to my video result](./project_video_out.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are eight frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The main issue with my implementation is the performance. To detect cars in a single image, it takes 10 secs. This cannot be used in real time. The second issue that I see is that I don't track the cars from frame to frame to ensure that the car detections in successive frames are within some margins of each other.
