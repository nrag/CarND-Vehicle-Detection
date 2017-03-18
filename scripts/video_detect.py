import cv2
import matplotlib.image as mpimg
import numpy as np

from moviepy.editor import VideoFileClip
from vehicle_searcher import VehicleSearcher
from sliding_window import SlidingWindow

class VideoCarDetection:
    def __init__(self, classifier, featurizer, video_file, output_file, save_frames=False):
        self.video_file = video_file
        self.output_file = output_file

        self.save_frames = save_frames
        self.featurizer = featurizer
        self.classifier = classifier
        self.window_size = 5
        self.previous_bboxes = []
        self.frame_count = 0

        slider0 = SlidingWindow(featurizer, classifier, x_start_stop=[550,900], y_start_stop=[350, 450], xy_window=[20,20])
        slider1 = SlidingWindow(featurizer, classifier, x_start_stop=[600,900], y_start_stop=[370, 430], xy_window=[32,32], xy_step=(5, 5))
        slider2 = SlidingWindow(featurizer, classifier, x_start_stop=[520,1200], y_start_stop=[400, 470], xy_window=[48,48], xy_step=(5, 5))
        slider3 = SlidingWindow(featurizer, classifier, x_start_stop=[450,None], y_start_stop=[420, 500], xy_window=[64,64], xy_step=(5, 5))
        slider4 = SlidingWindow(featurizer, classifier, x_start_stop=[400,None], y_start_stop=[400, 530], xy_window=[96,96], xy_step=(5, 5))
        slider5 = SlidingWindow(featurizer, classifier, x_start_stop=[350,None], y_start_stop=[430, 560], xy_window=[128,128], xy_step=(5, 5))
        slider6 = SlidingWindow(featurizer, classifier, x_start_stop=[350,None], y_start_stop=[500, 690], xy_window=[192,192], xy_step=(5, 5))

        self.searcher=VehicleSearcher([slider1, slider2, slider3, slider4, slider5])

    def get_merged_bboxes(self):
        merged = []
        for bboxes in self.previous_bboxes:
            merged.extend(bboxes)
        return merged

    def add_bboxes(self, bboxes):
        self.previous_bboxes.append(bboxes)
        if len(self.previous_bboxes) > self.window_size:
            # throw out oldest rectangle set(s)
            _ = self.previous_bboxes.pop(0)
        return self.get_merged_bboxes()

    def detect(self, img):
        newboxes = self.searcher.detect_bboxes(img)
        merged = self.add_bboxes(newboxes)

        threshold = self.searcher.threshold + 1
        carboxes, heatmap, draw_img = self.searcher.annotate_image(img, merged, threshold)
        if (self.save_frames):
            self.frame_count += 1
            frame_img = np.copy(img)
            for bbox in newboxes:
                cv2.rectangle(frame_img, bbox[0], bbox[1], (0,0,255), 6)
            mpimg.imsave(self.output_file.split('.mp4')[0] + '_frame_' + str(self.frame_count) + '.png', frame_img)
            mpimg.imsave(self.output_file.split('.mp4')[0] + '_heatmap_' + str(self.frame_count) + '.png', np.dstack((heatmap, heatmap, heatmap)))
        return draw_img

    def annotate_video(self):
        """ Given input_file video, save annotated video to output_file """
        self.frame_count = 0
        video = VideoFileClip(self.video_file)
        annotated_video = video.fl_image(self.detect)
        annotated_video.write_videofile(self.output_file, audio=False)