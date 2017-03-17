from moviepy.editor import VideoFileClip
from vehicle_searcher import VehicleSearcher
from sliding_window import SlidingWindow

class VideoCarDetection:
    def __init__(self, classifier, featurizer, video_file, output_file):
        self.video_file = video_file
        self.output_file = output_file

        self.prev_bboxes = [] 
        self.featurizer = featurizer
        self.classifier = classifier

        slider0 = SlidingWindow(featurizer, classifier, x_start_stop=[None,None], y_start_stop=[350, 450], xy_window=[20,20])
        slider1 = SlidingWindow(featurizer, classifier, x_start_stop=[None,None], y_start_stop=[400, 500], xy_window=[32,32])
        slider2 = SlidingWindow(featurizer, classifier, x_start_stop=[None,None], y_start_stop=[400, 525], xy_window=[48,48], xy_overlap=(0.25, 0.25))
        slider3 = SlidingWindow(featurizer, classifier, x_start_stop=[None,None], y_start_stop=[400, 550], xy_window=[64,64], xy_overlap=(0.25, 0.25))
        slider4 = SlidingWindow(featurizer, classifier, x_start_stop=[None,None], y_start_stop=[400, 600], xy_window=[96,96], xy_overlap=(0.25, 0.25))
        slider5 = SlidingWindow(featurizer, classifier, x_start_stop=[None,None], y_start_stop=[400, 660], xy_window=[128,128], xy_overlap=(0.25, 0.25))

        self.searcher=VehicleSearcher([slider1, slider2, slider3, slider4, slider5])

    def add_bboxes(self, bboxes):
        self.prev_bboxes.extend(bboxes)
        if len(self.prev_bboxes) > 15:
            # throw out oldest rectangle set(s)
            self.prev_bboxes = self.prev_bboxes[len(self.prev_bboxes)-15:]

    def detect(self, img):
        bboxes = self.searcher.detect_boxes(img)
        self.add_bboxes(bboxes)
        threshold = 2
        heatmap, draw_img = self.searcher.annotate_image(img, self.prev_bboxes, threshold)

        return draw_img

    def annotate_video(self):
        """ Given input_file video, save annotated video to output_file """
        video = VideoFileClip(self.video_file)
        annotated_video = video.fl_image(self.detect)
        annotated_video.write_videofile(self.output_file, audio=False)