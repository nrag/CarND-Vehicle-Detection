from moviepy.editor import VideoFileClip
from vehicle_searcher import VehicleSearcher
from sliding_window import SlidingWindow

class VideoCarDetection:
    def __init__(self, classifier, featurizer, video_file, output_file):
        self.video_file = video_file
        self.output_file = output_file

        self.featurizer = featurizer
        self.classifier = classifier

        slider0 = SlidingWindow(featurizer, classifier, x_start_stop=[550,900], y_start_stop=[350, 450], xy_window=[20,20])
        slider1 = SlidingWindow(featurizer, classifier, x_start_stop=[550,900], y_start_stop=[370, 430], xy_window=[32,32], xy_step=(5, 5))
        slider2 = SlidingWindow(featurizer, classifier, x_start_stop=[480,1200], y_start_stop=[400, 470], xy_window=[48,48], xy_step=(5, 5))
        slider3 = SlidingWindow(featurizer, classifier, x_start_stop=[450,None], y_start_stop=[420, 500], xy_window=[64,64], xy_step=(5, 5))
        slider4 = SlidingWindow(featurizer, classifier, x_start_stop=[400,None], y_start_stop=[400, 530], xy_window=[96,96], xy_step=(5, 5))
        slider5 = SlidingWindow(featurizer, classifier, x_start_stop=[350,None], y_start_stop=[430, 560], xy_window=[128,128], xy_step=(5, 5))
        slider6 = SlidingWindow(featurizer, classifier, x_start_stop=[350,None], y_start_stop=[500, 690], xy_window=[192,192], xy_step=(5, 5))

        self.searcher=VehicleSearcher([slider1, slider2, slider3, slider4, slider5])

    def add_bboxes(self, bboxes):
        self.prev_bboxes.extend(bboxes)
        if len(self.prev_bboxes) > 15:
            # throw out oldest rectangle set(s)
            self.prev_bboxes = self.prev_bboxes[len(self.prev_bboxes)-15:]

    def detect(self, img):
        newboxes, heatmap, draw_img = self.searcher.detect(img)

        return draw_img

    def annotate_video(self):
        """ Given input_file video, save annotated video to output_file """
        video = VideoFileClip(self.video_file)
        annotated_video = video.fl_image(self.detect)
        annotated_video.write_videofile(self.output_file, audio=False)