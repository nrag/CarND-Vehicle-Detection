class SlidingWindow:
    def __init__(self, 
                 featurizer,
                 classifier,
                 x_start_stop=[None, None], 
                 y_start_stop=[None, None], 
                 xy_window=(64, 64), 
                 xy_overlap=(0.5, 0.5)):
        self.featurizer = featurizer
        self.classifier = classifier
        self.x_start_stop = x_start_stop
        self.y_start_stop = y_start_stop
        self.xy_window = xy_window
        self.xy_overlap = xy_overlap

    def check_vehicle(self, img, window):
        window_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))
        features = self.featurizer.extract_feature(window_img)
        return self.classifier.predict(features)

    def detect(self, img):
        # If x and/or y start/stop positions not defined, set to image size
        # Compute the span of the region to be searched    
        # Compute the number of pixels per step in x/y
        # Compute the number of windows in x/y
        if self.x_start_stop[0] == None:
            self.x_start_stop[0] = 0
        if self.x_start_stop[1] == None:
            self.x_start_stop[1] = img.shape[1]
        if self.y_start_stop[0] == None:
            self.y_start_stop[0] = 0
        if self.y_start_stop[1] == None:
            self.y_start_stop[1] = img.shape[0]
            
        # Initialize a list to append window positions to
        window_list = []
        # Loop through finding x and y window positions
        #     Note: you could vectorize this step, but in practice
        #     you'll be considering windows one by one with your
        #     classifier, so looping makes sense
            # Calculate each window position
            # Append window position to list
        # Return the list of windows
        current_window = [self.x_start_stop[0], self.y_start_stop[0]]
        while (current_window[0] + self.xy_window[0] <= self.x_start_stop[1]):
            current_window[1] = self.y_start_stop[0]
            while (current_window[1] + self.xy_window[1] <= self.y_start_stop[1]):
                window_start = (current_window[0], current_window[1])
                window_end = (window_start[0] + self.xy_window[0], window_start[1] + self.xy_window[1])
                prediction = check_vehicle((window_start, window_end))
                if prediction == 1:
                    window_list.append(img, (window_start, window_end))
                
                current_window[1] += int(self.xy_window[1] * self.xy_overlap[1])
            current_window[0] += int(self.xy_window[0] * self.xy_overlap[0])
        return window_list