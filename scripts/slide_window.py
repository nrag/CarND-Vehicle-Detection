# Define a function that takes an image,
# start and stop positions in both x and y, 
# window size (x and y dimensions),  
# and overlap fraction (for both x and y)
def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None], 
                    xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
    # If x and/or y start/stop positions not defined, set to image size
    # Compute the span of the region to be searched    
    # Compute the number of pixels per step in x/y
    # Compute the number of windows in x/y
    if x_start_stop[0] == None:
        x_start_stop[0] = 0
    if x_start_stop[1] == None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] == None:
        y_start_stop[0] = 0
    if y_start_stop[1] == None:
        y_start_stop[1] = img.shape[0]
        
    # Initialize a list to append window positions to
    window_list = []
    # Loop through finding x and y window positions
    #     Note: you could vectorize this step, but in practice
    #     you'll be considering windows one by one with your
    #     classifier, so looping makes sense
        # Calculate each window position
        # Append window position to list
    # Return the list of windows
    current_window = [x_start_stop[0], y_start_stop[0]]
    while (current_window[0] + xy_window[0] <= x_start_stop[1]):
        current_window[1] = y_start_stop[0]
        while (current_window[1] + xy_window[1] <= y_start_stop[1]):
            window_start = (current_window[0], current_window[1])
            window_end = (window_start[0] + xy_window[0], window_start[1] + xy_window[1])
            window_list.append((window_start, window_end))
            current_window[1] += int(xy_window[1] * xy_overlap[1])
        current_window[0] += int(xy_window[0] * xy_overlap[0])
    return window_list