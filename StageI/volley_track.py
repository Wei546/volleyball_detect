import cv2
import torch
import queue
import os
import sys
import argparse
import warnings
import time
import copy
from my_utils import *
from tqdm import tqdm
from datetime import datetime
from roboflow import Roboflow

class VolleyBallTracker(Roboflow):
    def __init__(self, input_path, output_path, model, confidence, show, marker, color, no_trace):
        self.input_video = input_path
        self.output_video = output_path
        self.model_name = model
        self.conf = confidence
        self.show = show
        self.marker = marker
        self.color = color
        self.no_trace = no_trace
    def ballTracker(self):
        if self.color == 'yellow':
            self.color = [0, 255, 255]
        elif self.color == 'black':
            self.color = [0, 0, 0]
        elif self.color == 'white':
            self.color = [255, 255, 255]
        elif self.color == 'red':
            self.color = [0, 0, 255]
        elif self.color == 'green':
            self.color = [0, 255, 0]
        elif self.color == 'blue':
            self.color = [255, 0, 0]
        elif self.color == 'cyan':
            self.color = [255, 255, 0]
    ###################


    ###    Start Time    ###
        t1 = datetime.now()
        ###################


        ### Capture Video ###
        video_in = cv2.VideoCapture(self.input_video)

        if (video_in.isOpened() == False):
            print("Error reading video file")
        ###################


        ### Video Writer ###
        basename = os.path.basename(self.input_video)


        if self.output_video == "":  #
            os.makedirs('VideoOutput', exist_ok=True)
            self.output_video = os.path.join(
                "VideoOutput", self.model_name + 'Track' + '_' + basename)
        else:  # check if user path exists, create otherwise
            f = os.path.split(self.output_video)[0]
            if not os.path.isdir(f):
                os.makedirs(f)

        fname = self.output_video
        fps = video_in.get(5)
        frame_width = int(video_in.get(3))
        frame_height = int(video_in.get(4))
        dims = (frame_width, frame_height)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        total_frames = video_in.get(cv2.CAP_PROP_FRAME_COUNT)
        video_out = cv2.VideoWriter(fname, fourcc, fps, dims)
        ###################


        ### Model Selection ###
        if self.model_name == 'roboflow':
            #  API key, if doesn't work, refer -->
            #  https://github.com/shukkkur/VolleyVision/discussions/5#discussioncomment-7737081
            rf = Roboflow(api_key="5lhXjuq3PEKqgKnLEZW6")
            project = rf.workspace().project("volleyball-tracking")
            model = project.version(13).model
        elif self.model_name == 'yolov7':
            model = custom(path_or_model='./yV7-tiny/weights/best.pt')
            model.conf = self.conf

        model = RoboYOLO(self.model_name, model, self.conf)
        ###################


        ### Find Desired Object ####
        bbox = (0, 0, 0, 0)

        while bbox == (0, 0, 0, 0):
            ret, frame = video_in.read()
            if not ret:
                break

            pred = model.predict(frame)
            bbox = x_y_w_h(pred, self.model_name)

        if bbox == (0, 0, 0, 0):
            raise Exception(
                "Processed the whole video but failed to detect any volleyball")
        ###################


        ### Trajectory of volleyball ###
        q = queue.deque()  # we need to save the coordinate of previous 7 frames
        for i in range(0, 8):
            q.appendleft(None)

        q.appendleft(bbox)
        ###################


        ### Initialize Tracker ###
        ok, image = video_in.read()   # get first frame
        tracker = initialize_tracker(image, bbox)  # image, bounding box
        ###################


        ### Flag Variables & Progress Bar ###
        previous = bbox
        counter = 0
        pbar = tqdm(total=int(total_frames),
                    bar_format='Processing: {desc}{percentage:3.0f}%|{bar:10}')
        ###################


        ### Process Video & Write Frames ###
        while video_in.isOpened():

            ret, image = video_in.read()
            if not ret:
                break
            debug_image = copy.deepcopy(image)

            # Update Progress Bar
            pbar.update(1)

            # Updating Tracker
            ok, bbox = tracker.update(image)
            counter += 1

            if counter == 10:
                #  calculate Euclidean Distance
                #  between bbox 10 frames apart
                distance = calc_distance(previous, bbox)
                previous = bbox

            if ok:
                if counter < 10:
                    q.appendleft(bbox)
                    q.pop()
                else:
                    if distance > 50:
                        #  significant change in bbox location / all good
                        q.appendleft(bbox)
                        q.pop()
                        counter = 0
                    else:
                        #  bbox hasn't moved / stuck on non-volleyball object
                        #  since we know that volleyball woud always be in motion
                        pred = model.predict(image)
                        bbox = x_y_w_h(pred, self.model_name)
                        q.appendleft(bbox)
                        q.pop()
                        counter = 0

                        if bbox != (0, 0, 0, 0):
                            tracker = initialize_tracker(image, bbox)
                            previous = bbox
                        else:
                            q.appendleft(None)
                            q.pop()

            ### marker, color & trace ###
            for i in range(0, 8):
                if q[i] is not None:

                    if i == 0:  # current detection
                        if self.marker == 'box':
                            cv2.rectangle(debug_image, q[i], self.color, thickness=2)
                        elif self.marker == 'circle':
                            *center, r = get_circle(q[i])
                            cv2.circle(debug_image, center, r, self.color, 5)

                    elif (i != 0) and (self.no_trace is False):  # pass detections
                        if self.marker == 'box':
                            cv2.rectangle(debug_image, q[i], self.color, thickness=2)
                        elif self.marker == 'circle':
                            *center, r = get_circle(q[i])
                            try:
                                cv2.circle(debug_image, center, r-10, self.color, -1)
                            except:
                                cv2.circle(debug_image, center, r, self.color, -1)
            ###################

            video_out.write(debug_image)
        ###################
            if self.show:
                cv2.imshow('"p" - PAUSE, "Esc" - EXIT', debug_image)

            k = cv2.waitKey(1)
            if k == ord('p'):
                cv2.waitKey(-1)  # PAUSE
            if k == 27:  # ESC
                break

        video_in.release()
        video_out.release()
        cv2.destroyAllWindows()
        pbar.close()

        ###     End Time     ###
        t2 = datetime.now()
        dt = t2 - t1
        ###################
        print(f'Done - {dt.seconds/60:.2f} minutes')

test = VolleyBallTracker('./assets/back_view.mp4', './VideoOutput/new_output.mp4', 'yolov7', 0.4, False, 'circle', 'yellow', False)
test.ballTracker()
            