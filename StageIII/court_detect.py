import os
import io
import cv2
import base64
import argparse
import numpy as np
from tqdm import tqdm
from roboflow import Roboflow


class VolleyCourtDetector:
    def __init__(self,input_path,output_path,model):
        self.input_path = input_path
        self.output_path = output_path
        self.model = model
    def check_input(self):
        img_detect = VolleyCourtDetector(self.input_path,self.output_path,self.model)
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

        # Determine if input is an image or video based on file extension
        file_extension = os.path.splitext(self.input_path)[1]

        if file_extension in ['.jpg', '.png', '.jpeg']:
            # If it's an image, call process_image
            output_image_path = os.path.join(self.output_path, 'output_image.jpg')
            print(self.input_path)
            img_detect.process_image(self.input_path, output_image_path)
        elif file_extension in ['.mp4', '.avi']:
            # If it's a video, call process_video
            output_video_path = os.path.join(self.output_path)
            img_detect.process_video(self.input_path,output_video_path)
        else:
            print('Invalid file type. Please provide an image (jpg, png, jpeg) or a video (mp4, avi).')

    def process_image(self,input_path, output_path):
        img_detect = VolleyCourtDetector(self.input_path,self.output_path,self.model)
        output = img_detect.model.predict(input_path).json()['predictions'][0]
        segmentation_mask = output['segmentation_mask']
        image_width = output['image']['width']
        image_height = output['image']['height']
        decoded_mask = base64.b64decode(segmentation_mask)
        mask_array = np.frombuffer(decoded_mask, dtype=np.uint8)
        mask_image = cv2.imdecode(mask_array, cv2.IMREAD_GRAYSCALE)
        mask_image = cv2.resize(mask_image, (image_width, image_height))

        # Find and Draw Contours
        contours, _ = cv2.findContours(
            mask_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        largest_contour = max(contours, key=cv2.contourArea)
        epsilon = 0.01 * cv2.arcLength(largest_contour, True)
        trapezoid = cv2.approxPolyDP(largest_contour, epsilon, True)
        img = cv2.imread(input_path)
        cv2.drawContours(img, [trapezoid], 0, (0, 0, 0), 5)
        cv2.imwrite(output_path, img)


    def process_video(self,input_path, output_path):
        img_detect = VolleyCourtDetector(self.input_path,self.output_path,self.model)
        # Load the video file
        video_capture = cv2.VideoCapture(input_path)

        # Get the video properties
        frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = video_capture.get(cv2.CAP_PROP_FPS)
        total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

        # Define the codec for the output video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_video = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

        # Loop through each frame of the video
        for _ in tqdm(range(total_frames), desc="Processing video"):
            # Read a frame from the video
            ret, frame = video_capture.read()
            if not ret:
                break

            # Process
            cv2.imwrite('temp.jpg', frame)
            img_detect.process_image('temp.jpg', 'temp_processed.jpg')
            frame = cv2.imread('temp_processed.jpg')

            # Write the modified frame to the output video
            output_video.write(frame)

        # Release the video capture and output video
        video_capture.release()
        output_video.release()

        # Destroy any remaining windows
        cv2.destroyAllWindows()

        # Delete temporary files
        if os.path.exists('temp.jpg'):
            os.remove('temp.jpg')
        if os.path.exists('temp_processed.jpg'):
            os.remove('temp_processed.jpg')


rf = Roboflow(api_key="5lhXjuq3PEKqgKnLEZW6")
project = rf.workspace().project("court-segmented")
model = project.version(1).model
test = VolleyCourtDetector('C://Users//s6324//Desktop//volleyball_detect//StageIII//assets//event_detection.mp4','C://Users//s6324//Desktop//volleyball_detect//StageIII//output//output_court.mp4',model)
test.check_input()
test.process_image()    
test.process_video()
# Initialize the Roboflow model
# API Key, if doesn't work, refer -->
# https://github.com/shukkkur/VolleyVision/discussions/5#discussioncomment-7737081
