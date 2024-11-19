import os
import cv2
import argparse
from ultralytics import YOLO

class VolleyPlayerDetector:
    def __init__(self, weight_path,input_path,output_path,show_conf,show_labels,conf,max_det,classes,line_width,font_size):
        self.weight_path = weight_path
        self.input_path = input_path
        self.output_path = output_path
        self.show_conf = show_conf
        self.show_labels = show_labels
        self.conf = conf
        self.max_det = max_det
        self.classes = classes
        self.line_width = line_width
        self.font_size = font_size
    def playerDetect(self):
        # Load the YOLOv8 model
        model = YOLO(self.weight_path)

        # Check if the input is an image or video
        is_image = self.input_path.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif'))
        is_video = self.input_path.endswith(('.mp4', '.avi', '.mkv', '.mov'))

        if is_image:
            # Load and preprocess the image
            img = cv2.imread(self.input_path)

            # Perform prediction
            results = model(img,
                            conf=self.conf,
                            max_det=self.max_det,
                            classes=self.classes,
                            verbose=False)

            # Create the output directory if it doesn't exist
            try:
                os.makedirs(os.path.split(self.output_path)[0], exist_ok=True)
            except:
                pass

            # Annotate the image with bounding boxes
            annotated = results[0].plot(conf=self.show_conf,
                                        labels=self.show_labels,
                                        line_width=self.line_width,
                                        font_size=self.font_size)

            # Save the annotated image
            cv2.imwrite(self.output_path, annotated)

        elif is_video:
            # Open the video file
            cap = cv2.VideoCapture(self.input_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_size = (
                int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            )

            # Create the output directory if it doesn't exist
            os.makedirs(os.path.split(self.output_path)[0], exist_ok=True)

            # Define the codec and create VideoWriter object
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(self.output_path, fourcc, fps, frame_size)

            # Loop through the video frames
            while cap.isOpened():
                # Read a frame from the video
                success, frame = cap.read()

                if success:
                    # Run YOLOv8 inference on the frame
                    results = model(frame,
                                    conf=self.conf,
                                    max_det=self.max_det,
                                    classes=self.classes,
                                    verbose=False)

                    # Annotate the frame with bounding boxes
                    annotated_frame = results[0].plot(conf=self.show_conf,
                                                    labels=self.show_labels,
                                                    line_width=self.line_width,
                                                    font_size=self.font_size)

                    # Write the annotated frame to the output video
                    out.write(annotated_frame)

                else:
                    # Break the loop if the end of the video is reached
                    break

            # Release the video capture object, close the display window, and release the output video writer
            cap.release()
            cv2.destroyAllWindows()
            out.release()

        else:
            raise ValueError(
                "Invalid input format. Please provide either an image or a video file.")

test = VolleyPlayerDetector(weight_path='./actions/yV8_medium/weights/best.pt',input_path='./assets/ballGame.mp4',output_path='./Output/TrackVideo.mp4',show_conf=True,show_labels=False,conf=0.25,max_det=100,classes=None,line_width=3,font_size=3)
test.playerDetect()