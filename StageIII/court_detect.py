import os
import cv2
import base64
import numpy as np
from tqdm import tqdm
from roboflow import Roboflow


class CourtProcessor:
    def __init__(self, api_key, project_name="court-segmented", version=1):
        # Initialize the Roboflow model
        self.rf = Roboflow(api_key=api_key)
        self.project = self.rf.workspace().project(project_name)
        self.model = self.project.version(version).model

    def process_image(self, input_path, output_path):
        output = self.model.predict(input_path).json()['predictions'][0]
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

    def process_video(self, input_path, output_path):
        # Load the video file
        video_capture = cv2.VideoCapture(input_path)

        # Get the video properties
        frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = video_capture.get(cv2.CAP_PROP_FPS)
        total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

        # Define the codec for the output video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_video = cv2.VideoWriter(
            output_path, fourcc, fps, (frame_width, frame_height))

        # Loop through each frame of the video
        for _ in tqdm(range(total_frames), desc="Processing video"):
            # Read a frame from the video
            ret, frame = video_capture.read()
            if not ret:
                break

            # Process
            temp_input = 'temp.jpg'
            temp_output = 'temp_processed.jpg'
            cv2.imwrite(temp_input, frame)
            self.process_image(temp_input, temp_output)
            frame = cv2.imread(temp_output)

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


# Function interface to call the class
def process_court(api_key, input_path, output_path="./Output"):
    # Initialize the processor
    processor = CourtProcessor(api_key=api_key)

    # Ensure output directory exists
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Determine if input is an image or video based on file extension
    file_extension = os.path.splitext(input_path)[1]

    if file_extension in ['.jpg', '.png', '.jpeg']:
        # Process image
        output_image_path = os.path.join(output_path, 'output_image.jpg')
        print(f"Processing image: {input_path}")
        processor.process_image(input_path, output_image_path)
        print(f"Image saved to: {output_image_path}")
    elif file_extension in ['.mp4', '.avi']:
        # Process video
        output_video_path = os.path.join(output_path, 'output_video.mp4')
        print(f"Processing video: {input_path}")
        processor.process_video(input_path, output_video_path)
        print(f"Video saved to: {output_video_path}")
    else:
        print('Invalid file type. Please provide an image (jpg, png, jpeg) or a video (mp4, avi).')


if __name__ == "__main__":
    API_KEY = "5lhXjuq3PEKqgKnLEZW6"
    INPUT_PATH = "./assets/event_detection.mp4"
    OUTPUT_PATH = "./StageIII/output"

    processor = CourtProcessor(api_key=API_KEY)

    if INPUT_PATH.endswith(('.jpg', '.png', '.jpeg')):
        processor.process_image(INPUT_PATH, os.path.join(OUTPUT_PATH, "output_image.jpg"))
    elif INPUT_PATH.endswith(('.mp4', '.avi')):
        processor.process_video(INPUT_PATH, os.path.join(OUTPUT_PATH, "output_video.mp4"))
