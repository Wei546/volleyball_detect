from ultralytics import YOLO
import mediapipe as mp
import cv2
import numpy as np

# Initialize MediaPipe Pose and drawing utilities
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
# Initialize YOLO model
model = YOLO('yolo11n.pt')
model = YOLO('yolo11n-pose.pt')
results = model.predict(source="ballGame.mp4", save=True)