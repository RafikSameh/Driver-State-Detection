import time
import pprint
import os
import json
import cv2
import numpy as np
import requests
import io
from PIL import Image
import matplotlib.pyplot as plt

from attention_scorer import AttentionScorer as AttScorer
from custom_parser import get_args
from utils import load_camera_parameters

# Azure Face API configuration
FACE_KEY = '4XyoHZAbyKUq6dffHXoJnu2RAnOT5HUUsWT3yCpBIChM2YI7NhMDJQQJ99BEACYeBjFXJ3w3AAAFACOGd97H'
FACE_ENDPOINT = 'https://cameras.cognitiveservices.azure.com/'  # Replace with your Azure Face API endpoint
# Create a modified Eye Detector that uses Azure instead of local processing
class AzureEyeDetector:
    def __init__(self, show_processing=False):
        self.show_processing = show_processing

    def show_eye_keypoints(self, color_frame, face_landmarks, frame_size):
        """Display eye keypoints on the frame if show_processing is True"""
        if not self.show_processing or face_landmarks is None:
            return color_frame
        
        # Extract eye landmarks from Azure face landmarks
        if 'eyeLeftTop' in face_landmarks:
            # Draw landmarks for eyes
            for point_name in ['eyeLeftTop', 'eyeLeftBottom', 'eyeLeftInner', 'eyeLeftOuter',
                             'eyeRightTop', 'eyeRightBottom', 'eyeRightInner', 'eyeRightOuter']:
                if point_name in face_landmarks:
                    x = int(face_landmarks[point_name]['x'] * frame_size[0])
                    y = int(face_landmarks[point_name]['y'] * frame_size[1])
                    cv2.circle(color_frame, (x, y), 2, (0, 255, 0), -1)
        
        return color_frame
    
    def get_EAR(self, face_landmarks, frame_size):
        """Calculate Eye Aspect Ratio using Azure face landmarks"""
        if face_landmarks is None or 'eyeLeftTop' not in face_landmarks:
            return None
        
        # Extract eye landmark positions
        try:
            # Left eye
            left_eye_top = np.array([
                face_landmarks['eyeLeftTop']['x'] * frame_size[0],
                face_landmarks['eyeLeftTop']['y'] * frame_size[1]
            ])
            left_eye_bottom = np.array([
                face_landmarks['eyeLeftBottom']['x'] * frame_size[0],
                face_landmarks['eyeLeftBottom']['y'] * frame_size[1]
            ])
            left_eye_inner = np.array([
                face_landmarks['eyeLeftInner']['x'] * frame_size[0],
                face_landmarks['eyeLeftInner']['y'] * frame_size[1]
            ])
            left_eye_outer = np.array([
                face_landmarks['eyeLeftOuter']['x'] * frame_size[0],
                face_landmarks['eyeLeftOuter']['y'] * frame_size[1]
            ])
            
            # Right eye
            right_eye_top = np.array([
                face_landmarks['eyeRightTop']['x'] * frame_size[0],
                face_landmarks['eyeRightTop']['y'] * frame_size[1]
            ])
            right_eye_bottom = np.array([
                face_landmarks['eyeRightBottom']['x'] * frame_size[0],
                face_landmarks['eyeRightBottom']['y'] * frame_size[1]
            ])
            right_eye_inner = np.array([
                face_landmarks['eyeRightInner']['x'] * frame_size[0],
                face_landmarks['eyeRightInner']['y'] * frame_size[1]
            ])
            right_eye_outer = np.array([
                face_landmarks['eyeRightOuter']['x'] * frame_size[0],
                face_landmarks['eyeRightOuter']['y'] * frame_size[1]
            ])
            
            # Calculate vertical distances
            left_eye_height = np.linalg.norm(left_eye_top - left_eye_bottom)
            right_eye_height = np.linalg.norm(right_eye_top - right_eye_bottom)
            
            # Calculate horizontal distances
            left_eye_width = np.linalg.norm(left_eye_inner - left_eye_outer)
            right_eye_width = np.linalg.norm(right_eye_inner - right_eye_outer)
            
            # Calculate EAR (Eye Aspect Ratio)
            left_ear = left_eye_height / left_eye_width
            right_ear = right_eye_height / right_eye_width
            
            # Average EAR of both eyes
            ear = (left_ear + right_ear) / 2.0
            return ear
        except KeyError:
            # Some landmarks might be missing
            return None
    
    def get_Gaze_Score(self, face_attributes, frame_size):
        """Calculate gaze score based on Azure face API head pose data"""
        if face_attributes is None or 'headPose' not in face_attributes:
            return None
            
        # Azure provides head pose as a proxy for gaze direction
        pitch = face_attributes['headPose']['pitch']
        yaw = face_attributes['headPose']['yaw']
        
        # Simplified gaze scoring based on head pose
        # Assuming looking directly at camera is ideal (pitch and yaw near 0)
        gaze_score = 1.0 - min(1.0, (abs(pitch) + abs(yaw)) / 90.0)
        
        return gaze_score


# Create a modified Head Pose Estimator that uses Azure instead of local processing
class AzureHeadPoseEstimator:
    def __init__(self, show_axis=False):
        self.show_axis = show_axis
        
    def get_pose(self, frame, face_attributes, face_landmarks, frame_size):
        """Get head pose from Azure Face API results"""
        if face_attributes is None or 'headPose' not in face_attributes:
            return frame, None, None, None
            
        head_pose = face_attributes['headPose']
        
        # Extract Euler angles
        roll = np.array([head_pose['roll']])
        pitch = np.array([head_pose['pitch']])
        yaw = np.array([head_pose['yaw']])
        
        # If requested, visualize head pose axis on the frame
        if self.show_axis:
            frame = self._draw_axis(frame, roll, pitch, yaw, face_landmarks, frame_size)
            
        return frame, roll, pitch, yaw
    
    def _draw_axis(self, frame, roll, pitch, yaw, face_landmarks, frame_size):
        """Draw 3D axis on the face to visualize head pose"""
        # Simplified axis drawing using Azure Face API landmarks
        if face_landmarks is None or 'noseTip' not in face_landmarks:
            return frame
            
        # Get face center point (nose tip)
        nose_x = int(face_landmarks['noseTip']['x'] * frame_size[0])
        nose_y = int(face_landmarks['noseTip']['y'] * frame_size[1])
        
        # Calculate face size for scaling the axes
        face_width = 0
        if 'eyeLeftOuter' in face_landmarks and 'eyeRightOuter' in face_landmarks:
            left_x = int(face_landmarks['eyeLeftOuter']['x'] * frame_size[0])
            right_x = int(face_landmarks['eyeRightOuter']['x'] * frame_size[0])
            face_width = abs(right_x - left_x)
        
        if face_width > 0:
            # Draw simplified axes
            length = face_width / 2
            
            # Draw axes lines
            # X-axis (roll) - red
            roll_rad = np.deg2rad(roll)
            x_end = int(nose_x + length * np.cos(roll_rad.item()))
            y_end = int(nose_y + length * np.sin(roll_rad.item()))
            cv2.line(frame, (nose_x, nose_y), (x_end, y_end), (0, 0, 255), 2)
            
            # Y-axis (pitch) - green
            pitch_rad = np.deg2rad(pitch)
            y_end = int(nose_y - length * np.sin(pitch_rad.item()))
            z_end = int(nose_x + length * np.cos(pitch_rad.item()))
            cv2.line(frame, (nose_x, nose_y), (z_end, y_end), (0, 255, 0), 2)
            
            # Z-axis (yaw) - blue
            yaw_rad = np.deg2rad(yaw)
            z_end = int(nose_x + length * np.sin(yaw_rad.item()))
            x_end = int(nose_y + length * np.cos(yaw_rad.item()))
            cv2.line(frame, (nose_x, nose_y), (z_end, x_end), (255, 0, 0), 2)
        
        return frame


def main():
    args = get_args()
    
    if not FACE_KEY or not FACE_ENDPOINT:
        print("Please set AZURE_FACE_KEY and AZURE_FACE_ENDPOINT environment variables")
        return

    if args.verbose:
        print("Arguments and Parameters used:\n")
        pprint.pp(vars(args), indent=4)
        print("\n")
    
    # Initialize Azure-based detectors
    Eye_det = AzureEyeDetector(show_processing=args.show_eye_proc)
    Head_pose = AzureHeadPoseEstimator(show_axis=args.show_axis)
    
    # Get the current time in seconds
    t_now = time.perf_counter()
    
    # Instantiate the attention scorer object with thresholds
    Scorer = AttScorer(
        t_now=t_now,
        ear_thresh=args.ear_thresh,
        gaze_time_thresh=args.gaze_time_thresh,
        roll_thresh=args.roll_thresh,
        pitch_thresh=args.pitch_thresh,
        yaw_thresh=args.yaw_thresh,
        ear_time_thresh=args.ear_time_thresh,
        gaze_thresh=args.gaze_thresh,
        pose_time_thresh=args.pose_time_thresh,
        verbose=args.verbose,
    )
    
    # Capture video from the specified camera
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    
    # Initialize variables for distraction logging
    distraction_start_time = None
    total_distraction_time = 0
    log_file_path = "distraction_log.txt"
    state_message = ""
    
    # Open the log file in append mode
    with open(log_file_path, "a") as log_file:
        log_file.write("\nDistraction Log\n")
        log_file.write(f"{time.asctime(time.localtime(time.time()))}\n")
        log_file.write("Distraction events:\n")
        log_file.write("================\n")
    
    # Set up the Azure Face API request
    headers = {
        'Ocp-Apim-Subscription-Key': FACE_KEY,
        'Content-Type': 'application/octet-stream'
    }
    
    # Make sure endpoint doesn't have trailing slash for direct API calls
    face_endpoint = FACE_ENDPOINT
    if face_endpoint.endswith('/'):
        face_endpoint = face_endpoint[:-1]
    
    # Azure Face API parameters
    face_api_params = {
        'returnFaceLandmarks': 'true',
        'returnFaceAttributes': 'headPose,glasses,occlusion',
        'detectionModel': 'detection_01'
    }
    
    # For frame skipping to reduce API calls
    frame_count = 0
    skip_frames = 2  # Process every 3rd frame
    
    try:
        while True:
            # Get current time in seconds
            t_now = time.perf_counter()
            
            # Read a frame from the webcam
            ret, frame = cap.read()
            
            if not ret:
                print("Can't receive frame from camera/stream end")
                break
            
            # If the frame comes from webcam, flip it so it looks like a mirror
            if args.camera == 0:
                frame = cv2.flip(frame, 2)
            
            # Get frame dimensions
            frame_size = frame.shape[1], frame.shape[0]
            
            # Skip frames to reduce API calls
            frame_count += 1
            if frame_count % (skip_frames + 1) == 0:
            
                # Create a copy of the frame for processing
                process_frame = frame.copy()
                
                # Reduce image size for faster API calls
                process_frame = cv2.resize(process_frame, (320, 240))
                
                # Convert the frame to a format suitable for Azure Face API
                _, img_encoded = cv2.imencode('.jpg', process_frame)
                img_bytes = img_encoded.tobytes()
                
                try:
                    # Direct REST API call to Azure Face API
                    response = requests.post(
                        f"{face_endpoint}/face/v1.0/detect",
                        headers=headers,
                        params=face_api_params,
                        data=img_bytes
                    )
                    
                    if response.status_code == 200:
                        detected_faces = response.json()
                        
                        if detected_faces:
                            # Use the first detected face (assuming it's the driver)
                            face = detected_faces[0]
                            face_landmarks = face.get('faceLandmarks', {})
                            face_attributes = face.get('faceAttributes', {})
                            
                            # Show eye keypoints if requested
                            if args.show_eye_proc and face_landmarks:
                                Eye_det.show_eye_keypoints(frame, face_landmarks, frame_size)
                            
                            # Calculate EAR (Eye Aspect Ratio)
                            ear = Eye_det.get_EAR(face_landmarks, frame_size)
                            
                            # Calculate Gaze Score
                            gaze = Eye_det.get_Gaze_Score(face_attributes, frame_size)
                            
                            # Calculate head pose
                            frame_det, roll, pitch, yaw = Head_pose.get_pose(
                                frame, face_attributes, face_landmarks, frame_size
                            )
                            
                            # Evaluate attention scores
                            asleep, looking_away, distracted = Scorer.eval_scores(
                                t_now=t_now,
                                ear_score=ear,
                                gaze_score=gaze,
                                head_roll=roll,
                                head_pitch=pitch,
                                head_yaw=yaw,
                            )
                            
                            # If the head pose estimation is successful, show the results
                            if frame_det is not None:
                                frame = frame_det
                            
                            # Determine driver's state
                            if looking_away or distracted or asleep:
                                if asleep:
                                    state_message = "ASLEEP"
                                elif distracted:
                                    state_message = "DISTRACTED"
                                elif looking_away:
                                    state_message = "LOOKING AWAY"
                                    
                                cv2.putText(
                                    frame,
                                    f"{state_message}!",
                                    (10, 320),
                                    cv2.FONT_HERSHEY_PLAIN,
                                    1,
                                    (0, 0, 255),
                                    1,
                                    cv2.LINE_AA,
                                )
                                
                                # Start tracking distraction time if not already started
                                if distraction_start_time is None:
                                    distraction_start_time = time.time()
                            else:
                                # End tracking distraction time if previously distracted
                                if distraction_start_time is not None:
                                    distraction_end_time = time.time()
                                    distraction_duration = distraction_end_time - distraction_start_time
                                    total_distraction_time += distraction_duration
                                    
                                    # Log the distraction event
                                    with open(log_file_path, "a") as log_file:
                                        log_file.write(
                                            f"{state_message} from {time.strftime('%H:%M:%S', time.gmtime(distraction_start_time))} "
                                            f"to {time.strftime('%H:%M:%S', time.gmtime(distraction_end_time))} "
                                            f"Duration: {round(distraction_duration, 2)} seconds\n"
                                        )
                                    
                                    # Reset distraction start time
                                    distraction_start_time = None
                    else:
                        print(f"Azure Face API error: {response.status_code} - {response.text}")
                        
                except Exception as e:
                    print(f"Error calling Azure Face API: {str(e)}")
            
            # Show the frame on screen
            cv2.imshow("Press 'q' to terminate", frame)
            
            # If the key "q" is pressed on the keyboard, terminate the program
            if cv2.waitKey(20) & 0xFF == ord("q"):
                break
                
    except KeyboardInterrupt:
        print("Program terminated by user")
    
    cap.release()
    cv2.destroyAllWindows()
    
    # After the loop, log the total distraction time
    with open(log_file_path, "a") as log_file:
        log_file.write("================\n")
        log_file.write("\nTotal Distraction Time: {:.2f} seconds\n".format(total_distraction_time))
    
    return


if __name__ == "__main__":
    main()
