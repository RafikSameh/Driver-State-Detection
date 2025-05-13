import cv2
import time
import os
import numpy as np
import datetime
import argparse

# Import your existing processing modules
from attention_scorer import AttentionScorer as AttScorer
from eye_detector import EyeDetector as EyeDet
from pose_estimation import HeadPoseEstimator as HeadPoseEst
from utils import get_landmarks, load_camera_parameters

import mediapipe as mp

def parse_args():
    parser = argparse.ArgumentParser(description="Receive video stream from Raspberry Pi")
    parser.add_argument("--port", type=int, default=8554, help="Port for receiving stream")
    parser.add_argument("--show-display", action="store_true", help="Show video display")
    parser.add_argument("--log-dir", type=str, default="driver_logs", help="Directory for logs")
    return parser.parse_args()

def setup_logging(log_dir):
    """Set up logging for a driver monitoring session"""
    # Create log directory if it doesn't exist
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Create a log file with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    session_id = f"session_{timestamp}"
    log_file_path = f"{log_dir}/driver_session_{session_id}.log"
    
    # Initialize the log file with header
    with open(log_file_path, "w") as log_file:
        log_file.write(f"Driver Behavior Analysis Log\n")
        log_file.write(f"Session ID: {session_id}\n")
        log_file.write(f"Start Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_file.write("=====================================\n\n")
        log_file.write("Distraction Events:\n")
    
    return log_file_path, session_id

def main():
    args = parse_args()
    port = args.port
    show_display = args.show_display
    log_dir = args.log_dir
    
    # Set up logging
    log_file_path, session_id = setup_logging(log_dir)
    print(f"Logs will be saved to: {log_file_path}")
    
    # Initialize metrics
    total_distraction_time = 0
    distraction_events = []
    distraction_start_time = None
    current_state = "ATTENTIVE"
    
    # Set up MediaPipe face mesh detector
    Detector = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        refine_landmarks=True,
    )
    
    # Initialize detection objects
    Eye_det = EyeDet(show_processing=False)
    Head_pose = HeadPoseEst(show_axis=False)
    
    # Get current time
    t_now = time.perf_counter()
    
    # Initialize attention scorer
    Scorer = AttScorer(
        t_now=t_now,
        ear_thresh=0.15,        # Adjust these thresholds as needed
        gaze_time_thresh=2.0,
        roll_thresh=20,
        pitch_thresh=20,
        yaw_thresh=20,
        ear_time_thresh=2.0,
        gaze_thresh=0.2,
        pose_time_thresh=2.0,
        verbose=False,
    )
    
    # Set up video capture from the RTSP stream
    stream_url = f"udpsrc port={port} ! application/x-rtp,media=video,encoding-name=H264 ! rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! appsink"
    cap = cv2.VideoCapture(stream_url, cv2.CAP_GSTREAMER)
    
    if not cap.isOpened():
        print(f"Failed to open stream on port {port}")
        return
    
    print(f"Successfully opened stream on port {port}")
    
    try:
        while True:
            # Get current time
            t_now = time.perf_counter()
            
            # Read frame from stream
            ret, frame = cap.read()
            
            if not ret:
                print("Failed to receive frame. Reconnecting...")
                # Try to reconnect
                cap.release()
                time.sleep(2)
                cap = cv2.VideoCapture(stream_url, cv2.CAP_GSTREAMER)
                continue
            
            # Convert to grayscale for processing
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame_size = frame.shape[1], frame.shape[0]
            
            # Create 3D matrix from grayscale
            gray_3d = np.expand_dims(gray, axis=2)
            gray_3d = np.concatenate([gray_3d, gray_3d, gray_3d], axis=2)
            
            # Process with MediaPipe
            lms = Detector.process(gray_3d).multi_face_landmarks
            
            if lms:
                # Extract landmarks
                landmarks = get_landmarks(lms)
                
                # Show eye keypoints if display is enabled
                if show_display:
                    Eye_det.show_eye_keypoints(frame, landmarks, frame_size)
                
                # Compute EAR score
                ear = Eye_det.get_EAR(gray_3d, landmarks)
                
                # Compute Gaze Score
                gaze = Eye_det.get_Gaze_Score(gray_3d, landmarks, frame_size)
                
                # Compute head pose
                frame_det, roll, pitch, yaw = Head_pose.get_pose(frame, landmarks, frame_size)
                
                # Evaluate attention scores
                asleep, looking_away, distracted = Scorer.eval_scores(
                    t_now=t_now,
                    ear_score=ear,
                    gaze_score=gaze,
                    head_roll=roll,
                    head_pitch=pitch,
                    head_yaw=yaw,
                )
                
                # If head pose estimation successful, use the processed frame
                if frame_det is not None and show_display:
                    frame = frame_det
                
                # Handle distraction states
                if looking_away or distracted or asleep:
                    if asleep:
                        state_message = "ASLEEP"
                    elif distracted:
                        state_message = "DISTRACTED"
                    elif looking_away:
                        state_message = "LOOKING AWAY"
                    
                    # Add visual indicator if display is enabled
                    if show_display:
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
                    
                    # Start tracking distraction if not already tracking
                    if distraction_start_time is None:
                        distraction_start_time = time.time()
                        current_state = state_message
                else:
                    # If returning to attentive state, log the distraction event
                    if distraction_start_time is not None:
                        distraction_end_time = time.time()
                        distraction_duration = distraction_end_time - distraction_start_time
                        total_distraction_time += distraction_duration
                        
                        # Log the distraction event
                        with open(log_file_path, "a") as log_file:
                            log_file.write(
                                f"{current_state} from {time.strftime('%H:%M:%S', time.localtime(distraction_start_time))} "
                                f"to {time.strftime('%H:%M:%S', time.localtime(distraction_end_time))} "
                                f"Duration: {round(distraction_duration, 2)} seconds\n"
                            )
                        
                        # Add to events list
                        distraction_events.append({
                            "state": current_state,
                            "start_time": time.strftime('%H:%M:%S', time.localtime(distraction_start_time)),
                            "end_time": time.strftime('%H:%M:%S', time.localtime(distraction_end_time)),
                            "duration": round(distraction_duration, 2)
                        })
                        
                        # Reset distraction start time
                        distraction_start_time = None
            
            # Show frame if display is enabled
            if show_display:
                cv2.imshow("Driver Monitoring", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    
    except KeyboardInterrupt:
        print("Program terminated by user")
    
    finally:
        # Close resources
        cap.release()
        if show_display:
            cv2.destroyAllWindows()
        
        # Write summary to log file
        with open(log_file_path, "a") as log_file:
            log_file.write("\n=====================================\n")
            log_file.write("Session Summary:\n")
            log_file.write(f"End Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            log_file.write(f"Total Distraction Time: {round(total_distraction_time, 2)} seconds\n")
            log_file.write(f"Total Distraction Events: {len(distraction_events)}\n")
            
            # Add more insights
            if distraction_events:
                max_event = max(distraction_events, key=lambda x: x["duration"])
                log_file.write(f"Longest Distraction: {max_event['state']} for {max_event['duration']} seconds\n")
                
                # Count by type
                by_type = {}
                for event in distraction_events:
                    by_type[event["state"]] = by_type.get(event["state"], 0) + 1
                
                log_file.write("\nDistraction by Type:\n")
                for state, count in by_type.items():
                    log_file.write(f"- {state}: {count} events\n")
        
        print(f"Session complete. Log saved to {log_file_path}")

if __name__ == "__main__":
    main()