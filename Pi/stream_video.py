import cv2
import time
import socket
import picamera
import picamera.array
import numpy as np
import subprocess
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Stream video from Raspberry Pi to Azure VM")
    parser.add_argument("--vm-ip", type=str, required=True, help="IP address of Azure VM")
    parser.add_argument("--port", type=int, default=8554, help="Port for streaming")
    parser.add_argument("--resolution", type=str, default="640x480", help="Video resolution")
    parser.add_argument("--fps", type=int, default=15, help="Frames per second")
    return parser.parse_args()

def stream_with_gstreamer(vm_ip, port, resolution, fps):
    """Stream using GStreamer pipeline"""
    width, height = resolution.split('x')
    
    # Create GStreamer pipeline
    gst_command = [
        'raspivid',
        '-t', '0',                # No timeout
        '-w', width,              # Width
        '-h', height,             # Height
        '-fps', str(fps),         # Framerate
        '-b', '2000000',          # Bitrate
        '-o', '-',                # Output to stdout
        '|',
        'gst-launch-1.0',
        'fdsrc',
        '!',
        'h264parse',
        '!',
        f'rtph264pay config-interval=1 pt=96 ! udpsink host={vm_ip} port={port}'
    ]
    
    # Execute the pipeline
    command = ' '.join(gst_command)
    print(f"Starting stream with command: {command}")
    process = subprocess.Popen(command, shell=True)
    
    try:
        # Keep the stream running until interrupted
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        process.terminate()
        print("Stream terminated")

def main():
    args = parse_args()
    vm_ip = args.vm_ip
    port = args.port
    resolution = args.resolution
    fps = args.fps
    
    print(f"Starting video stream to {vm_ip}:{port}")
    print(f"Resolution: {resolution}, FPS: {fps}")
    
    # Check if VM is reachable
    try:
        socket.create_connection((vm_ip, port), timeout=5)
        print(f"Successfully connected to {vm_ip}:{port}")
    except socket.error as e:
        print(f"Error connecting to {vm_ip}:{port}: {e}")
        print("Check if the VM is running and the port is open.")
        return
    
    # Start streaming
    stream_with_gstreamer(vm_ip, port, resolution, fps)

if __name__ == "__main__":
    main()