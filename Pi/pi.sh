# Update and upgrade
sudo apt update && sudo apt upgrade -y

# Install required packages
sudo apt install -y python3-pip python3-opencv libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev

# Install Python libraries
pip3 install picamera opencv-python gstreamer-python