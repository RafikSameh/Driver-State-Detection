#!/bin/bash

# Update system
sudo apt update && sudo apt upgrade -y

# Install dependencies
sudo apt install -y python3-pip python3-dev cmake build-essential libgtk2.0-dev \
    libavcodec-dev libavformat-dev libswscale-dev python3-opencv \
    libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad gstreamer1.0-plugins-ugly gstreamer1.0-tools

# Install NVIDIA drivers if VM has GPU
if lspci | grep -i nvidia > /dev/null; then
    sudo apt install -y nvidia-driver-515 nvidia-cuda-toolkit
fi

# Create project directory
mkdir -p ~/driver-monitor
cd ~/driver-monitor

# Install Python packages
pip3 install numpy opencv-python mediapipe

# Create systemd service for auto-start
cat > driver-monitor.service << EOF
[Unit]
Description=Driver Behavior Monitoring Service
After=network.target

[Service]
User=$USER
WorkingDirectory=$HOME/driver-monitor
ExecStart=/usr/bin/python3 $HOME/driver-monitor/receive_stream.py --port 8554
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Install the service
sudo mv driver-monitor.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable driver-monitor
sudo systemctl start driver-monitor

# Open port in firewall
sudo ufw allow 8554/udp

echo "VM setup complete!"