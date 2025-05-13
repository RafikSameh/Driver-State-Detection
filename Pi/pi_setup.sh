#!/bin/bash

# Create a systemd service file
cat > stream-video.service << EOF
[Unit]
Description=Stream Video to Azure VM
After=network.target

[Service]
User=$USER
WorkingDirectory=$HOME
ExecStart=/usr/bin/python3 $HOME/stream_video.py --vm-ip YOUR_AZURE_VM_IP --port 8554
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Install the service
sudo mv stream-video.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable stream-video
sudo systemctl start stream-video

echo "Raspberry Pi setup complete!"