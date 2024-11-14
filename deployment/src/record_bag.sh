#!/bin/bash

# Create a new tmux session
session_name="record_bag_$(date +%s)"
tmux new-session -d -s $session_name

# Split the window into three panes
tmux selectp -t 0    # select the first (0) pane
tmux splitw -v -p 50 # split it into two halves
tmux selectp -t 0    # go back to the first pane

# Run the roslaunch command in the first pane
tmux select-pane -t 0
tmux send-keys "python video_stream_publisher.py" Enter

# Change the directory to ../topomaps/bags and run the rosbag record command in the third pane
tmux select-pane -t 1
tmux send-keys "cd ../topomaps/bags" Enter
tmux send-keys "rosbag record /robot/front_camera/image_raw -o $1" # change topic if necessary

# Attach to the tmux session
tmux -2 attach-session -t $session_name