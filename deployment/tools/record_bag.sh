#!/bin/bash

# Create a new tmux session
session_name="record_bag_$(date +%s)"
tmux new-session -d -s $session_name

# Split the window into three panes
tmux selectp -t 0    # select the first (0) pane
tmux splitw -v -p 50 # split it into two halves
tmux selectp -t 0    # go back to the first pane
tmux splitw -h -p 50 # split it into two halves

# Run the roslaunch command in the first pane
tmux select-pane -t 0
tmux send-keys "2" Enter
tmux send-keys "python ../src/video_stream_publisher.py" Enter

tmux select-pane -t 1
tmux send-keys "2" Enter
tmux send-keys "python pub_robot_odom.py" Enter
n  
# Change the directory to ../topomaps/bags and run the rosbag record command in the third pane
tmux select-pane -t 2
tmux send-keys "2" Enter
tmux send-keys "cd ../../go2_bags" Enter
tmux send-keys "rosbag record --lz4 /robot_odom /robot/front_camera/image_raw -O $1" # change topic if necessary
# Attach to the tmux session
tmux -2 attach-session -t $session_name