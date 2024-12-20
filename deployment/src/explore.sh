#!/bin/bash

# Create a new tmux session
session_name="go2_$(date +%s)"
tmux new-session -d -s $session_name

# Split the window into four panes
tmux selectp -t 0    # select the first (0) pane
tmux splitw -h -p 50 # split it into two halves
tmux selectp -t 0    # select the first (0) pane
tmux splitw -v -p 50 # split it into two halves

tmux selectp -t 2    # select the new, second (2) pane
tmux splitw -v -p 50 # split it into two halves
tmux selectp -t 0    # go back to the first pane

# Run the videopublisher command in the first pane
tmux select-pane -t 0
tmux send-keys "2" Enter
tmux send-keys "python video_stream_publisher.py" Enter

# Run the navigate.py script with command line args in the second pane
tmux select-pane -t 1
tmux send-keys "2" Enter
# tmux send-keys "conda activate vint_deployment" Enter
tmux send-keys "python explore.py $@" Enter


# Run the pd_controller.py script in the fourth pane
tmux select-pane -t 2
tmux send-keys "2" Enter
# tmux send-keys "conda activate vint_deployment" Enter
tmux send-keys "python pd_controller.py" Enter


# Run the go2_control.py script in the third pane
tmux select-pane -t 3
tmux send-keys "2" Enter
# tmux send-keys "conda activate vint_deployment" Enter
tmux send-keys "python go2_control.py" Enter


# Attach to t  

tmux session
tmux -2 attach-session -t $session_name

