#!/bin/bash

# Create a new tmux session
session_name="pd_control$(date +%s)"
tmux new-session -d -s $session_name

# Split the window into four panes
tmux split-window -v


# Run the pd_controller.py script in the third pane
tmux select-pane -t 0
tmux send-keys "2" Enter
# tmux send-keys "conda activate vint_deployment" Enter
tmux send-keys "python pd_controller.py" 

# Run the go2_control.py.py script in the fourth pane
tmux select-pane -t 1
tmux send-keys "2" Enter
# tmux send-keys "conda activate vint_deployment" Enter
tmux send-keys "python go2_control.py" Enter
tmux selectp -t 0    # go back to the first pane

# Attach to the tmux session
tmux -2 attach-session -t $session_name