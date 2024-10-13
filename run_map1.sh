#!/bin/bash

# Check if the folder name argument is provided
if [ $# -eq 0 ]; then
    echo "Please provide a folder name as an argument."
    exit 1
fi

# Store the folder name argument
folder_name=$1

# Check if the -source argument is provided
if [ "$2" == "-source" ]; then
    # Activate the virtual environment from the yolov8 folder
    source yolov8/bin/activate
fi

# Execute the Python scripts consecutively with the folder name argument
python3 recognize_humans.py "$folder_name"

python3 prepare_cutoff.py "$folder_name"
python3 warp_select.py "$folder_name"

python3 warp_video.py "$folder_name"

python3 measure.py "$folder_name"
#python3 warp_positions.py "$folder_name"
#python3 group_coordinates2.py "$folder_name"
#python3 paint_overlay5.py "$folder_name"

# Deactivate the virtual environment if it was activated
#if [ "$2" == "-source" ]; then
#    deactivate
#fi