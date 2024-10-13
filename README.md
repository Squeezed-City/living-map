This code is part of STARTS AIR project 2024.

Use it to create living maps of public places, as seen on https://squeezed.city/ 

The process is straightforward but has many steps. It is advisable to start with a short video of a few seconds and see how that goes.

0. clone the repository, create a virtual environment and run 'pip install -r requirements.txt'.
1. enter source (your venv path)/bin/activate to work in the virtual environment
2. download a video stream (using e.g. yt-dlp command line tool), convert to an appropriate fps (60 is too large, 10 is too low) and place it into videos/yourvideo/video_original.mp4 Place or copy a dummy config.txt next to it.
3. run python3 recognize_humans.py "$folder_name" with the path to your folder instead of $folder_name. This process will recognize all humans in each frame, and save the coordinates of their feet to positions_original.txt
4. run python3 prepare_cutoff.py "$folder_name" to select the cutoff point. This point should be a bit lower than the horizon. The selection will be saved to config.txt as crop_height_percentage setting.
5. run python3 warp_select.py "$folder_name" to select four points for inverse perspective homography transform. To get a top-down view from a side view, we need to select four points on the ground that would constitute a rectangle when seen from above. Select in the clockwise order: top left, top right, bottom right, bottom left. The coordinates and the warped coordinates will be saved to config.txt.
6. run python3 warp_video.py "$folder_name" – for the video warp some additional config settings are needed -
dst_width = 480
dst_height = 480
x_offset = 380
y_offset = -50
dst height and width set the size of the projection, while x and y offset let you place the projection as you like, starting from bottom left. The warp function is a custom one since cv2.findHomography crops the view to the four selected points and we want to maintain the full view.
8. run python3 measure.py "$folder_name" – this generates a median_warped.png file, that is the median of 60 frames from the video, which for long enough videos means that only the space can be seen, without any human bodies (or only with ones who remained very static throughout the video). The warped median frame is then shown to you to click two times in order to select a distance of 1 meter (an approximation of typical personal space). Consult with google maps to see what constitutes a meter in your particular place. This setting is saved as proximity_threshold in the config.txt file. Adjust if you think it is too large or too small.
9. run python3 warp_positions.py "$folder_name" – the positions in the positions_original.txt are warped using warped_ settings in config.txt and exported to positions_warped.txt
10. run python3 group_coordinates2.py "$folder_name" - the warped positions are grouped using the DBSCAN spatio temporal clustering algorithm and exported to positions_warped_grouped.txt. After all, we do not want to paint kinesphere infringements between the members of the same group, as they are often close together and maintain a 'group space' and not personal space.
11. run python3 paint_overlay5.py "$folder_name". This creates a living heatmap of kinesphere infringements - from blue to purple to yellow they go in a growing intensity of overlaps in a particular place. Over time, governed by 'fade_rate' variable in the script file, they go back in intensity and fade away if no new overlaps occur. A video file annotated_video.mp4 is created and is already a strong outcome but can be adjusted to a faster and richer version:
12. run python3 aura.py "$folder_name" to create a 10x faster smoother version. The script combines the next 60 frames into one, saves it as a frame, and slides the 60-frame window one frame further. This creates a flowing, softly changing effect. annotated_video_aura_final.mp4 is exported.
13. run python3 paint_dots_ten_traces.py "$folder_name/videoname" (this script expects video file name as input too, not just the folder). This will generate dots where people stood, and leave traces where they were in the previous frames. The traces show their movement. Alternatively run paint_dots_ten.py - this will generate the dots where people stood, but not their traces. wormy_traces.mp4 is exported.
