Run the streamer on the PI:

gst-launch-v1.0 -v v4l2src device=/dev/video0 ! image/jpeg,width=1280,height=720,framerate=30/1 ! rtpjpegpay ! udpsink host=<IP_ADDRESS> port=5000