import cv2
import gi
from gi.repository import Gst

Gst.init(None)
pipeline = Gst.parse_launch("v4l2src device=/dev/video0 ! video/x-raw,width=640,height=480 ! videoconvert ! appsink")
cam = cv2.VideoCapture()
cam.open(pipeline)
_, frame = cam.read()
cv2.imwrite('photo.jpg',frame)
cam.release()
print(frame)