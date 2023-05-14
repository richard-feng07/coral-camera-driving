import cv2
import gi
gi.require_version('Gst','1.0')
from gi.repository import Gst
print("imported")
Gst.init(None)
print("pipeline")
pipeline = Gst.parse_launch("v4l2src device=/dev/video0 ! video/x-raw,width=640,height=480 ! videoconvert ! appsink")
print("cam starting")
cam = cv2.VideoCapture(0)
# cam.open(pipeline)
_, frame = cam.read()
cv2.imwrite('photo.jpg',frame)
cam.release()
print(frame)