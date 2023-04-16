import cv2

cam = cv2.VideoCapture(0)
_, frame = cam.read()
cv2.imwrite('photo.jpg',frame)
cam.release()
print(frame)