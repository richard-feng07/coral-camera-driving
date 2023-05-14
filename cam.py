import cv2
import gi
import time
gi.require_version('Gst','1.0')
from gi.repository import Gst

from PIL import Image
from PIL import ImageDraw
from pycoral.adapters import common
from pycoral.utils.edgetpu import make_interpreter

_NUM_KEYPOINTS = 17

Gst.init(None)

pipeline = Gst.parse_launch("v4l2src device=/dev/video0 ! video/x-raw,width=640,height=480 ! videoconvert ! appsink")
cam = cv2.VideoCapture(1)
# cam.open(pipeline)








#cam.release()


def processImage(interpreter):
    _, frame = cam.read()
    cv2.imwrite('photo.jpg',frame)
    img = Image.fromarray(frame)
    resizeimg = img.resize(common.input_size(interpreter),Image.ANTIALIAS)
    common.set_input(interpreter,resizeimg)
    interpreter.invoke()
    pose = common.output_tensor(interpreter,0).copy().reshape(_NUM_KEYPOINTS,3)
    return pose

def main():
    modelpath = "lite-model_movenet_singlepose_lightning_3.tflite"
    imagepath = "image.jpg"

    interpreter = make_interpreter(modelpath)
    interpreter.allocate_tensors()
    counter = 0

    while True:
        pose = processImage(interpreter)

        counter = counter + 1
        if(counter >= 300):
            counter = 0
            print(pose)
        time.sleep(0.0333)

    drawpose(pose,img)

def drawpose(pose, img):
    draw = ImageDraw.Draw(img)
    width, height = img.size
    for i in range(0,_NUM_KEYPOINTS):
        draw.ellipse(
            xy = [
                pose[i][1] * width - 2, pose[i][0] * height - 2,
                pose[i][1] * width + 2, pose[i][0] * height + 2
            ],
            fill = (255,0,0)
        )
    img.save("results.jpg")
    resultImage = Image.open("results.jpg")
    resultImage.show()

if __name__ == '__main__':
    main()
