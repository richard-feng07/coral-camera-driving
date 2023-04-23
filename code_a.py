from PIL import Image
from PIL import ImageDraw
from pycoral.adapters import common
from pycoral.utils.edgetpu import make_interpreter

_NUM_KEYPOINTS = 17

def main():
    modelpath = "lite-model_movenet_singlepose_lightning_3.tflite"
    imagepath = "image.jpg"

    interpreter = make_interpreter(modelpath)
    interpreter.allocate_tensors()

    img = Image.open(imagepath)
    resizeimg = img.resize(common.input_size(interpreter),Image.ANTIALIAS)

    common.set_input(interpreter,resizeimg)

    interpreter.invoke()

    pose = common.output_tensor(interpreter,0).copy().reshape(_NUM_KEYPOINTS,3)
    print(pose)
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
