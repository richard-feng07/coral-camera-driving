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
    resizeimg = img.resize(common.inputsize(interpreter),Image.ANTIALIAS)

    common.set_input(interpreter,resizeimg)

    interpreter.invoke()

    pose = common.output_tensor(interpreter,0).copy().reshape(_NUM_KEYPOINTS,3)
    print(pose)


if __name__ == '__main__':
    main()
