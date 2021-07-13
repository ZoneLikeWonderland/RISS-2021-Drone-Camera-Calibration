from openvino.inference_engine import IECore
import numpy as np
import cv2
import time

ie = IECore()
# model="ctdet_coco_dlav0_512.onnx"
model = "card/card.xml"
net = ie.read_network(model=model)
input_blob = next(iter(net.input_info))
out_blob = next(iter(net.outputs))
net.batch_size = 1

n, c, h, w = net.input_info[input_blob].input_data.shape
print(n, c, h, w)
images = np.ndarray(shape=(n, c, h, w))

image = cv2.imread(r"C:\Users\14682\Downloads\image_left_03.png")
# image = cv2.resize(image, (640, 480))
image = cv2.resize(image, (640, 480))
image = (image.astype(np.float32)/255).transpose(2, 0, 1)[None]

exec_net = ie.load_network(network=net, device_name="CPU")
start = time.time()
for i in range(100):
    res = exec_net.infer(inputs={input_blob: image})
print('infer total time is %.4f s' % ((time.time()-start)/100))

res = res["output"]
block = res[0, 0]
point = res[0, 1]
# print(res.shape)

cv2.imshow("block", block)
cv2.imshow("point", point)
cv2.waitKey()
