import torch
import glob
import cv2
import numpy as np
from postprocess import *
import glob
import json
from fit import *
import time

WIDTH = 640
HEIGHT = 480


data = []

mul = None
x_base, y_base = None, None


if __name__ == "__main__":
    device = "cuda"
    # device = "cpu"
    net = torch.load(r"best_test_error.pth")
    net.to(device)
    net.eval()

    # cap = cv2.VideoCapture(0)
    # while True:

    # l = r"C:\Users\14682\Documents\CODE\RISS\ros_test\color_files\20210604_indoor_01.bag\_xic_stereo_left_image_raw\*.jpg"
    # l = r"C:\Users\14682\Documents\CODE\RISS\ros_test\color_files\20210604_outdoor_01.bag\_xic_stereo_left_image_raw\*.jpg"
    l = r"C:\Users\14682\Documents\CODE\RISS\ros_test\color_files\20210607_indoor_spider_board_01.bag\_xic_stereo_left_image_raw\*.jpg"

    xs = []
    ys = []
    coeff = np.array([1, 1, 1])
    last_fit_time = time.time()
    last_fit_n = 0
    for i, path in enumerate(glob.glob(l)):

        img = cv2.imread(path).astype(np.float32)/255
        # img = cv2.resize(img, (960, 540))
        img = cv2.resize(img, (WIDTH, HEIGHT), cv2.INTER_AREA)
        img_resize = cv2.resize(img, (WIDTH, HEIGHT), cv2.INTER_AREA)

        with torch.no_grad():
            pred = net(torch.tensor(img_resize).to(device).permute(
                2, 0, 1).unsqueeze(0)).squeeze().cpu().numpy()

        cv2.imshow("img", img)
        cv2.imshow("pred", pred[0])
        cv2.imshow("pred_c", pred[1])

        cv2.imwrite("img.png", img * 255)
        cv2.imwrite("pred.png", pred[0] * 255)
        cv2.imwrite("pred_c.png", pred[1] * 255)

        ret = pickout(img, pred[0], pred[1])
        if ret is not None:
            new_coeff, intensity = ret
            coeff = new_coeff
            data.append(intensity)
            xs.append(intensity[0])
            ys.append(intensity[1])

            if time.time()-last_fit_time > 5 and len(xs) > last_fit_n:
                last_fit_time = time.time()
                last_fit_n = len(xs)
                alpha = fit(xs, ys)
                print(alpha)

                # if mul is None:
                #     mul = np.zeros(img.shape[:2])
                #     x_base, y_base = np.meshgrid(
                #         range(img.shape[1]), range(img.shape[0]))
                #     # mul=(x_base-c.shape[0]//2)**2+(x_base-c.shape[0]//2)

                #     fade_dist = (
                #         ((x_base - img.shape[1] / 2))**2 + ((y_base - img.shape[0] / 2))**2)**0.5/max(img.shape)
                # # mul=a
                # mul = np.zeros(img.shape[:2])
                # # yt = np.zeros_like(xt)
                # k = 3
                # for i in range(0, k+1):
                #     mul += alpha[i]*fade_dist**(i*2)
                # cv2.imshow("mul", mul)

        img *= coeff
        # if mul is not None:
        #     img /= mul

        cv2.imshow("img_balanced", img)

        cv2.waitKey(1)

    json.dump(data, open("data.json", "w"))
    plt.ioff()
    plt.show()
