import torch
import glob
import cv2
import numpy as np
from postprocess import *

if __name__ == "__main__":
    device = "cuda"
    # device = "cpu"
    net = torch.load(r"E:\RISS\runs\May29_15-54-13_DESKTOP-HN2581F\best_test_error.pth")
    net.to(device)
    net.eval()

    cap = cv2.VideoCapture(0)
    count = 0
    coeff = [1, 1, 1]
    while True:
        ret, img = cap.read()
        img_raw = img
        img = img.astype(np.float32) / 255

        doe = [1, np.sin(count * 0.2) * 0.3 + 0.6, np.cos(count * 0.1) * 0.3 + 0.6]
        img *= doe
        img_resize = cv2.resize(img, (320, 240))

        import time
        start = time.time()
        with torch.no_grad():
            pred = net(torch.tensor(img_resize).to(device).permute(2, 0, 1).unsqueeze(0)).squeeze().cpu().numpy()
        # print("pred time", time.time() - start)
        # print(pred.mean(), pred.min(), pred.max())

        cv2.imshow("img", img)
        cv2.imshow("pred", pred[0])
        cv2.imshow("pred_c", pred[1])

        cv2.imwrite("img.png", img * 255)
        cv2.imwrite("pred.png", pred[0] * 255)
        cv2.imwrite("pred_c.png", pred[1] * 255)

        new_coeff = pickout(img, pred[0], pred[1])
        if new_coeff is not None:
            coeff = new_coeff
            # print(new_coeff * doe)

        img *= coeff

        cv2.imshow("img_balanced", img)

        cv2.waitKey(1)

        count += 1
