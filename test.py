import torch
import glob
import cv2
import numpy as np


if __name__ == "__main__":

    net = torch.load(r"E:\RISS\runs\Aug02_23-07-37_DESKTOP-HN2581Fhide2\best_test_error.pth")
    net.eval()

    for path in glob.glob("test_images/*.png") + glob.glob("test_images/*.jpg"):
        img = cv2.imread(path).astype(np.float32) / 255
        img = cv2.resize(img, (640, 480))

        # card = cv2.imread(r"E:\RISS\dataclip\card\1d1cd88737f4f3b746837abb478631d3.jpg").astype(np.float32) / 255
        # card = cv2.resize(card, (30, 50))
        # img = cv2.seamlessClone(card, img, np.ones_like(card, np.uint8), (100, 100), cv2.NORMAL_CLONE)
        # img[100:100 + card.shape[0], 100:100 + card.shape[1]] = card

        with torch.no_grad():
            pred = net(torch.tensor(img).cuda().permute(2, 0, 1).unsqueeze(0)).squeeze().cpu().numpy()

        c = np.argmax(pred[1])

        cv2.imshow("img", img)
        cv2.imshow("pred", pred[0])
        cv2.imshow("pred_c", pred[1])
        cv2.waitKey()
