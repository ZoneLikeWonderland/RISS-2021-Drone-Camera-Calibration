import torch
import glob
import cv2
import numpy as np


# WIDTH = 320
# HEIGHT = 240
WIDTH = 160 * 4
HEIGHT = 160 * 3

x_base, y_base = np.meshgrid(range(WIDTH), range(HEIGHT))

LEN = max(WIDTH, HEIGHT)

fade_dist = (((x_base - WIDTH / 2) / LEN)**2 + ((y_base - HEIGHT / 2) / LEN)**2)**0.5


def read_extra():
    extraset = sorted(glob.glob(r"E:\RISS\dataclip\extra\*.png"))
    length = len(extraset) // 3
    ith = np.random.randint(0, length)
    foreground_mask, background, foreground_point = (
        cv2.imread(extraset[ith * 3 + 1], -1).astype(np.float32) / 255,
        cv2.imread(extraset[ith * 3]).astype(np.float32) / 255,
        cv2.imread(extraset[ith * 3 + 2], -1).astype(np.float32) / 65535,
    )
    if np.random.random() < 0.5:
        foreground_mask = foreground_mask[::-1, ::-1].copy()
        background = background[::-1, ::-1].copy()
        foreground_point = foreground_point[::-1, ::-1].copy()
    return foreground_mask[None], background.transpose(2, 0, 1), foreground_point[None]


class CardSet(torch.utils.data.Dataset):
    def __init__(self):
        super().__init__()

        self.foreground_paths = glob.glob("dataclip/card/*.jpg") + glob.glob("dataclip/card/*.png")
        # self.background_paths = glob.glob("dataclip/images/*.jpg")
        self.background_paths = glob.glob(r"E:\RISS\dataclip\challenge2018.tar\challenge2018/*.jpg")
        self.background_paths += glob.glob(r"E:\RISS\test_images_bg/*.jpg") + glob.glob(r"E:\RISS\test_images_bg/*.png")

    def __getitem__(self, index):
        if np.random.random() < 0.1:
            return read_extra()
        foreground_index = index // len(self.background_paths)
        background_index = index % len(self.background_paths)

        foreground = cv2.imread(self.foreground_paths[foreground_index])
        length = max(foreground.shape) / 368
        foreground = cv2.resize(foreground, (int(foreground.shape[1] / length), int(foreground.shape[0] / length)))
        foreground = foreground.astype(np.float32) / 255

        foreground_mask = np.ones((foreground.shape[0], foreground.shape[1]), dtype=np.float32)
        foreground_point = np.zeros((foreground.shape[0], foreground.shape[1]), dtype=np.float32)

        foreground = np.pad(
            foreground,
            (
                ((max(foreground.shape) - foreground.shape[0]) // 2,
                 max(foreground.shape) - foreground.shape[0] - (max(foreground.shape) - foreground.shape[0]) // 2),
                ((max(foreground.shape) - foreground.shape[1]) // 2,
                 max(foreground.shape) - foreground.shape[1] - (max(foreground.shape) - foreground.shape[1]) // 2),
                (0, 0)
            ),
            # constant_values=1 if np.random.random() > 0.7 else 0
        )

        foreground *= (np.random.random(3) - 0.7) * 0.1 + 1
        foreground *= (np.random.random() - 0.7) * 0.3 + 1
        foreground = foreground.clip(0, None)
        foreground **= (np.random.normal(size=3, scale=0.1) + 1.5).clip(0.1, None)
        foreground = foreground.clip(0, None)
        foreground **= max(np.random.normal(scale=0.5) + 1.5, 0.1)
        foreground = foreground.clip(0, None)
        foreground *= (np.random.normal(size=3, scale=0.1) + 1).clip(0.1, None)
        foreground = foreground.clip(0, None)

        foreground_mask = np.pad(
            foreground_mask,
            (
                ((max(foreground_mask.shape) - foreground_mask.shape[0]) // 2,
                 max(foreground_mask.shape) - foreground_mask.shape[0] - (max(foreground_mask.shape) - foreground_mask.shape[0]) // 2),
                ((max(foreground_mask.shape) - foreground_mask.shape[1]) // 2,
                 max(foreground_mask.shape) - foreground_mask.shape[1] - (max(foreground_mask.shape) - foreground_mask.shape[1]) // 2),
            )
        )

        while np.random.random() < 0.2:
            foreground_overlay = np.ones_like(foreground)
            contours = np.array([
                [np.random.randint(0, foreground_overlay.shape[1]), np.random.randint(0, foreground_overlay.shape[0])],
                [np.random.randint(0, foreground_overlay.shape[1]), np.random.randint(0, foreground_overlay.shape[0])],
                [np.random.randint(0, foreground_overlay.shape[1]), np.random.randint(0, foreground_overlay.shape[0])],
            ])
            cv2.fillPoly(foreground_overlay, pts=[contours], color=((np.random.beta(0.01, 0.01) * 2)**2,) * 3)
            foreground *= foreground_overlay

        while np.random.random() < 0.1:
            foreground_overlay = np.ones_like(foreground)
            contours = np.array([
                [np.random.randint(0, foreground_overlay.shape[1]), np.random.randint(0, foreground_overlay.shape[0])],
                [np.random.randint(0, foreground_overlay.shape[1]), np.random.randint(0, foreground_overlay.shape[0])],
                [np.random.randint(0, foreground_overlay.shape[1]), np.random.randint(0, foreground_overlay.shape[0])],
            ])
            cv2.fillPoly(foreground, pts=[contours], color=(np.random.random(3)))
            # foreground_mask *= foreground_overlay

        background = cv2.imread(self.background_paths[background_index]).astype(np.float32) / 255
        background = cv2.resize(background, (WIDTH, HEIGHT))
        background **= max(np.random.normal(scale=0.5) + 1.3, 0.1)
        if np.random.random() > 0.5:
            background = background[:, ::-1]
        if np.random.random() > 0.5:
            background = background[::-1]

        src_points = np.array([
            [0, 0],
            [foreground.shape[1] - 1, 0],
            [foreground.shape[1] - 1, foreground.shape[0] - 1],
            [0, foreground.shape[0] - 1],
        ])

        # r = (np.random.random() + 0.1) * max(background.shape) * 0.2
        low = 0.05
        high = 0.3
        r = (np.random.random() * (high - low) + low) * max(background.shape)
        dst_center = np.random.random(2) * [background.shape[1] - r * 2, background.shape[0] - r * 2] + r

        start_theta = np.random.random() * 2 * np.pi
        interval_theta = np.random.beta(10, 10, 4)
        interval_theta = np.random.beta(25, 25, 4)
        interval_theta[interval_theta < 0] = 0
        interval_theta[interval_theta > 1] = 1
        interval_theta /= interval_theta.sum()
        interval_theta *= 2 * np.pi
        interval_theta = np.cumsum(interval_theta)
        theta = interval_theta + start_theta
        dx = np.cos(theta)
        dy = np.sin(theta)
        corner = dst_center[..., None] + np.array((dx, dy)) * r
        dst_points = corner.T

        H, inuse = cv2.findHomography(src_points, dst_points)
        foreground_mask = cv2.warpPerspective(foreground_mask, H, background.shape[1::-1])
        p1 = H@np.array((
            (
                (max(foreground_point.shape) - foreground_point.shape[1]) // 2,
                (max(foreground_point.shape) - foreground_point.shape[0]) // 2,
                1
            ),
            (
                (max(foreground_point.shape) - foreground_point.shape[1]) // 2 + foreground_point.shape[1],
                (max(foreground_point.shape) - foreground_point.shape[0]) // 2,
                1
            ),
            (
                (max(foreground_point.shape) - foreground_point.shape[1]) // 2,
                (max(foreground_point.shape) - foreground_point.shape[0]) // 2 + foreground_point.shape[0],
                1
            ),
            (
                (max(foreground_point.shape) - foreground_point.shape[1]) // 2 + foreground_point.shape[1],
                (max(foreground_point.shape) - foreground_point.shape[0]) // 2 + foreground_point.shape[0],
                1
            ),
        )).T
        p1 /= p1[2]
        # print(p1)
        foreground_point = np.zeros_like(foreground_mask)
        foreground_point[int(p1[1, 0]), int(p1[0, 0])] = 1
        foreground_point[int(p1[1, 1]), int(p1[0, 1])] = 1
        foreground_point[int(p1[1, 2]), int(p1[0, 2])] = 1
        foreground_point[int(p1[1, 3]), int(p1[0, 3])] = 1

        foreground = cv2.warpPerspective(foreground, H, background.shape[1::-1])
        foreground *= np.random.normal(size=foreground.shape, scale=0.05) + 1
        if np.random.random() > 0.5:
            ksize = np.random.randint(0, 4) * 2 + 1
            foreground = cv2.GaussianBlur(foreground, (ksize, ksize), 0)

        foreground_mask_apply = foreground_mask
        ksize = 0
        if np.random.random() > 0.5:
            ksize = np.random.randint(0, 20)
            foreground_mask_apply = cv2.dilate(foreground_mask, np.ones((ksize, ksize)))

        ksize = np.random.randint(1, max(3, ksize // 2)) * 2 + 1
        foreground_mask_apply = cv2.GaussianBlur(foreground_mask_apply, (ksize, ksize), 0)
        ksize = 23
        foreground_point = cv2.GaussianBlur(foreground_point, (ksize, ksize), 0)
        # / 0.040226486
        foreground_point /= foreground_point.max()
        # print(foreground_point.max())
        # foreground_point /= (1 / (2 * np.pi)**0.5 / ksize * 2)
        # print(foreground_point.max())
        background = foreground_mask_apply[..., None] * foreground + (1 - foreground_mask_apply[..., None]) * background

        background *= np.random.beta(1, 1, 3) * 1.2
        if np.random.random() < 0.3:
            background *= np.random.normal(loc=2, scale=0.2)
        background += np.random.normal(scale=0.01, size=background.shape)
        background *= np.random.normal(loc=1, scale=0.01 / 3, size=background.shape)

        # cv2.imshow("fade_dist", (fade_dist * 3 / (np.random.random() + 1.5))**(np.random.random() + 1.5))
        fade = 1 - (fade_dist * 2.5 / (np.random.random() + 1.5))**(np.random.random() + 1.5)
        background *= fade[..., None] * 1.2
        background = background.clip(0, 1)

        return foreground_mask[None], background.transpose(2, 0, 1), foreground_point[None]

    def __len__(self):
        return len(self.foreground_paths) * len(self.background_paths)


if __name__ == "__main__":

    card_set = CardSet()

    for f, g, p in card_set:
        print(p.max())
        # p /= p.max()
        cv2.imshow("f", f.transpose(1, 2, 0))
        cv2.imshow("g", g.transpose(1, 2, 0))
        cv2.imshow("p", p.transpose(1, 2, 0))
        cv2.waitKey(1000)
