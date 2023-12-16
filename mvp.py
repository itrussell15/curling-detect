import cv2
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass

np.seterr(divide='ignore', invalid='ignore')

HOUSE_DIAM = 90

@dataclass
class StoneData:
    color: str
    display_color: list[int]
    top_range: list[int]
    bottom_range: list[int]

    def __repr__(self):
        return f"Stone(color={self.color})"

class HouseDetect:

    def __init__(self, image = None):
        if image is not None:
            self.original = image.copy()

    def estimate_center(self, img):
        self.image = self.process_image(img)
        canny = cv2.Canny(self.image, 100, 200)
        self.canny = cv2.GaussianBlur(canny, ksize=(5, 5), sigmaX=1)

        centers, diameters = self.detect_circles(self.canny)

        min_diam = 70
        max_diam = 1e3
        std_range = 0.75
        centers, diameters = self.filter_circles(centers, diameters, min_diam, max_diam, std_range)


        if centers is not None:
            return centers.mean(axis = 0).astype(np.int64)
        else:
            return None

    def show_center(self, draw_img, center_coords, text = True, color = (0, 255, 0)):
        cv2.circle(
            draw_img,
            center_coords,
            radius = 2,
            thickness = -1,
            color = color
        )


        if text:
            text_add = np.array([-30, -10])

            cv2.putText(
                draw_img,
                f"Center",
                center_coords + text_add,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 0),
                2,
                cv2.LINE_AA
            )

    def process_image(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        sharp = sharpen_image(img)
        return cv2.cvtColor(sharp, cv2.COLOR_RGB2GRAY)

    def detect_circles(self, canny):
        contours, _ = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        axes = np.full((len(contours), 2), fill_value=-1)
        centers = np.full_like(axes, fill_value=-1)

        for n, contour in enumerate(contours):

            if len(contour) > 5:
                ellipse = cv2.fitEllipse(contour)
                centers[n], axes[n], _ = ellipse

        mask = np.all(axes > 1, axis=1)
        axes = axes[mask]
        centers = centers[mask]
        ratio = np.max(axes, axis=1) / np.min(axes, axis=1)

        mask = ratio < 1.2
        axes = axes[mask]

        centers = centers[mask].astype(np.int32)
        diameters = axes.mean(axis=1)
        mask = np.logical_and(diameters > 70, diameters < 550)

        centers = centers[mask]
        diameters = diameters[mask]

        return (centers, diameters)

    def filter_circles(self, center, diam, min, max, std_range):
        mask = np.logical_and(diam > min, diam < max)
        center = center[mask]
        diameter = diam[mask]

        std = center.std(axis=0)
        means = center.mean(axis=0)

        if np.any(std > 10):
            within_std = std * np.array([
                [-std_range],
                [ std_range]
            ]) + means

            lower_bound = np.all(center > within_std[0], axis=1)
            upper_bound = np.all(center < within_std[1], axis=1)
            mask = np.logical_and(lower_bound, upper_bound)

            center = center[mask]
            diameter = diameter[mask]

        if len(center) != 0:
            return (center, diameter)
        else:
            return None, None

    def filter_ice(self, img):
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        lower = np.array([0, 0, 30], dtype=np.uint8)
        upper = np.array([255, 130, 210], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)
        no_ice = cv2.bitwise_and(hsv, hsv, mask=np.invert(mask))
        return cv2.cvtColor(no_ice, cv2.COLOR_HSV2RGB)

    def filter_house(self, img):

        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

        lower = np.array([
            0,
            0,
            0
        ], dtype=np.uint8)
        upper = np.array([
            30,
            255,
            255
        ], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)

        # mask = np.bitwise_or(mask1, mask2)

        mask = cv2.erode(
            mask,
            kernel = (5, 5),
            iterations=3
        )
        # mask = cv2.dilate(
        #     mask,
        #     kernel=(5, 5),
        #     iterations=3
        # )

        return cv2.bitwise_and(img, img, mask=mask), mask

def find_stones(img, color_range, min_area = 20):


    contours, hierarchy = cv2.findContours(img.copy(), 1, 2)



def sharpen_image(img):
    kernel = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0],
    ])
    return cv2.filter2D(img, -1, kernel)

def to_canny(img):
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # sharp = sharpen_image(img)
    bw = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return cv2.Canny(bw, 100, 200)

if __name__ == "__main__":

    import numpy as np
    import cv2
    from mss import mss
    from PIL import Image
    import time

    # Top
    frame = {'left': 2368, 'top': -450, 'width': 178, 'height': 290}

    # Bottom
    # frame = {'left': 2368, 'top': -145, 'width': 178, 'height': 290}

    House = HouseDetect()

    limit = 30
    rolling_center = np.empty((limit, 2))

    n = limit
    with mss() as sct:
        while True:
            screenShot = sct.grab(frame)
            img = Image.frombytes(
                'RGB',
                (screenShot.width, screenShot.height),
                screenShot.rgb,
            )

            img = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)
            center = House.estimate_center(img)
            if n > 0:
                rolling_center[n - 1] = center
                n -= 1
            else:
                rolling_center = np.roll(rolling_center, axis=0, shift=1)
                rolling_center[0] = center
                clean_center = rolling_center[np.any(~np.isnan(rolling_center), axis = 1)]
                center = clean_center.mean(axis = 0, dtype = np.int64)

            if center is not None and not np.any(np.isnan(center)):
                print(f"Center is estimated to be {center}")
                house_mask = cv2.circle(
                    img=np.zeros_like(img),
                    center=tuple(center),
                    radius=HOUSE_DIAM,
                    color=(255, 255, 255),
                    thickness=-1
                )

                img = House.filter_ice(img)
                img, mask = House.filter_house(img)
                img = cv2.bitwise_and(img, img, mask=house_mask[..., -1])

                img = np.where(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) > 0, 255, 0).astype(np.uint8)
                stones = find_stones(img)
                House.show_center(img, center, text=False)
            else:
                pass

            cv2.imshow('test', img)


            if cv2.waitKey(33) & 0xFF in (
                    ord('q'),
                    27,
            ):
                break
            time.sleep(0.2)







