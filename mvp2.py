import cv2
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List
import json, munch


np.seterr(divide='ignore', invalid='ignore')

HOUSE_DIAM = 90

@dataclass
class StoneData:
    color: str
    display_color: List[int]
    top_range: List[int]
    bottom_range: List[int]
    label: int

    def __repr__(self):
        return f"Stone(color={self.color})"

class HouseDetect:

    def __init__(self, env_variables_file, image = None):
        if image is not None:
            self.original = image.copy()
        variables = self._load_envs(env_variables_file)

        self.stone_color_options = {}
        for color in variables.stones.colors:
            self.stone_color_options[color] = StoneData(color=color, **variables.stones.colors[color])

        self._variables = variables

    def _load_envs(self, file):
        with open(file, 'r') as f:
            data = json.load(f)
        return munch.munchify(data)

    def _euclidean_distance(self, target, options):
        pass

    def _display_circle(self, img, center, color, radius):
        return cv2.circle(
            img = img,
            center = center,
            color = color,
            radius = radius,
            thickness = 1
        )

    def estimate_center(self, img, display = True, display_img = None):
        self.image = self.process_image(img)
        canny = cv2.Canny(self.image, 100, 200)
        self.canny = cv2.GaussianBlur(canny, ksize=(5, 5), sigmaX=1)

        centers, diameters = self.detect_circles(self.canny)

        min_diam = 70
        max_diam = 1e3
        std_range = 0.75
        centers, diameters = self.filter_circles(centers, diameters, min_diam, max_diam, std_range)

        if centers is None:
            return None


        center = centers.mean(axis = 0).astype(np.int64)
        if display:
            if display_img is None:
                raise ValueError(f"Please pass in an image to draw on")

            return center, self.show_center(display_img, center)

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
        return draw_img

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

    def find_stones(self, img, stone_color, display = True, display_img = None):
        stone_data = self.stone_color_options[stone_color]

        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        lower = np.array(stone_data.bottom_range, dtype=np.uint8)
        upper = np.array(stone_data.top_range, dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)
        stones = cv2.bitwise_and(img, img, mask = mask)
        stones = cv2.cvtColor(stones, cv2.COLOR_RGB2GRAY)
        contours, hierarchy = cv2.findContours(stones.astype(np.uint8), 1, 1)

        centers = []
        for contour in contours:
            hull = cv2.convexHull(contour)
            area = cv2.contourArea(hull)
            if area > self._variables.stones.min_area:
                ellipse = cv2.fitEllipse(hull)
                if np.diff(ellipse[1]) < self._variables.stones.axis_tolerance:
                    centers.append(ellipse[0])

        centers = np.array(centers).astype(np.int64)

        if not display:
            return centers
        else:
            if len(centers) > 0:

                if display_img is None:
                    raise ValueError(f"Please pass in an image to draw on")

                for center in centers:
                    self._display_circle(
                        img = display_img,
                        center = center,
                        color = stone_data.display_color,
                        radius = self._variables.stones.draw_radius
                    )
                return centers, display_img
            else:
                return centers, img

    def find_closest(self, center, stones):

        if len(stones) <= 0 or stones is None:
            raise ValueError(f"No stones detected. Unable to determine closest")

        if center is None:
            raise ValueError(f"No center able to be determined. Unable to determine closest")

        distances = self._euclidean_distance(
            target=center,
            options=stones[:, :1]
        )


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
    # from mss import mss
    from PIL import Image
    import time

    file = "data/lots2.png"
    envs = "data/env_variables.json"

    img = cv2.imread(file)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


    House = HouseDetect(envs)
    center, display_img = House.estimate_center(img, display = True, display_img=img.copy())
    blue_stones, display_img = House.find_stones(
        img = img,
        stone_color = "blue",
        display = True,
        display_img = display_img
    )
    yellow_stones, display_img = House.find_stones(
        img=img,
        stone_color="yellow",
        display=True,
        display_img=display_img
    )

    plt.imshow(display_img)
    plt.show()








