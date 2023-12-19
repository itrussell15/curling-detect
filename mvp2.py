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

class StoneInfo(StoneData):

    def __init__(self, centers, **kwargs):
        super().__init__(**kwargs)
        self.coords = centers

    def __repr__(self):
        return f"Stone(numStones={len(self.coords)}, color={self.color})"

    def distance_to_target(self, target):
        if len(self.coords) > 0:
            x_dist = np.power(self.coords[:, 1] - target[1], 2)
            y_dist = np.power(self.coords[:, 0] - target[0], 2)
            return np.sqrt(x_dist + y_dist)
        else:
            return None

class HouseDetect:

    # TODO Write ReadMe. Publish to LinkedIn?
    def __init__(self, env_variables_file, image = None):
        if image is not None:
            self.original = image.copy()
        variables = self._load_envs(env_variables_file)

        self.stone_color_options = {}
        for color in variables.stones.colors:
            self.stone_color_options[color] = StoneData(color=color, **variables.stones.colors[color])

        self._variables = variables

        if "rolling_center_n" in self._variables.house:
            self._n = self._variables.house.rolling_center_n
            self._rolling_center = np.zeros((self._variables.house.rolling_center_n, 2))
            self.clean_center = np.zeros((2,))

        # self.rolling_n = rolling_n


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
            return None, None

        center = centers.mean(axis = 0).astype(np.int64)

        if "_rolling_center" in vars(self):
            if self._n > 0:
                self._rolling_center[self._n - 1] = center
                self._n -= 1
                self.clean_center = center
            else:
                self._rolling_center = np.roll(self._rolling_center, axis = 0, shift = 1)
                self._rolling_center[0] = center
                self.clean_center = self._rolling_center[np.any(~np.isnan(self._rolling_center), axis = 1)].mean(axis = 0)

            out_center = self.clean_center.astype(np.int64)
        else:
            out_center = center

        if display:
            if display_img is None:
                raise ValueError(f"Please pass in an image to draw on")
            return out_center, self.show_center(display_img, out_center, text = False)
        else:
            return out_center, None

    def show_center(self, draw_img, center_coords, text = True, color = (0, 255, 0)):
        cv2.circle(
            draw_img,
            center_coords.astype(np.int64),
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

    # TODO Reduce false positives on Blue
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
            if area > self._variables.stones.min_area and len(hull) > 5:
                try:
                    ellipse = cv2.fitEllipse(hull)
                except Exception as e:
                    print()

                if np.diff(ellipse[1]) < self._variables.stones.axis_tolerance:
                    centers.append(ellipse[0])

        centers = StoneInfo(np.array(centers).astype(np.int64), **vars(stone_data))

        if not display:
            return centers, None
        else:
            if len(centers.coords) > 0:

                if display_img is not None:
                    # raise ValueError(f"Please pass in an image to draw on")

                    for center in centers.coords:
                        self._display_circle(
                            img = display_img,
                            center = center,
                            color = stone_data.display_color,
                            radius = self._variables.stones.draw_radius
                        )
                    return centers, display_img
                else:
                    return centers, display_img
            else:
                return centers, img

    def stone_distances(self, target, stones):

        if len(stones) <= 0 or stones is None:
            raise ValueError(f"No stones detected. Unable to determine closest")

        if target is None:
            raise ValueError(f"No center able to be determined. Unable to determine closest")

        distances = np.array([])
        for n, color in enumerate(stones):
            if len(color.coords) <= 0 or color.coords is None:
                print(f"No stones detected for {color.color}.")
                # raise ValueError(f"No stones detected for {color.color}.")
            else:
                try:
                    tmp_distance = color.distance_to_target(target)
                    if tmp_distance is not None:
                        tmp_distance = np.vstack([tmp_distance, np.full_like(tmp_distance, fill_value=color.label)]).transpose()
                        if n == 0 or len(distances) <= 0:
                            distances = tmp_distance
                        else:
                            distances = np.vstack([distances, tmp_distance])
                except Exception as e:
                    print()

        if len(distances) > 0:
            return distances[distances[:, 0].argsort()]
        else:
            return None

    # TODO Figure out error handling of None values
    def current_score(self, img, color1, color2, display = True):
        display_img = img.copy() if display else None
        center, display_img = self.estimate_center(
            img = img,
            display = display,
            display_img = display_img
        )

        color1_stones, display_img = House.find_stones(
            img = img,
            stone_color = color1,
            display = True,
            display_img = display_img,
        )

        color2_stones, display_img = House.find_stones(
            img=img,
            stone_color=color2,
            display=True,
            display_img=display_img,
        )

        distances = self.stone_distances(
            target = center,
            stones = [color1_stones, color2_stones]
        )

        if distances is not None:
            score = 0
            close_label = distances[0, 1]
            next_label = close_label
            while next_label == close_label or score == len(distances):
                score += 1
                if score <= len(distances) - 1:
                    next_label = distances[score, 1]
                else:
                    break
                try:
                    next_label = distances[score, 1]
                except Exception as e:
                    print()

            score_summary = {
                "score": score,
                "color": color1 if color1_stones.label == close_label else color2
            }

            if display:
                cv2.putText(
                    display_img,
                    f"{score_summary['color'].capitalize()}: {score_summary['score']}",
                    (5, 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA
                )
        else:
            score_summary = None

        return score_summary, display_img



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

    file = "data/lots.png"
    envs = "data/env_variables.json"

    # img = cv2.imread(file)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #
    #
    # House = HouseDetect(envs)
    # score, display_img = House.current_score(
    #     img = img,
    #     color1 = "blue",
    #     color2 = "yellow"
    # )
    #
    # plt.imshow(display_img)
    # plt.show()

    # Top
    frame = {'left': 2368, 'top': -450, 'width': 178, 'height': 290}

    # Bottom
    # frame = {'left': 2368, 'top': -145, 'width': 178, 'height': 290}

    House = HouseDetect(env_variables_file = envs)

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
            img = np.array(img)

            score, img = House.current_score(img, "blue", "yellow")

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            cv2.imshow('test', img)
            print(f"Running")

            if cv2.waitKey(33) & 0xFF in (
                    ord('q'),
                    27,
            ):
                break
            time.sleep(0.1)








