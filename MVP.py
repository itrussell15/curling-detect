import cv2
import numpy as np
import matplotlib.pyplot as plt
import json, munch
import pandas as pd

from Stones import StoneData, StoneLocations


np.seterr(divide='ignore', invalid='ignore')

HOUSE_DIAM = 90

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

    def estimate_center(self, img):
        self.image = self._process_image(img)
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

        return out_center

    def _process_image(self, img):
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
    def find_stones(self, img, stone_color, center = None):
        stone_data = self.stone_color_options[stone_color]

        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        lower = np.array(stone_data.bottom_range, dtype=np.uint8)
        upper = np.array(stone_data.top_range, dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)
        stones = cv2.bitwise_and(img, img, mask = mask)
        stones = cv2.cvtColor(stones, cv2.COLOR_RGB2GRAY)
        _, stones = cv2.threshold(stones, 20, 255, cv2.THRESH_BINARY)
        _, stones = cv2.threshold(stones, 60, 255, cv2.THRESH_TRUNC)
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

        stones = StoneLocations(np.array(centers).astype(np.int64), **vars(stone_data))

        if center is not None:
            stones.distance_to_target(center)

        return stones

    def stone_df(self, stone1, stone2):
        stones = pd.concat([stone1.toDf(), stone2.toDf()])

        if "distance" in stones.columns:
            stones = stones.sort_values(by="distance")
        return stones

    def current_score(self, stones):

        if "distance" in stones.columns:
            valid_stones = stones[stones["distance"] < self._variables.house.house_size]

            score = 0

            if len(valid_stones) >= 1:
                closest = valid_stones.iloc[0]
                next_closest = valid_stones.iloc[1] if len(valid_stones) >= 2 else None
                while closest is not None and next_closest is not None and next_closest.color == closest.color:
                    score += 1

                    if score >= len(valid_stones):
                        break

                    closest = next_closest
                    next_closest = valid_stones.iloc[score]

                score = score + 1 if score == 0 else score


            if score > 0:
                return {"color": stones.iloc[0].color, "score": score}
            else:
                return None

    def show_stones(self, draw_img, stones, scale = 1):
        for n, stone in stones.iterrows():
            if "distance" in stones.columns:
                thickness = -1 if stone.distance > self._variables.house.house_size else 1
            else:
                thickness = 1

            cv2.circle(
                draw_img,
                (int(scale * stone.x_coords), int(scale * stone.y_coords)),
                radius=4,
                thickness=thickness,
                color=self._variables.stones.colors[stone.color].display_color
            )
        return draw_img

    def show_center(self, draw_img, center_coords, text=True, color=(0, 255, 0)):
        if center_coords is not None:
            cv2.circle(
                draw_img,
                center_coords.astype(np.int64),
                radius=2,
                thickness=-1,
                color=color
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

    def show_score(self, display_img, score):
        if score is not None:
            text = f"{score['color'].capitalize()}: {score['score']}"
        else:
            text = "Blank"

        cv2.putText(
            display_img,
            text,
            (5, 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )
        return display_img

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
    # frame = {'left': 2368, 'top': -440, 'width': 178, 'height': 285}
    #
    # Bottom
    frame = {'left': 2368, 'top': -145, 'width': 178, 'height': 290}

    House = HouseDetect(env_variables_file = envs)

    color1 = "blue"
    color2 = "yellow"

    with mss() as sct:
        while True:
            screenShot = sct.grab(frame)
            img = Image.frombytes(
                'RGB',
                (screenShot.width, screenShot.height),
                screenShot.rgb,
            )
            img = np.array(img)

            house_center = House.estimate_center(img)
            stones1 = House.find_stones(img, color1, center=house_center)
            stones2 = House.find_stones(img, color2, center=house_center)
            stones_df = House.stone_df(stones1, stones2)
            score = House.current_score(stones_df)

            img = House.show_center(img, house_center, text=False)
            img = House.show_stones(img, stones_df, scale = 1)
            img = House.show_score(img, score)

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            cv2.imshow('test', img)
            print(f"Current Score: {score if score is not None else 'None'}")

            if cv2.waitKey(33) & 0xFF in (
                    ord('q'),
                    27,
            ):
                break
            time.sleep(0.1)








