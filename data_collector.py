import cv2
import numpy as np
import pyautogui as pg
import time
import sys, os


if __name__ == "__main__":

    if sys.platform == "darwin":
        region = (791, 200, 220, 750)
    else:
        raise NotImplementedError

    if len(os.listdir("collected_data")) > 0:
        img_nums = []
        for file in os.listdir("collected_data"):
            split_file_name = file.split(".")
            if split_file_name[-1] == "png":
                img_nums.append(int(split_file_name[0].split("_")[-1]))
        n = max(img_nums) + 1
    else:
        n = 0

    time.sleep(3)
    while True:

        img = np.array(pg.screenshot("collected_data/1.png", region = region))
        file_path = f"collected_data/image_{n}.png"
        cv2.imwrite(file_path, cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        print(f"Captured -- {file_path}")

        n += 1
        time.sleep(10)

