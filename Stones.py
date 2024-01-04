from dataclasses import dataclass
import numpy as np
from typing import List
import pandas as pd

@dataclass
class StoneData:
    color: str
    display_color: List[int]
    top_range: List[int]
    bottom_range: List[int]
    label: int

    def __repr__(self):
        return f"Stone(color={self.color})"

class StoneLocations(StoneData):

    def __init__(self, centers, **kwargs):
        super().__init__(**kwargs)
        self.coords = centers
        self.distances = None

    def __repr__(self):
        return f"Stone(numStones={len(self.coords)}, color={self.color})"

    def distance_to_target(self, target):
        if len(self.coords) > 0:
            x_dist = np.power(self.coords[:, 1] - target[1], 2)
            y_dist = np.power(self.coords[:, 0] - target[0], 2)
            self.distances = np.sqrt(x_dist + y_dist)
            return self.distances
        else:
            return None

    def toDf(self):
        df = pd.DataFrame()
        df[["x_coords", "y_coords"]] = self.coords
        df["label"] = self.label
        df["color"] = self.color

        if self.distances is not None:
            df["distance"] = self.distances

        return df
