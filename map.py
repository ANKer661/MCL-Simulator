from __future__ import annotations

import json
import typing
from typing import TypeAlias

import matplotlib.pyplot as plt
import numpy as np
from shapely import Point, Polygon
from shapely.plotting import plot_polygon

RealNumber: TypeAlias = int | float

if typing.TYPE_CHECKING:
    from matplotlib.axes import Axes


class Map:
    """
    A class representing a 2D map with obstacles.
    The map is defined by a polygonal boundary and a list of obstacles.
    The map is represented as a Shapely Polygon object.
    Obstacles are represented as holes in the polygon.
    """

    def __init__(self, boundary: list[tuple], obstacles: list[list[tuple]]) -> None:
        self.boundary = boundary
        self.obstacles = obstacles
        self.world = Polygon(boundary, holes=obstacles)

    def visualize(self, ax=None) -> Axes:
        """
        Visualize the map on the given axes.
        If no axes are provided, create a new figure and axes.
        """
        if ax is None:
            fig, ax = plt.subplots()
            ax.set_aspect("equal")
        plot_polygon(self.world, ax=ax, add_points=False, color="gray", alpha=0.4)

        return ax

    def sample_points(self, n: int, robot_radius: RealNumber) -> list[tuple]:
        """
        Sample n points in the world, avoiding obstacles.

        Returns:
            list[tuple]: List of (x, y) coordinates of the sampled points.
        """
        min_x, min_y, max_x, max_y = self.world.bounds
        points = []
        while len(points) < n:
            x = np.random.uniform(min_x, max_x)
            y = np.random.uniform(min_y, max_y)
            point = Point(x, y).buffer(robot_radius)
            if self.world.contains(point):
                points.append((x, y))
        return points

    def get_bounds(self) -> tuple[float, float, float, float]:
        """
        Get the bounds of the map.

        Returns:
            tuple[float, float, float, float]: (min_x, min_y, max_x, max_y)
        """
        return self.world.bounds

    def save2json(self, file_name: str) -> None:
        """
        Save the map to a json file.

        Args:
            file_name (str): The name of the file to save the map to.
        """
        if not file_name.endswith(".json"):
            raise ValueError("File name must end with .json")

        map_data = {
            "boundary": self.boundary,
            "obstacles": self.obstacles,
        }
        with open(file_name, "w") as f:
            json.dump(map_data, f, indent=4)

        ax = self.visualize()
        fig = ax.figure
        fig.tight_layout()
        fig.savefig(file_name.replace(".json", "_preview.png"), dpi=300)

    @classmethod
    def load_map_from_json(cls, file_name: str) -> None:
        """
        Load the map from a json file.

        Args:
            file_name (str): The name of the file to load the map from.
        """
        with open(file_name, "r") as f:
            map_data = json.load(f)

        boundary = map_data["boundary"]
        obstacles = map_data["obstacles"]
        return cls(boundary, obstacles)
