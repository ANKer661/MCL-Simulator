from typing import TypeAlias

import matplotlib.pyplot as plt
import numpy as np
from shapely import Point, Polygon
from shapely.plotting import plot_polygon

RealNumber: TypeAlias = int | float


class Map:
    """
    A class representing a 2D map with obstacles.
    The map is defined by a polygonal boundary and a list of obstacles.
    The map is represented as a Shapely Polygon object.
    Obstacles are represented as holes in the polygon.
    """

    def __init__(self, boundary: list[tuple], obstacles: list[list[tuple]]) -> None:
        self.world = Polygon(boundary, holes=obstacles)

    def visualize(self, ax=None) -> None:
        """
        Visualize the map on the given axes.
        If no axes are provided, create a new figure and axes.
        """
        if ax is None:
            fig, ax = plt.subplots()
            ax.set_aspect("equal")
        plot_polygon(self.world, ax=ax, add_points=False, color="gray", alpha=0.4)

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
