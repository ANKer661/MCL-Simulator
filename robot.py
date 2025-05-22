from __future__ import annotations

import typing
from typing import TypeAlias

import matplotlib.pyplot as plt
import numpy as np
from shapely import LineString, Point
from shapely.plotting import _path_from_polygon, plot_polygon

if typing.TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.patches import PathPatch
    from matplotlib.quiver import Quiver
    from shapely import Polygon

RealNumber: TypeAlias = int | float


class Robot:
    """
    Robot class representing a robot in a 2D space.

    Attributes:
        x (float): x-coordinate of the robot's center position.
        y (float): y-coordinate of the robot's center position.
        theta (float): orientation of the robot in radians.
        radius (float): radius of the robot.
        max_distance (float): maximum distance for robot's distance sensor.
        v_sigma (float): Standard deviation of linear velocity noise.
        w_sigma (float): Standard deviation of angular velocity noise.
        measurement_sigma (float): Standard deviation of measurement noise.
    """

    def __init__(
        self,
        x: RealNumber,
        y: RealNumber,
        theta: float,
        radius: RealNumber,
        max_distance: RealNumber,
        v_sigma: RealNumber = 0.1,
        w_sigma: RealNumber = 0.1,
        measurement_sigma: RealNumber = 1,
    ) -> None:
        self.theta = theta
        self.center_pos = [x, y]
        self.radius = radius
        self.max_distance = max_distance
        self.v_sigma = v_sigma
        self.w_sigma = w_sigma
        self.measurement_sigma = measurement_sigma

    def get_shape(self) -> Polygon:
        """
        Get the shape of the robot as a shapely Polygon.
        The shape is a point with a buffer of the robot's radius.
        """
        x, y = self.center_pos
        return Point(x, y).buffer(self.radius)

    def get_direction(self) -> tuple[float, float, float, float]:
        """
        Get the direction of the robot as a tuple of (x, y, dx, dy).
        Used for visualization.
        """
        x, y = self.center_pos
        dx = self.radius * np.cos(self.theta)
        dy = self.radius * np.sin(self.theta)
        return x + dx, y + dy, 4 * dx, 4 * dy

    def visualize(
        self, ax: Axes = None, alpha: float = 1.0, color: str = "blue"
    ) -> list[PathPatch, Quiver]:
        """
        Visualize the robot on a given axis.

        Returns:
            list[PathPatch, Quiver]: A list containing:
                - polygon_path_patch: PathPatch object for the robot shape
                - arrow: Quiver object for the robot direction
        """
        if ax is None:
            fig, ax = plt.subplots()
            ax.set_aspect("equal")

        polygon_path_patch = plot_polygon(
            self.get_shape(),
            ax=ax,
            color=color,
            add_points=False,
            alpha=alpha,
            edgecolor="black",
            linewidth=0,
        )

        arrow = ax.quiver(
            *self.get_direction(),
            angles="xy",
            scale_units="xy",
            scale=1,
            width=0.003,
            color=color,
            alpha=alpha,
        )

        return [polygon_path_patch, arrow]

    def measure_distance(self, world: Polygon) -> float:
        """
        Simulate a ray from the robot's center in the direction of its
        orientation (theta) and measure the distance to the world boundary.

        Returns:
            distance (float): The distance to the world boundary.
                if the ray does not intersect the boundary, return max_distance.
        """

        ray = LineString(
            [
                self.center_pos,
                (
                    self.center_pos[0] + self.max_distance * np.cos(self.theta),
                    self.center_pos[1] + self.max_distance * np.sin(self.theta),
                ),
            ]
        )
        intersection = ray.intersection(world.boundary, grid_size=0.1)

        if intersection.is_empty:
            distance = self.max_distance
        else:
            # if multiple intersections
            # after test, shapely returns the closest one
            distance = np.random.normal(
                Point(*self.center_pos).distance(intersection), self.measurement_sigma
            )

        return distance

    def move(self, v: float, w: float, dt: float) -> None:
        """
        Move the robot with linear velocity v and angular velocity w
        for a small time dt. Update the robot's position and orientation.

        Args:
            v (float): Linear velocity.
            w (float): Angular velocity.
            dt (float): Time step.
        """
        x, y = self.center_pos
        if abs(v) > 1e-5:  # if v is small, don't add noise
            v = np.random.normal(v, self.v_sigma)

        if abs(w) < 1e-5:
            # if w is small, don't add noise
            new_x = x + v * np.cos(self.theta) * dt
            new_y = y + v * np.sin(self.theta) * dt
        else:
            w = np.random.normal(w, self.w_sigma)
            new_x = x + v / w * (np.sin(self.theta + w * dt) - np.sin(self.theta))
            new_y = y - v / w * (np.cos(self.theta + w * dt) - np.cos(self.theta))
            self.theta = (self.theta + w * dt) % (2 * np.pi)

        self.center_pos = [new_x, new_y]

    def update_artist(self, artists: list[PathPatch, Quiver]) -> None:
        """
        Update the artist (PathPatch and Quiver) to visualize the new position and
        orientation of the robot.
        """
        polygon_path_patch, arrow = artists
        polygon_path_patch.set_path(_path_from_polygon(self.get_shape()))
        x, y, dx, dy = self.get_direction()
        arrow.set_offsets([x, y])
        arrow.set_UVC(dx, dy)


class ParticleGroup:
    """
    ParticleGroup class representing a group of particles in a 2D space.

    Attributes:
        num_particles (int): Number of particles in the group.
        positions (np.ndarray): Array of particle positions.
        thetas (np.ndarray): Array of particle orientations.
        weights (np.ndarray): Array of particle weights.
        radius (float): Radius of the particles.
        max_distance (float): Maximum distance for particle's distance sensor.
        v_sigma (float): Standard deviation of linear velocity noise.
        w_sigma (float): Standard deviation of angular velocity noise.
        measurement_sigma (float): Standard deviation of measurement noise.
    """

    def __init__(
        self,
        positions: np.ndarray,
        thetas: np.ndarray,
        weights: np.ndarray,
        radius: RealNumber,
        max_distance: RealNumber,
        v_sigma: RealNumber = 0.1,
        w_sigma: RealNumber = 0.1,
        measurement_sigma: RealNumber = 1,
    ) -> None:
        self.num_particles = len(positions)
        self.positions = positions
        self.thetas = thetas
        self.weights = weights
        self.radius = radius
        self.max_distance = max_distance
        self.v_sigma = v_sigma
        self.w_sigma = w_sigma
        self.measurement_sigma = measurement_sigma
        self.world_segments = None

    def measure_distance(self, world: Polygon) -> np.ndarray:
        """
        Simulate rays from each particle's center in the direction of its
        orientation (theta) and measure the distance to the world boundary.

        To calculate the intersection of 2 lines, we express 2 lines as:
            p: p0 + t * r, where r = p1 - p0                     •q1
            q: q0 + u * s, where s = q1 - q0                    /
        and the intersection is given by:            p0•-------•----• p1
            t = (q0 - p0) x s / (r x s)                       /
            u = (q0 - p0) x r / (r x s)                      /
        where "x" is the cross product.                   q0•

        Only intersections where t in (0,1] and u in [0,1] are considered valid.

        Returns:
            distances (np.ndarray): Array of distances from each particle to the world boundary.
        """

        # cache the world segments
        if self.world_segments is None:
            edge_starts = []
            edge_ends = []
            ext_coords = list(world.exterior.coords)
            edge_starts.extend(ext_coords[:-1])
            edge_ends.extend(ext_coords[1:])

            for obstacle in world.interiors:
                obstacle_coords = list(obstacle.coords)
                edge_starts.extend(obstacle_coords[:-1])
                edge_ends.extend(obstacle_coords[1:])

            edge_starts = np.array(edge_starts)
            edge_ends = np.array(edge_ends)

            self.world_segments = {
                "start": edge_starts,
                "end": edge_ends,
            }

        ray = self.max_distance * np.stack(
            [np.cos(self.thetas), np.sin(self.thetas)], axis=1
        )

        edge_start = self.world_segments["start"]
        edge_end = self.world_segments["end"]

        p0 = self.positions[:, None, :]  # (n, 1, 2)
        q0 = edge_start[None, :, :]  # (1, m, 2)
        q1 = edge_end[None, :, :]  # (1, m, 2)

        # ray is equivalent to p1 - p0
        ray = ray[:, None, :]  # (n, 1, 2)
        segment = q1 - q0  # (1, m, 2)

        r_cross_s = np.cross(ray, segment)  # (n, m)
        q0_minus_p0 = q0 - p0  # (n, m, 2)
        parallel = np.isclose(r_cross_s, 0)
        
        t = np.cross(q0_minus_p0, segment) / (r_cross_s + 1e-5)  # (n, m)
        u = np.cross(q0_minus_p0, ray) / (r_cross_s + 1e-5)  # (n, m)

        valid = (t <= 1) & (t > 0) & (u <= 1) & (u >= 0) & ~parallel
        t = np.where(valid, t, 1)  # set t to max_distance if not intersecting

        # the minimum is the distance that sensor measures
        distances = t.min(axis=1) * self.max_distance

        return distances + np.random.normal(
            0, self.measurement_sigma, self.num_particles
        )

    def update_weights(self, weights: np.ndarray) -> None:
        """
        Update the weights of the particles.
        """
        self.weights = weights

    def resample(
        self,
        new_positions: np.ndarray,
        new_thetas: np.ndarray,
    ) -> None:
        """
        Resample the particles, updating their positions and orientations.
        """
        assert len(new_positions) == len(new_thetas), (
            "new_positions and new_thetas must have the same length"
        )
        self.positions = new_positions
        self.thetas = new_thetas

        # set the weights of the resampled particles to 1
        self.weights = np.ones(len(self.positions))

    def move(self, v: float, w: float, dt: float) -> None:
        """
        Move the particles with linear velocity v and angular velocity w
        for a small time dt. Update the particles' positions and orientations.
        Args:
            v (float): Linear velocity.
            w (float): Angular velocity.
            dt (float): Time step.
        """

        if abs(v) > 1e-5:  # if v is small, don't add noise
            v = np.random.normal(
                v, self.v_sigma, self.num_particles
            )  # shape: (num_particles,)

        if abs(w) < 1e-5:
            self.positions[:, 0] += v * np.cos(self.thetas) * dt
            self.positions[:, 1] += v * np.sin(self.thetas) * dt
        else:
            w = np.random.normal(
                w, self.w_sigma, self.num_particles
            )  # shape: (num_particles,)
            self.positions[:, 0] += (
                v / w * (np.sin(self.thetas + w * dt) - np.sin(self.thetas))
            )
            self.positions[:, 1] -= (
                v / w * (np.cos(self.thetas + w * dt) - np.cos(self.thetas))
            )
            self.thetas = (self.thetas + w * dt) % (2 * np.pi)

    def visualize(
        self, ax: Axes = None, color: str = "red"
    ) -> list[list[PathPatch], Quiver]:
        """
        Visualize the particles on a given axis.
        Returns:
            list[list[PathPatch], Quiver]: A list containing:
                - polygon_path_patch: A list of PathPatch object for each particle shape
                - arrows: Quiver object for the particle directions
        """
        if ax is None:
            fig, ax = plt.subplots()
            ax.set_aspect("equal")

        # create a polygon for each particle to set alpha
        polygon_path_patch = [
            plot_polygon(
                self.get_shape(i),
                ax=ax,
                color=color,
                add_points=False,
                alpha=self.weights[i],
                edgecolor="black",
                linewidth=0,
            )
            for i in range(self.num_particles)
        ]

        arrows = ax.quiver(
            *self.get_direction(),
            angles="xy",
            scale_units="xy",
            scale=1,
            width=0.003,
            color=color,
            alpha=self.weights,
        )

        return [polygon_path_patch, arrows]

    def get_shape(self, idx: int) -> Polygon:
        """
        Get the shape of the `idx-th` particle as a shapely Polygon.
        The shape is a point with a buffer of the particle's radius.
        """
        return Point(self.positions[idx, 0], self.positions[idx, 1]).buffer(self.radius)

    def get_direction(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Get the direction of the particles as a tuple of (x, y, dx, dy).
        Used for visualization.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: A tuple containing:
                - x: x-coordinates of the particles
                - y: y-coordinates of the particles
                - dx: x-components of the direction vectors
                - dy: y-components of the direction vectors
                - shape of x, y, dx, dy: (num_particles,)
        """
        dx = self.radius * np.cos(self.thetas)
        dy = self.radius * np.sin(self.thetas)
        return (
            self.positions[:, 0] + dx,
            self.positions[:, 1] + dy,
            4 * dx,
            4 * dy,
        )

    def update_artist(self, artists: list) -> None:
        """
        Update the artist (PathPatch and Quiver) to visualize the new positions and
        orientations of the particles.
        """
        polygon_path_patches, arrows = artists[:-1], artists[-1]
        polygon_path_patches: list[PathPatch]
        arrows: Quiver

        for i in range(self.num_particles):
            polygon_path_patches[i].set_path(_path_from_polygon(self.get_shape(i)))
            # polygon_path_patches[i].set_alpha(np.maximum(self.weights[i], 0.1))
            polygon_path_patches[i].set_alpha(self.weights[i])

        x, y, dx, dy = self.get_direction()
        arrows.set_offsets(np.column_stack([x, y]))
        arrows.set_UVC(dx, dy)
        # arrows.set_alpha(np.maximum(self.weights, 0.1))
        arrows.set_alpha(self.weights)
