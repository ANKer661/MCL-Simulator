from __future__ import annotations

import typing
from typing import TypeAlias
import numpy as np

if typing.TYPE_CHECKING:
    from worldmap import WorldMap

RealNumber: TypeAlias = int | float


class MCL:
    """
    A class to implement 2 key components of the Monte Carlo Localization (MCL) algorithm:
        1. Weight update: Update the weights of the particles based on the sensor measurements.
        2. Resampling: Resample the particles based on their weights.
    
    Refer to the method `update_weights` and `resample` for the details of the algorithms.

    Attributes:
        num_particles (int): The number of particles to use in the MCL algorithm.
        alpha (float): The exponential moving average (EMA) factor for the weights.
            - w_new = (1 - alpha) * w_new + alpha * w_old
            - Default is 0.0 (no EMA).
        sigma (RealNumber): The standard deviation of the Gaussian distribution used to update the weights.
        random_probability (float): The probability of resampling randomly.
            - Default is 0.1 (10% probability).
    """
    def __init__(
        self,
        num_particles: int,
        alpha: float = 0.0,
        likelyhood_sigma: RealNumber = 10,
        random_probability: float = 0.1,
    ) -> None:
        self.num_particles = num_particles
        self.alpha = alpha
        self.sigma = likelyhood_sigma
        self.random_probability = random_probability

    def update_weights(
        self,
        real_distance: np.ndarray,
        distances: np.ndarray,
        prev_weights: np.ndarray,
    ) -> np.ndarray:
        """
        Update the weights of the particles based on the sensor measurements and the world_map.
        The weights are updated using a Gaussian distribution centered around the real distance.

        Formula:
            wi = (1 - alpha) * exp(-(di - d_real)^2 / (2 * sigma^2)) + alpha * wi_prev
        """
        error = np.array(distances) - real_distance
        new_weights = (1 - self.alpha) * (
            np.exp(-error**2 / (2 * self.sigma**2))
        ) + self.alpha * prev_weights

        return new_weights

    def resample(
        self,
        positions: np.ndarray,
        thetas: np.ndarray,
        radius: RealNumber,
        world_map: WorldMap,
        weights: np.ndarray,
        resample_count: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Resample the particles by Low Variance Resampling + Random Resampling.

        The algorithm works as follows:
            Low Variance Resampling (LVR, Prob = 1 - p):

                1. Normalize the weights so that they sum to 1.
                2. Compute the CDF of the weights.
                3. Sample i/N + s for i in range(N), where s = Uniform(0, 1/N).
                4. Use the CDF to find the indices of the particles to resample.
            Random Resampling (Prob = p):
                Resample the particles uniformly in the world_map.


        Args:
            positions (np.ndarray): The positions of the particles.
            thetas (np.ndarray): The thetas of the particles.
            radius (RealNumber): The radius of the robot.
            world_map (Map): The world_map object, used to sample random positions.
            weights (np.ndarray): The weights of the particles.
            resample_count (np.ndarray): The resample survival counts for each particle.

        Returns:
            new_positions (np.ndarray): The positions of the particles after resampling.
            new_thetas (np.ndarray): The thetas of the particles after resampling.
            new_resample_count (np.ndarray): The resample survival counts after resampling.
        """
        # Random Resampling (p)
        random_mask = np.random.uniform(0, 1, self.num_particles) < self.random_probability

        random_positions = world_map.sample_points(n=np.sum(random_mask), robot_radius=radius)
        random_thetas = np.random.uniform(0, 2 * np.pi, np.sum(random_mask))

        # Low Variance Resampling (1 - p)
        # normalize weights

        normalized_weights = weights / np.sum(weights)
        cdf = np.cumsum(normalized_weights)

        # resample particles
        start_point = np.random.uniform(0, 1 / self.num_particles, 1)
        selected_points = (
            np.arange(self.num_particles) / self.num_particles + start_point
        )

        selected_idx = np.searchsorted(cdf, selected_points, side="right")
        selected_idx = selected_idx[~random_mask]

        new_positions = np.concatenate(
            [
                positions[selected_idx],
                random_positions,
            ]
        )

        new_thetas = np.concatenate(
            [
                thetas[selected_idx],
                random_thetas,
            ]
        )

        new_resample_count = np.concatenate(
            [
                resample_count[selected_idx] + 1,
                np.zeros(np.sum(random_mask)),
            ]
        )

        return new_positions, new_thetas, new_resample_count
