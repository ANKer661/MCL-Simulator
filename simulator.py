from __future__ import annotations

import typing
from typing import TypeAlias

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
from shapely import Point
from tqdm import tqdm

from mcl import MCL
from robot import ParticleGroup, Robot

if typing.TYPE_CHECKING:
    from matplotlib.artist import Artist
    from matplotlib.figure import Figure

    from worldmap import WorldMap


RealNumber: TypeAlias = int | float


class Simulator:
    """
    A class to simulate a robot using Monte Carlo Localization (MCL) in a given world_map.

    The simulator manages:
        - world_map, robot and particles initialization
        - the robot's (particles') movement
        - noise in the robot's movement and sensor measurements
        - visualization of the MCL process
    """

    def __init__(
        self,
        world_map: WorldMap,
        control_node: ControlNode,
        num_particles: int,
        init_x: RealNumber,
        init_y: RealNumber,
        init_theta: float,
        robot_radius: RealNumber,
        sample_radius: RealNumber,
        sensor_max_distance: RealNumber = 100,
        likelyhood_sigma: RealNumber = 10,
        measurement_sigma: RealNumber = 1,
        ema_alpha: float = 0.9,
        v_sigma: RealNumber = 0.1,
        w_sigma: RealNumber = 0.1,
        resample_factor: float = 0.5,
        resample_random_probability: float = 0.1,
        fps: int = 30,
        speedup: int = 1,
        dpi: int = 100,
        save_file_name: str = "simulation.mp4",
        add_size_bar: bool = False,
        random_seed: int = 0,
    ) -> None:
        # set random seed
        np.random.seed(random_seed)

        self.world_map = world_map
        self.control_node = control_node
        self.num_particles = num_particles
        self.mcl_solver = MCL(
            num_particles,
            likelyhood_sigma=likelyhood_sigma,
            alpha=ema_alpha,
            random_probability=resample_random_probability,
        )
        self.measurement_sigma = measurement_sigma
        self.v_sigma = v_sigma
        self.w_sigma = w_sigma
        self.resample_threshold = resample_factor
        self.fps = fps
        self.dt = 1 / fps
        self.dpi = dpi
        self.speedup = speedup
        self.save_file_name = save_file_name
        self.all_artists = []
        self.add_size_bar = add_size_bar

        # set up real robot
        if self.world_map.world.contains(Point(init_x, init_y).buffer(robot_radius)):
            self.real_robot = Robot(
                x=init_x,
                y=init_y,
                theta=init_theta,
                radius=robot_radius,
                max_distance=sensor_max_distance,
                v_sigma=self.v_sigma,
                w_sigma=self.w_sigma,
                measurement_sigma=self.measurement_sigma,
            )
        else:
            raise ValueError(
                f"The initial position of the robot (x={init_x}, y={init_y}) is outside the world_map boundary."
            )
        # init measurement
        self.prev_distance = np.random.normal(
            self.real_robot.measure_distance(self.world_map.world), self.measurement_sigma
        )

        # set up particles
        self.particles = ParticleGroup(
            positions=np.array(self.world_map.sample_points(num_particles, sample_radius)),
            thetas=np.random.uniform(0, 2 * np.pi, num_particles),
            weights=np.ones(num_particles),
            radius=sample_radius,
            max_distance=sensor_max_distance,
            v_sigma=self.v_sigma,
            w_sigma=self.w_sigma,
            measurement_sigma=self.measurement_sigma,
        )

    def get_effective_sample_size(self) -> float:
        """
        Calculate the effective sample size (N_eff) of the particles.

        Definition:
            N_eff = 1 / sum(w_i^2)
            where w_i is the normalized weight of the i-th particle.
        """
        normalized_weights = self.particles.weights / np.sum(self.particles.weights)
        return 1 / np.sum(normalized_weights**2)

    def run_step(self, step: int, prev_distance: float) -> float:
        # ACT Model
        v, w = self.control_node.get_command(step, prev_distance)
        # v = 0
        # w = 1
        # move real robot
        self.real_robot.move(v, w, self.dt)
        # move particles
        self.particles.move(v, w, self.dt)

        # SEE Model
        # measure distance
        real_distance = self.real_robot.measure_distance(self.world_map.world)

        distances = self.particles.measure_distance(self.world_map.world)

        # update weights
        weights = self.mcl_solver.update_weights(
            real_distance,
            distances,
            prev_weights=self.particles.weights,
        )
        self.particles.update_weights(weights)

        # resample particles
        N_eff = self.get_effective_sample_size()
        if N_eff < self.resample_threshold * self.num_particles:
            # resample particles
            new_positions, new_thetas, new_resample_count = self.mcl_solver.resample(
                positions=self.particles.positions,
                thetas=self.particles.thetas,
                radius=self.real_robot.radius,
                world_map=self.world_map,
                weights=self.particles.weights,
                resample_count=self.particles.resample_count,
            )

            self.particles.resample(new_positions, new_thetas, new_resample_count)

        return real_distance

    def main_simulation(self, num_steps: int) -> None:
        # progress bar
        self.progress_bar = tqdm(total=num_steps, desc="Simulating: ", ncols=80)

        def update_progress_bar(n: int, total: int) -> None:
            self.progress_bar.update(1)
            if n == total - 1:
                self.progress_bar.close()
                print("Simulation finished. Saving animation...")

        # set up the figure
        fig = self.init_ani()

        target_fps = self.fps * self.speedup
        self.ani = animation.FuncAnimation(
            fig,
            self.update_frame,
            frames=num_steps,
            interval=1000 / target_fps,
            blit=True,
        )

        self.ani.save(
            self.save_file_name,
            dpi=self.dpi,
            writer=animation.FFMpegWriter(fps=target_fps),
            progress_callback=update_progress_bar,
        )

        np.save(
            f"high_prob_variance_{self.measurement_sigma}.npy",
            np.array(self.particles.high_prob_variance),
        )

    def init_ani(self) -> Figure:
        """
        Initialize the animation by creating a figure and initializing the artists.
        """

        x_min, y_min, x_max, y_max = self.world_map.get_bounds()
        x_padding = 0.05 * (x_max - x_min)
        y_padding = 0.1 * (y_max - y_min)
        ratio = (x_max - x_min) / (y_max - y_min)
        fig, ax = plt.subplots(figsize=(4.8 * ratio, 4.8), dpi=self.dpi)
        ax.axis("off")
        ax.set_aspect("equal")
        ax.set_xlim(x_min - x_padding, x_max + x_padding)
        ax.set_ylim(y_min - y_padding, y_max + y_padding)
        fig.tight_layout()

        # draw world_map once
        self.world_map.visualize(ax)

        # add a size bar
        if self.add_size_bar: 
            size_bar_length = (x_max - x_min) / 10  # 10% of the width
            scalebar = AnchoredSizeBar(
                ax.transData,
                size=size_bar_length,
                label=str(size_bar_length),
                loc="lower right",
                pad=0.1,
                color="black",
                frameon=False,
                borderpad=0.5,
                sep=5,
                fontproperties={"size": 12},
            )
            ax.add_artist(scalebar)

        # show steps
        step_text = ax.text(
            0.05, 0.95, "Step: 0", fontsize=12, color="black", transform=ax.transAxes
        )

        # add particles' artists: positions and directions
        samples_patch, samples_arrows = self.particles.visualize(ax, color="red")
        self.all_artists.extend(samples_patch)
        self.all_artists.append(samples_arrows)

        # add real robot's artists: position and direction
        robot_patch, robot_arrow = self.real_robot.visualize(
            ax, alpha=0.6, color="blue"
        )
        self.all_artists.append(robot_patch)
        self.all_artists.append(robot_arrow)

        # add text artist
        self.all_artists.append(step_text)

        self.fig = fig

        return fig

    def update_frame(self, frame: int) -> list[Artist]:
        cur_distance = self.run_step(frame, self.prev_distance)

        # update particles
        self.particles.update_artist(self.all_artists[:-3])
        # update real robot
        # make sure the real robot is on top
        self.real_robot.update_artist(self.all_artists[-3:-1])
        # update step text
        self.all_artists[-1].set_text(f"Step: {frame + 1}")

        self.prev_distance = cur_distance

        # save for presentation
        # if frame == 499 or frame == 999:
        #     self.fig.savefig(
        #         "step_" + str(frame + 1) + ".png",
        #         dpi=self.dpi,
        #         bbox_inches="tight",
        #     )

        return self.all_artists


class ControlNode:
    def __init__(
        self, max_linear=10.0, max_angular=2, safe_distance=30, turning_step=1000
    ):
        self.max_linear = max_linear
        self.max_angular = max_angular
        self.safe_distance = safe_distance
        self.turning_step = turning_step

    def get_command(self, step: int, front_dist: float) -> tuple[float, float]:
        """
        Given a front distance, return linear and angular velocity.
        """
        v = self.max_linear
        omega = 0.0

        if front_dist < self.safe_distance:
            v = 0.0
            omega = self.max_angular if step < self.turning_step else -self.max_angular

        return v, omega
