# Monte Carlo Localization (MCL) Robot Simulation

A Python implementation of Monte Carlo Localization for robot navigation in 2D environments with obstacles. This project simulates a robot using particle filters to estimate its position and orientation while navigating through a known map.

## Features

- **Monte Carlo Localization Algorithm**: Particle filter-based localization with configurable parameters
- **Robot Simulation**: Realistic robot movement with noise modeling
- **Sensor Simulation**: Distance sensor with measurement noise
- **Efficient Visualization**: High-resolution animation of robot and particle states and movement in resonable time
- **Flexible World Maps**: JSON-based map definition with obstacles

## Project Structure

![Architecture](figs/architect.png)

```
├── simulator.py       # Main simulation engine, control and visualization logic
├── robot.py           # Robot and particle group implementations
├── mcl.py             # Monte Carlo Localization algorithm
├── worldmap.py        # World map handling
├── assets/            # Asset folder containing map definitions and related images
│   ├── map1.json
│   ├── map2.json
│   ├── map1_preview.png
│   └── map2_preview.png
└── README.md          # This file

```

## Quick Start

### Prerequisites

```bash
pip install numpy matplotlib shapely tqdm
```

### Basic Usage

```python
ffrom map import Map
from simulator import Simulator, ControlNode

    
map = Map.load_map_from_json("assets/map2.json")

simulator = Simulator(
    world_map=map,
    control_node=ControlNode(10, 2, 30),
    num_particles=5000,
    init_x=100,
    init_y=100,
    init_theta=0.5,
    robot_radius=3,
    sample_radius=1.5,
    sensor_max_distance=500,
    likelyhood_sigma=5,
    measurement_sigma=3,
    ema_alpha=0.66,
    v_sigma=0.1,
    w_sigma=0.05,
    resample_factor=0.6,
    resample_random_probability=0.2,
    fps=5,
    speedup=5,
    dpi=100,
    save_file_name="simulation.mp4",
    add_size_bar=False,
    random_seed=123
)


simulator.main_simulation(num_steps=1000)

```

### Map Creation

check `usage_example/map_creation.ipynb`

## Configuration Parameters

### Simulator Parameters

| Parameter                        | Description                                  | Default         |
|----------------------------------|----------------------------------------------|-----------------|
| `world_map`                      | World map object                             |                 |
| `control_node`                   | Motion control input                         |                 |
| `num_particles`                  | Number of particles in the filter            |                 |
| `init_x`, `init_y`, `init_theta` | Robot's initial position and orientation     |                 |
| `robot_radius`                   | Robot's physical radius                      |                 |
| `sample_radius`                  | Particle sampling radius                     |                 |
| `sensor_max_distance`            | Maximum sensor range                         | 100             |
| `likelyhood_sigma`               | Likelihood function standard deviation       | 10              |
| `measurement_sigma`              | Sensor measurement noise                     | 1               |
| `ema_alpha`                      | Exponential moving average smoothing factor  | 0.9             |
| `v_sigma`, `w_sigma`             | Motion model noise parameters                | 0.1             |
| `resample_factor`                | Resampling threshold (fraction of particles) | 0.5             |
| `resample_random_probability`    | Random resampling probability                | 0.1             |
| `fps`                            | Frames per second for simulation rendering   | 30              |
| `speedup`                        | Simulation speed multiplier                  | 1               |
| `dpi`                            | DPI for video rendering                      | 100             |
| `save_file_name`                 | Output video file name                       | "simulation.mp4"|
| `add_size_bar`                   | Whether to render a size scale bar           | False           |
| `random_seed`                    | Random seed for reproducibility              | 0               |


### Control Node Parameters

| Parameter       | Description                           | Default |
| --------------- | ------------------------------------- | ------- |
| `max_linear`    | Maximum linear velocity               | 10.0    |
| `max_angular`   | Maximum angular velocity              | 2.0     |
| `safe_distance` | Obstacle avoidance distance           | 30      |
| `turning_step`  | Steps to turn when avoiding obstacles | 1000    |


## Animation Output

The simulator generates:
- **MP4 Animation**: Real-time visualization of the MCL process

### Visualization Elements

![Screenshot of Produced Animation](figs/step_500.png)

- **Blue Circle**: Real robot position and orientation
- **Red/Gray Particles**: Particle filter estimates
  - Red: High-confidence particles (survived ≥10 resampling cycles)
  - Gray: Regular particles
  - Transparency: Particle weight/confidence
- **Gray Areas**: Free space.
- **White Areas**: Obstacle.




