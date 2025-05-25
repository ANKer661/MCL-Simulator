from worldmap import WorldMap
from simulator import Simulator, ControlNode


world_map = WorldMap.load_map_from_json("assets/map1.json")

simulator = Simulator(
    world_map=world_map,
    control_node=ControlNode(
        max_linear=10, max_angular=2, safe_distance=30, turning_step=800
    ),
    num_particles=5000,
    init_x=30,
    init_y=10,
    init_theta=0.1,
    robot_radius=3,
    sample_radius=1.5,
    sensor_max_distance=500,
    likelyhood_sigma=3,
    measurement_sigma=30,
    ema_alpha=0.66,
    v_sigma=0.1,
    w_sigma=0.05,
    resample_factor=0.5,
    resample_random_probability=0.2,
    fps=5,
    speedup=6,
    dpi=300,
    save_file_name="noise_30.mp4",
    add_size_bar=True,
    random_seed=0,
)

simulator.main_simulation(num_steps=1600)
