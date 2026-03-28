import numpy as np


def carla_transform_to_standard(x, y, heading_deg):
    return x, y, np.deg2rad(heading_deg)

def carla_velocity_to_standard(vx, vy):
    return vx, vy

def standard_to_carla(x, y, heading_rad):
    return x, y, np.rad2deg(heading_rad)

