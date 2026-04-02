"""
agent_process.py — Maintains a rolling history buffer of neighbor agent states
and formats them into the tensor DiffusionPlanner expects.

Uses agent_past_process() from the DiffusionPlanner repo directly so that
slot ordering, distance sorting, ped capping, and SE(2) transforms exactly
match the training pipeline.

DiffusionPlanner expects:
    neighbor_agents_past: (num_agents, 21, 11)
        - 21 = current + 20 past timesteps (2s at 10Hz)
        - 11 = [x, y, cos_h, sin_h, vx, vy, width, length, type_vehicle, type_ped, type_bike]
        - all ego-centric at the CURRENT ego pose

All AgentState positions/headings must be in standard frame (CARLA x, y, heading in radians)
before calling these functions.  agent_past_process handles the SE(2) transform to ego frame
internally via _global_state_se2_array_to_local — the same transform used by the lane pipeline.
"""

import numpy as np
from collections import deque
from typing import Dict, List, Optional

from nuplan.planning.training.preprocessing.utils.agents_preprocessing import AgentInternalIndex
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType

from diffusion_planner.data_process.agent_process import agent_past_process

from utils.coord_utils import (
    carla_transform_to_standard
)

# ---------------------------------------------------------------------------
# Constants matching DataProcessor, do not change

NUM_AGENTS     = 32
NUM_STATIC     = 5
NUM_PAST_STEPS = 21   # current frame + 20 history frames (2s @ 10Hz)
MAX_PED_BIKE   = 10

# ---------------------------------------------------------------------------
# Agent state snapshot

class AgentState:
    """Snapshot of a single actor at one timestep, in standard global frame."""
    __slots__ = ('actor_id', 'x', 'y', 'heading', 'vx', 'vy', 'width', 'length', 'agent_type')

    def __init__(self, actor_id, x, y, heading, vx, vy, width, length, agent_type='vehicle'):
        self.actor_id   = actor_id
        self.x          = x
        self.y          = y
        self.heading    = heading
        self.vx         = vx
        self.vy         = vy
        self.width      = width
        self.length     = length
        self.agent_type = agent_type  # 'vehicle', 'pedestrian', 'bicycle'


# ---------------------------------------------------------------------------
# History buffer

class AgentHistoryBuffer:
    """
    Maintains a rolling buffer of per-agent state histories.
    Each call to update() adds one timestep of observations.
    """

    def __init__(self, max_history: int = NUM_PAST_STEPS):
        self.max_history = max_history
        self._histories: Dict[int, deque] = {}

    def update(self, agent_states: List[AgentState]):
        for state in agent_states:
            if state.actor_id not in self._histories:
                self._histories[state.actor_id] = deque(maxlen=self.max_history)
            self._histories[state.actor_id].append(state)

    def get_histories(self) -> Dict[int, List[AgentState]]:
        return {actor_id: list(buf) for actor_id, buf in self._histories.items()}


# ---------------------------------------------------------------------------
# Internal helpers

def _agent_type_to_tracked_object_type(agent_type: str) -> TrackedObjectType:
    if agent_type == 'pedestrian':
        return TrackedObjectType.PEDESTRIAN
    elif agent_type == 'bicycle':
        return TrackedObjectType.BICYCLE
    else:
        return TrackedObjectType.VEHICLE


def _histories_to_array_list(histories: Dict[int, List[AgentState]]):
    """
    Convert history buffer into the list-of-arrays format agent_past_process() expects:
        List of length NUM_PAST_STEPS, each (num_agents, AgentInternalIndex.dim())
    Positions/headings are in standard world frame (not yet ego-centric).
    agent_past_process handles the SE(2) transform internally.
    """
    if not histories:
        dim = AgentInternalIndex.dim()
        return [np.zeros((0, dim), dtype=np.float64) for _ in range(NUM_PAST_STEPS)], []

    agent_ids  = list(histories.keys())
    num_agents = len(agent_ids)
    dim        = AgentInternalIndex.dim()

    array = np.zeros((NUM_PAST_STEPS, num_agents, dim), dtype=np.float64)
    types = []

    for col, actor_id in enumerate(agent_ids):
        states = histories[actor_id]
        n      = len(states)
        types.append(_agent_type_to_tracked_object_type(states[-1].agent_type))

        for step in range(NUM_PAST_STEPS):
            states_idx = n - (NUM_PAST_STEPS - step)
            s = states[max(0, states_idx)]

            array[step, col, AgentInternalIndex.track_token()] = float(actor_id)
            array[step, col, AgentInternalIndex.x()]           = s.x
            array[step, col, AgentInternalIndex.y()]           = s.y
            array[step, col, AgentInternalIndex.heading()]     = s.heading
            array[step, col, AgentInternalIndex.vx()]          = s.vx
            array[step, col, AgentInternalIndex.vy()]          = s.vy
            array[step, col, AgentInternalIndex.width()]       = s.width
            array[step, col, AgentInternalIndex.length()]      = s.length

    array_list = [array[t] for t in range(NUM_PAST_STEPS)]
    return array_list, types  # types = per-agent list at most-recent frame


def _parked_cars_to_rows(parked_cars, ego_x: float, ego_y: float, max_dist: float = 35.0):
    dim = AgentInternalIndex.dim()
    rows = []
    types = []

    for vehicle in parked_cars:
        loc = vehicle.get_location()
        rot = vehicle.get_transform().rotation
        
        # 1. Distance check
        dist = np.sqrt((loc.x - ego_x)**2 + (loc.y - ego_y)**2)
        if dist > max_dist:
            continue
            
        # 2. Use YOUR standard converter for 100% alignment
        # This handles the CARLA -> Standard (Right-Handed) conversion
        sx, sy, sh_rad = carla_transform_to_standard(loc.x, loc.y, rot.yaw)
        
        bb = vehicle.bounding_box
        row = np.zeros(dim, dtype=np.float64)
        row[AgentInternalIndex.track_token()] = float(vehicle.id)
        row[AgentInternalIndex.x()]           = sx
        row[AgentInternalIndex.y()]           = sy
        row[AgentInternalIndex.heading()]     = sh_rad
        row[AgentInternalIndex.vx()]          = 0.0
        row[AgentInternalIndex.vy()]          = 0.0
        # Extents in CARLA are half-lengths; model wants full width/length
        row[AgentInternalIndex.width()]       = float(bb.extent.y * 2) 
        row[AgentInternalIndex.length()]      = float(bb.extent.x * 2) 
        
        rows.append(row)
        types.append(TrackedObjectType.VEHICLE)

    if not rows:
        return np.zeros((0, dim), dtype=np.float64), []
    return np.stack(rows, axis=0), types


# ---------------------------------------------------------------------------
# Public API

# def build_neighbor_agents_past(
#     history_buffer: AgentHistoryBuffer,
#     anchor_ego_state: np.ndarray,
#     parked_cars=None,
#     num_agents: int = NUM_AGENTS,
# ) -> np.ndarray:
#     """
#     Build the neighbor_agents_past tensor expected by DiffusionPlanner.

#     Delegates to agent_past_process() so slot ordering, distance sorting,
#     pedestrian capping (MAX_PED_BIKE=10), and SE(2) ego-centric transforms
#     all match the training pipeline exactly.

#     :param history_buffer: AgentHistoryBuffer — must NOT contain parked car ids
#                            (pass parked_cars separately to avoid double-counting)
#     :param anchor_ego_state: (3,) [x, y, heading] in standard frame
#     :param parked_cars: list of CARLA vehicle actors for parked cars (optional)
#     :param num_agents: slots to fill (32)
#     :return: np.ndarray (num_agents, NUM_PAST_STEPS, 11)
#     """
#     histories = history_buffer.get_histories()

#     # --- Split ego vs neighbors ---
#     ego_hist = {}
#     neighbor_hists = {}

#     for aid, h in histories.items():
#         if aid == -1:
#             ego_hist[aid] = h
#         else:
#             neighbor_hists[aid] = h

#     # --- Build arrays separately ---
#     ego_array_list, _ = _histories_to_array_list(ego_hist)
#     neighbor_array_list, neighbor_types = _histories_to_array_list(neighbor_hists)

#     # Convert ego list [(1,8), (1,8), ...] → (T,8)
#     ego_np = np.stack([a.squeeze() for a in ego_array_list], axis=0)

#     # Append parked cars (stationary — same row across all timesteps)
#     parked_types = []
#     if parked_cars:
#         ego_x, ego_y = float(anchor_ego_state[0]), float(anchor_ego_state[1])
#         parked_rows, parked_types = _parked_cars_to_rows(parked_cars, ego_x, ego_y)
#         if parked_rows.shape[0] > 0:
#             for t_idx in range(NUM_PAST_STEPS):
#                 if neighbor_array_list[t_idx].shape[0] > 0:
#                     neighbor_array_list[t_idx] = np.vstack([neighbor_array_list[t_idx], parked_rows])
#                 else:
#                     neighbor_array_list[t_idx] = parked_rows.copy()


#     _, agents, _, _ = agent_past_process(
#         past_ego_states       = ego_np,
#         past_tracked_objects  = neighbor_array_list,
#         tracked_objects_types = [neighbor_types + parked_types] * NUM_PAST_STEPS,
#         num_agents            = num_agents,
#         static_objects        = np.zeros((0, 5), dtype=np.float64),
#         static_objects_types  = [],
#         num_static            = NUM_STATIC,
#         max_ped_bike          = MAX_PED_BIKE,
#         anchor_ego_state      = anchor_ego_state,
#     )

#     return agents  # (num_agents, NUM_PAST_STEPS, 11)


def build_neighbor_agents_past(
    history_buffer: AgentHistoryBuffer,
    anchor_ego_state: np.ndarray,
    parked_cars=None,
    num_agents: int = NUM_AGENTS,
) -> np.ndarray:
    histories = history_buffer.get_histories()

    # 1. Strictly separate Ego from Neighbors
    # Ensure -1 is ONLY the ego and is not duplicated
    neighbor_hists = {aid: h for aid, h in histories.items() if aid != -1}
    ego_hist = {aid: h for aid, h in histories.items() if aid == -1}

    # 2. Build Neighbor arrays
    neighbor_array_list, neighbor_types = _histories_to_array_list(neighbor_hists)
    
    # 3. Handle Parked Cars (Stationary)
    parked_types = []
    if parked_cars:
        ego_x, ego_y = float(anchor_ego_state[0]), float(anchor_ego_state[1])
        parked_rows, parked_types = _parked_cars_to_rows(parked_cars, ego_x, ego_y)
        
        if parked_rows.shape[0] > 0:
            for t_idx in range(NUM_PAST_STEPS):
                if neighbor_array_list[t_idx].shape[0] > 0:
                    neighbor_array_list[t_idx] = np.vstack([neighbor_array_list[t_idx], parked_rows])
                else:
                    neighbor_array_list[t_idx] = parked_rows.copy()

    # 4. Build Ego array (T, 8)
    ego_array_list, _ = _histories_to_array_list(ego_hist)
    # If ego history is missing (first frame), create a dummy at current pose
    if not ego_hist:
        ego_np = np.zeros((NUM_PAST_STEPS, 8))
        ego_np[:, 1:4] = anchor_ego_state # x, y, heading
    else:
        ego_np = np.stack([a.squeeze() for a in ego_array_list], axis=0)

    # 5. Correct the Types list
    # Combined types for the agents currently in the neighbor_array_list
    # 5. Correct the Types list
    all_types = neighbor_types + parked_types

    # --- CRITICAL FIX START ---
    # Get the actual number of agents currently in the array for the current frame
    # neighbor_array_list is a list of arrays (one per timestep)
    current_num_agents = neighbor_array_list[-1].shape[0] 

    # Safety: If all_types isn't a list or its length doesn't match the rows, 
    # the repo's internal indexing [i] will crash.
    if not isinstance(all_types, list):
        all_types = [all_types]
        
    if len(all_types) != current_num_agents:
        # If there's a mismatch, fill with VEHICLE as a fallback
        all_types = [TrackedObjectType.VEHICLE] * current_num_agents
    # --- CRITICAL FIX END ---

    # Call the repo function
    _, agents, _, _ = agent_past_process(
        past_ego_states       = ego_np,
        past_tracked_objects  = neighbor_array_list,
        tracked_objects_types = [all_types] * NUM_PAST_STEPS,
        num_agents            = num_agents,
        static_objects        = np.zeros((0, 5), dtype=np.float64),
        static_objects_types  = [],
        num_static            = NUM_STATIC,
        max_ped_bike          = MAX_PED_BIKE,
        anchor_ego_state      = anchor_ego_state,
    )

    return agents