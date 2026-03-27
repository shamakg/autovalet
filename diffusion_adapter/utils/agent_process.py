"""
agent_process.py — Maintains a rolling history buffer of neighbor agent states
and formats them into the tensor DiffusionPlanner expects.

Since nuPlan is installed, we call agent_past_process() directly from the repo
rather than reimplementing it.

DiffusionPlanner expects:
    neighbor_agents_past: (num_agents, 21, 11)
        - 21 = current + 20 past timesteps (2s at 10Hz)
        - 11 = [x, y, cos_h, sin_h, vx, vy, width, length, type_vehicle, type_ped, type_bike]
        - all ego-centric at the CURRENT ego pose

All inputs must be in standard right-handed frame (already converted from CARLA
via coord_utils) before calling these functions.
"""

import numpy as np
from collections import deque
from typing import Dict, List

from nuplan.planning.training.preprocessing.utils.agents_preprocessing import AgentInternalIndex
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType

from diffusion_planner.data_process.agent_process import agent_past_process

# ---------------------------------------------------------------------------
# Constants matching DataProcessor
# ---------------------------------------------------------------------------
NUM_AGENTS     = 32
NUM_STATIC     = 5
NUM_PAST_STEPS = 21   # current frame + 20 history frames (2s @ 10Hz)
MAX_PED_BIKE   = 10


# ---------------------------------------------------------------------------
# Agent state snapshot
# ---------------------------------------------------------------------------

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
# ---------------------------------------------------------------------------

class AgentHistoryBuffer:
    """
    Maintains a rolling buffer of per-agent state histories.
    Each call to update() adds one timestep of observations.
    """

    def __init__(self, max_history: int = NUM_PAST_STEPS):
        self.max_history = max_history
        # agent_id -> deque of AgentState (oldest first, newest last)
        self._histories: Dict[int, deque] = {}

    def update(self, agent_states: List[AgentState]):
        """
        Add a new frame of agent observations.

        :param agent_states: List of AgentState for all visible agents this tick.
        """
        for state in agent_states:
            aid = state.actor_id
            if aid not in self._histories:
                self._histories[aid] = deque(maxlen=self.max_history)
            self._histories[aid].append(state)

    def get_histories(self) -> Dict[int, List[AgentState]]:
        return {aid: list(buf) for aid, buf in self._histories.items()}


# ---------------------------------------------------------------------------
# Build agent array in AgentInternalIndex format
# ---------------------------------------------------------------------------

def _agent_type_to_tracked_object_type(agent_type: str) -> TrackedObjectType:
    if agent_type == 'pedestrian':
        return TrackedObjectType.PEDESTRIAN
    elif agent_type == 'bicycle':
        return TrackedObjectType.BICYCLE
    else:
        return TrackedObjectType.VEHICLE


def _histories_to_array_list(histories: Dict[int, List[AgentState]]):
    """
    Convert history buffer into the list-of-arrays format that
    agent_past_process() expects:
        List of length NUM_PAST_STEPS, each array (num_agents, AgentInternalIndex.dim())

    Agents with shorter histories are padded by repeating their oldest state.
    """
    if not histories:
        # Return empty list of empty arrays
        dim = AgentInternalIndex.dim()
        return [np.zeros((0, dim), dtype=np.float64) for _ in range(NUM_PAST_STEPS)], []

    agent_ids  = list(histories.keys())
    num_agents = len(agent_ids)
    dim        = AgentInternalIndex.dim()

    # Build (NUM_PAST_STEPS, num_agents, dim) array
    array = np.zeros((NUM_PAST_STEPS, num_agents, dim), dtype=np.float64)
    types = []  # tracked object type per agent (from most recent state)

    for col, aid in enumerate(agent_ids):
        states = histories[aid]
        n      = len(states)
        types.append(_agent_type_to_tracked_object_type(states[-1].agent_type))

        for step in range(NUM_PAST_STEPS):
            # Map step index to states list (step 0 = oldest, step -1 = newest)
            states_idx = n - (NUM_PAST_STEPS - step)
            s = states[max(0, states_idx)]  # clamp — pad with oldest if not enough history

            array[step, col, AgentInternalIndex.track_token()] = float(aid)
            array[step, col, AgentInternalIndex.x()]           = s.x
            array[step, col, AgentInternalIndex.y()]           = s.y
            array[step, col, AgentInternalIndex.heading()]     = s.heading
            array[step, col, AgentInternalIndex.vx()]          = s.vx
            array[step, col, AgentInternalIndex.vy()]          = s.vy
            array[step, col, AgentInternalIndex.width()]       = s.width
            array[step, col, AgentInternalIndex.length()]      = s.length

        if col == 0:
            print(f"Agent {aid} last step raw array: {array[-1, col, :]}")
            print(f"  x at idx {AgentInternalIndex.x()}: {array[-1, col, AgentInternalIndex.x()]} (expected {states[-1].x})")
            print(f"  y at idx {AgentInternalIndex.y()}: {array[-1, col, AgentInternalIndex.y()]} (expected {states[-1].y})")
            print(f"  vx at idx {AgentInternalIndex.vx()}: {array[-1, col, AgentInternalIndex.vx()]} (expected {states[-1].vx})")

    # Convert to list of per-timestep arrays
    array_list = [array[t] for t in range(NUM_PAST_STEPS)]
    types_list = [types] * NUM_PAST_STEPS  # same types at every timestep

    return array_list, types_list[-1]  # agent_past_process wants types at last frame


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_neighbor_agents_past(
    history_buffer: AgentHistoryBuffer,
    anchor_ego_state: np.ndarray,
    num_agents: int = NUM_AGENTS,
) -> np.ndarray:
    """
    Build the neighbor_agents_past tensor expected by DiffusionPlanner.

    :param history_buffer: AgentHistoryBuffer with recent observations
    :param anchor_ego_state: np.ndarray (3,) — current ego [x, y, heading] in standard frame
    :param num_agents: max agents to include
    :return: np.ndarray (num_agents, NUM_PAST_STEPS, 11)
    """
    histories = history_buffer.get_histories()
    array_list, agent_types = _histories_to_array_list(histories)

    # agent_past_process expects ego_agent_past=None at inference
    _, agents, _, static_objects = agent_past_process(
        past_ego_states        = None,
        past_tracked_objects   = array_list,
        tracked_objects_types  = [agent_types] * NUM_PAST_STEPS,
        num_agents             = num_agents,
        static_objects         = np.zeros((0, 5), dtype=np.float64),
        static_objects_types   = [],
        num_static             = NUM_STATIC,
        max_ped_bike           = MAX_PED_BIKE,
        anchor_ego_state       = anchor_ego_state,
    )

    return agents  # (num_agents, NUM_PAST_STEPS, 11)


def build_static_objects(num_static: int = NUM_STATIC) -> np.ndarray:
    """
    Returns zeroed static objects tensor.
    CARLA doesn't expose cones/barriers easily so we leave this empty for now.

    :return: np.ndarray (num_static, 10)
    """
    return np.zeros((num_static, 10), dtype=np.float32)