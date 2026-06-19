"""heatmap_agent.py

SimLingoAdapter variant that supplies a privileged, ground-truth collision-risk
heatmap as a SECOND model view at inference time -- the topdown collision
heatmap (topdown camera + colorized green->red risk overlay) produced at
training time by collect_data_topdown.py / crossing_gmm.py and saved to
topdown_heatmap/. This lets the benchmark evaluate the model with perfect
pedestrian-intent awareness in its visual input, instead of requiring the model
to infer risk purely from RGB.

Two-view (not stitched): the overlay is injected via the `_augment_input_data`
hook in agent_interface.run_step_testbed as input_data['topdown_0']; the front
camera (rgb_0) is left untouched. agent_simlingo.tick() turns the topdown into
one extra ViT tile when the checkpoint's cfg has data_module.base_dataset.use_topdown.

Usage (see benchmark.py --use-heatmap-input flag):
    adapter = HeatmapSimLingoAdapter('localhost', 2000)
    adapter.init_testbed(...)
    adapter.parking_scenario = parking_scenario   # gives access to live walker GMM state
"""
import os
import sys

import numpy as np
import cv2

_COLLISION_PROBS = os.path.dirname(os.path.abspath(__file__))
_VLA_ADAPTER = os.path.dirname(_COLLISION_PROBS)
for _p in (_COLLISION_PROBS, _VLA_ADAPTER):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from agent_interface import SimLingoAdapter
from crossing_gmm import render_route_risk_grid, heatmap_on_topdown
from config import HORIZON_SCALE as HEATMAP_HORIZON, ROUTE_CORRIDOR_HALF_WIDTH, TOPDOWN_SIZE
# Reuse the data-collection helpers as-is instead of re-deriving them: same
# camera setup, same grid geometry, same live obstacle-pose pairing logic.
from collect_data_topdown import setup_topdown_camera, _actor_live_poses, TOPDOWN_HALF, TOPDOWN_RES


class HeatmapSimLingoAdapter(SimLingoAdapter):
    """SimLingoAdapter that feeds the model the topdown collision heatmap as a
    separate second view (input_data['topdown_0']), not stitched into the camera.

    `parking_scenario` must be set (by the benchmark runner) right after
    init_testbed so live obstacle GMM state (pedestrians and vehicles) can be
    read each tick -- mirrors collect_data_topdown.py's gmm.json + _actor_live_poses,
    but built live from the running scenario instead of from saved measurements.
    """

    def init_testbed(self, *args, **kwargs):
        super().init_testbed(*args, **kwargs)
        self.parking_scenario = None
        self._latest_topdown_frame = None
        self._obstacles_gmm_base = None
        self._topdown_cam = setup_topdown_camera(self.world, self.hero_actor)
        self._topdown_cam.listen(lambda img: setattr(self, '_latest_topdown_frame', img))
        self._model_input_writer = None
        # Optional per-frame capture of the exact model input (set by test_model_input.py).
        self._capture_dir = None

    def start_model_input_recording(self, path, fps=20):
        """Opt-in: write every topdown collision-heatmap view the model sees to an
        mp4 -- the regular chase/topdown recordings don't include the heatmap.
        Writer is opened lazily once the first frame's size is known."""
        self._model_input_video_path = path
        self._model_input_fps = fps

    def destroy_cam(self):
        super().destroy_cam()
        if getattr(self, '_model_input_writer', None) is not None:
            self._model_input_writer.release()
            self._model_input_writer = None
        if getattr(self, '_topdown_cam', None) is not None and self._topdown_cam.is_alive:
            self._topdown_cam.destroy()

    def _obstacles_gmm_live(self):
        """Static gmm_candidate_futures (per sub-scenario), refreshed with live
        actor position via collect_data_topdown._actor_live_poses -- reused
        as-is, same pairing logic used at data-collection time. Handles both
        pedestrians and vehicles (the helper matches positionally, not by type).

        Building the static base list inline (rather than reusing
        save_gmm_label) is unavoidable: that helper writes gmm.json to disk,
        but here there's no saved file to read from at inference time.
        """
        scenario = self.parking_scenario
        if scenario is None:
            return []
        if self._obstacles_gmm_base is None:
            base = []
            for sub in getattr(scenario, 'list_scenarios', []):
                base.extend(getattr(sub, 'gmm_candidate_futures', []))
            self._obstacles_gmm_base = base
        return _actor_live_poses(scenario, self._obstacles_gmm_base)

    def _render_risk(self):
        """Corridor-clipped risk grid via the shared render_route_risk_grid (same
        function used by collect_data_topdown._compute_heatmaps, so the offline
        and live heatmaps are identical). The corridor uses the model's own latest
        predicted route (one tick stale) since there's no recorded A* label at
        inference time; route[0] is prepended as the ego origin."""
        tf = self.hero_actor.get_transform()
        ego_xy = np.array([tf.location.x, tf.location.y])
        yaw = np.deg2rad(tf.rotation.yaw)

        route = getattr(self, 'latest_pred_route', None)
        route_pts = ([[0.0, 0.0]] + np.asarray(route).tolist()) if route is not None else None
        return render_route_risk_grid(
            self._obstacles_gmm_live(), ego_xy, yaw, route_pts,
            TOPDOWN_SIZE, TOPDOWN_HALF, ROUTE_CORRIDOR_HALF_WIDTH,
            resolution=TOPDOWN_RES, horizon_scale=HEATMAP_HORIZON)

    def _augment_input_data(self, input_data):
        """Inject the topdown collision heatmap as a separate model view.

        Renders the corridor-clipped risk grid and overlays it on the live
        topdown camera frame (heatmap_on_topdown, the exact RGB written to
        topdown_heatmap/ at collection time), then hands it to the model as
        input_data['topdown_0'] (RGB HWC uint8). agent_simlingo.tick() turns it
        into one extra ViT tile when the checkpoint enables use_topdown. The
        front camera (rgb_0) is left untouched -- no stitching."""
        if self._latest_topdown_frame is None:
            return   # topdown camera hasn't produced a frame yet

        td = self._latest_topdown_frame
        td_bgra = np.frombuffer(td.raw_data, dtype=np.uint8).reshape((td.height, td.width, 4))
        topdown_rgb = td_bgra[:, :, :3][:, :, ::-1]

        risk = self._render_risk()
        overlay_rgb = np.ascontiguousarray(heatmap_on_topdown(topdown_rgb, risk))
        input_data['topdown_0'] = (None, overlay_rgb)

        if getattr(self, '_model_input_video_path', None):
            overlay_bgr = overlay_rgb[:, :, ::-1]
            if self._model_input_writer is None:
                h, w = overlay_bgr.shape[:2]
                self._model_input_writer = cv2.VideoWriter(
                    self._model_input_video_path, cv2.VideoWriter_fourcc(*'mp4v'),
                    self._model_input_fps, (w, h))
            self._model_input_writer.write(np.ascontiguousarray(overlay_bgr))

        if self._capture_dir is not None:
            self._capture_model_input(overlay_rgb)

    # ------------------------------------------------------------------
    # Test/inspection: dump the EXACT topdown view + the InternVL2 tile the
    # vision encoder actually receives for that view, per frame.
    # ------------------------------------------------------------------
    def start_model_input_capture(self, out_dir):
        self._capture_dir = out_dir
        self._capture_idx = 0
        os.makedirs(out_dir, exist_ok=True)

    def _capture_model_input(self, overlay_rgb):
        n, size = save_model_input_frame(
            self._capture_dir, self._capture_idx, overlay_rgb,
            use_global_img=self.cfg.model.vision_model.use_global_img)
        if self._capture_idx == 0:
            print(f"[capture] topdown view -> {n} tile/frame of {size} -> {self._capture_dir}")
        self._capture_idx += 1


def save_model_input_frame(out_dir, idx, topdown_rgb, use_global_img=False):
    """Save the EXACT topdown view for one frame plus the InternVL2 tile the
    vision encoder receives for it.

    topdown_rgb: the topdown collision-heatmap overlay (RGB uint8), as written to
    topdown_heatmap/ and supplied to the model as the second view. Tiling exactly
    mirrors agent_simlingo.tick()'s topdown path: dynamic_preprocess(max_num=1),
    no cut_bottom_quarter. Shared by the live capture (HeatmapSimLingoAdapter) and
    the no-model A* path (test_model_input.py) so both produce identical artifacts.
    """
    from PIL import Image
    from simlingo_training.utils.internvl2_utils import dynamic_preprocess

    os.makedirs(out_dir, exist_ok=True)
    Image.fromarray(np.ascontiguousarray(topdown_rgb)).save(
        os.path.join(out_dir, f'{idx:04d}_topdown.jpg'), quality=95)

    tiles = dynamic_preprocess(Image.fromarray(np.ascontiguousarray(topdown_rgb)),
                               image_size=448, use_thumbnail=use_global_img, max_num=1)
    for t, tile in enumerate(tiles):
        tile.save(os.path.join(out_dir, f'{idx:04d}_tile{t}.jpg'), quality=95)
    return len(tiles), tiles[0].size
