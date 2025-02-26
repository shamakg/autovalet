import importlib.util
import sys
import numpy as np

sys.path.extend(["bosch", "bosch/occ3d/models", "bosch/occ3d/models/occnet"])
from bosch.occ3d.utils import load_json
from bosch.occ3d.models.occnet_nrcs.infer import InferModel, InferSample

config_path = "bosch/config.py"
calibration = load_json("./bosch/occ3d/config/calibration/calibration_carla.json")
label_config = load_json("./bosch/occ3d/config/label.json")
id2label = {int(id): label for id, label in label_config["id2label"].items()}
id2rgb = {int(id): np.array(color) / 255.0 for id, color in label_config["id2rgb"].items()}
spec = importlib.util.spec_from_file_location("config", config_path)
config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config)
class_remapping = config.class_remapping
merge_map = np.arange(max(id2label.keys()) + 1).astype(np.uint8)
for key, value in config.class_remapping.items():
    merge_map[key] = value
try:
    adaptive_radius = float(config.adaptive_radius)
except:
    adaptive_radius = None

model = InferModel(
    model_dir="bosch/" + config.model,  # Change this if the model is in another dir.
    adaptive_range_xyz_min_max=config.adaptive_bbox,
    adaptive_radius=adaptive_radius,
    fine_skip_labels=[0, 1],
    logger=None,
)
scale = model.model_config['model']['pts_bbox_head']['occupancy_size'][0] * model.model_config['model']['pts_bbox_head']['cascade_ratio']
bbox = np.array(model.model_config['model']['pts_bbox_head']['point_cloud_range'])
shift = bbox[:3]
device = next(iter(model.model.parameters())).device
ctr = 0
prev_sample = None

def filter_occ(occ, threshold=10):
    # if True:
    #     return occ
    # Class merging
    occ[:, 3] = merge_map[occ[:, 3]]
    occ = occ[occ[:, 3] != 0]

    # Low-confidence filtering with threshold
    counts = np.bincount(occ[:, 3].flatten())
    # counts[6] = counts[6] if counts[6] >= 30 else 0
    valid_elements = np.nonzero(counts >= threshold)[0]
    mask = np.isin(occ[:, 3], valid_elements)
    occ = occ[mask]

    return occ

def run_perception_model(x, y, yaw, imgnps):
    global prev_sample, ctr
    # imgnps = [np.transpose(img, (1, 0, 2)) for img in imgnps]

    # img_width = 808
    # img_height = 640
    # imgnps = np.zeros((4, img_width, img_height, 3))  # 4 camera angles, see extrinsics.
    # for i, img in enumerate(imgs):
    #     imgnps[i] = img
    # re_imgs = [cv2.resize(img, (img_width // 8, img_height // 8)) for img in imgnps]

    # get_can_bus()
    can_bus = np.zeros(18)
    can_bus[0] = x #1.0  # odom.pose.pose.position.x
    can_bus[1] = y #0.0  # odom.pose.pose.position.y
    can_bus[2] = 0 # TODO: 0 ok?

    # quat = [
    #     1.0,
    #     0.0,
    #     0.0,
    #     0.0,
    # ]  # [odom.pose.pose.orientation.x, odom.pose.pose.orientation.y, odom.pose.pose.orientation.z, odom.pose.pose.orientation.w]

    # rotation = R.from_quat(quat)
    # can_bus[-1] = rotation.as_euler("zyx", degrees=True)[0]
    can_bus[-1] = np.rad2deg(yaw)

    # can_bus[0] = 295.4773254394531
    # can_bus[1] = 199.1146240234375
    # can_bus[2] = 0.0
    # can_bus[-1] = -118.35323369514315

    # uncomment in experiment
    # imgnps[1] = cv2.flip(imgnps[1], 0)
    # imgnps[1] = cv2.flip(imgnps[1], 1)

    # for i in range(4):
    #     img = imgnps[i]
    #     h, w = img.shape[:2]
    #     center = (w//2, h//2)
    #     mask = np.zeros((h, w), dtype=np.uint8)
    #     cv2.ellipse(mask, center, (924//2, 732//2), 0, 0, 360, 255, -1)
    #     imgnps[i] = cv2.bitwise_and(img, img, mask=mask)

    # for i, img in enumerate(imgnps):
    #     c_x = img.shape[1] // 2
    #     c_y = img.shape[0] // 2
    #     w = 808
    #     h = 640
    #     imgnps[i] = img[c_y - h // 2: c_y + h // 2, c_x - w // 2: c_x + w // 2]

    # for i, img in enumerate(imgnps):
    #     np.save('test-inputs/mine/test-{}.npy'.format(i), img)
    # re_imgs = [cv2.resize(img, (808 // 4, 640 // 4)) for img in imgnps]

    sample = InferSample(
        imgnps=imgnps,
        calibration=calibration,
        model_config=model.model_config,
        scene_token=None,
        sample_id=ctr,
        can_bus=can_bus,
        device=device,
    )

    if prev_sample is None:
        prev_sample = sample
        ctr += 1
        return

    result = model.run_sample(
        sample=sample, prev_samples=[prev_sample], range_xyz_min_max=config.bbox
    )

    prev_sample = sample
    ctr += 1

    occ = filter_occ(result['occ_coarse']).astype(np.float32)
    occ[:, :3] = occ[:, :3] * scale + shift
    return occ