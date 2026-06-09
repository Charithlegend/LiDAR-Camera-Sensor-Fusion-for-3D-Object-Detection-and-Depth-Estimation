"""LiDAR-Camera Sensor Fusion for 3D Object Detection & Depth Estimation
=======================================================================
Combines YOLO instance segmentation with Velodyne LiDAR point cloud data
to estimate bumper depth for detected cars across KITTI dataset frames.
Evaluates inlier vs bleed-out points and visualises 3D bounding boxes
projected onto the image alongside colour-coded LiDAR points.

Usage:
    python sensor_fusion.py --base ./KITTI --model yolov8n-seg.pt
    python sensor_fusion.py --base ./KITTI --model yolov8n-seg.pt --save --output ./results"""

import cv2
import numpy as np
import argparse
import os
import sys
from pathlib import Path
from ultralytics import YOLO


# Constants
# YOLO class index for 'car' in COCO
CAR_CLASS_ID = 2

# Percentile used for bumper depth estimation
DEPTH_PERCENTILE = 5

# Display window size
DISPLAY_WIDTH  = 1200
DISPLAY_HEIGHT = 400

# Point visualisation sizes
INLIER_RADIUS   = 2
BLEEDOUT_RADIUS = 1

# Colours (BGR)
COLOR_INLIER   = (255, 0,   0  )   # Blue
COLOR_BLEEDOUT = (0,   0,   255)   # Red
COLOR_GT_BOX   = (0,   255, 0  )   # Green
COLOR_DEPTH    = (0,   255, 255)   # Yellow

# Argument Parser
def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="LiDAR-Camera sensor fusion for 3D object detection and depth estimation on KITTI."
    )
    parser.add_argument("--base",   required=True,
                        help="Path to the root KITTI dataset directory.")
    parser.add_argument("--model",  required=True,
                        help="Path to the YOLO segmentation model weights (e.g. yolov8n-seg.pt).")
    parser.add_argument("--percentile", type=int, default=DEPTH_PERCENTILE,
                        help=f"Percentile for bumper depth estimation. Default: {DEPTH_PERCENTILE}")
    parser.add_argument("--save",   action="store_true",
                        help="Save annotated output frames to disk.")
    parser.add_argument("--output", default="./results",
                        help="Output directory for saved frames. Default: ./results")
    parser.add_argument("--no_display", action="store_true",
                        help="Skip the interactive display window (for headless environments).")
    return parser.parse_args()



# Path Resolution
def resolve_kitti_paths(base_dir: Path):
    """Resolve KITTI dataset directory paths, supporting both nested
    and flat folder structures.

    Args:
        base_dir: Root KITTI dataset directory.

    Returns:
        Tuple of (calib_dir, image_dir, label_dir, velo_dir) as Path objects.

    Raises:
        SystemExit: If no valid folder structure is found."""
  
    nested_image = base_dir / "data_object_image_2" / "training" / "image_2"

    if nested_image.exists():
        calib_dir = base_dir / "data_object_calib"  / "training" / "calib"
        image_dir = nested_image
        label_dir = base_dir / "data_object_label_2" / "training" / "label_2"
        velo_dir  = base_dir / "data_object_velodyne" / "training" / "velodyne"
        print("[INFO] Using nested KITTI folder structure.")
    else:
        calib_dir = base_dir / "data_object_calib"
        image_dir = base_dir / "data_object_image_2"
        label_dir = base_dir / "data_object_label_2"
        velo_dir  = base_dir / "data_object_velodyne"
        print("[INFO] Using flat KITTI folder structure.")

    if not image_dir.exists():
        print(f"[ERROR] Image directory not found: '{image_dir}'")
        sys.exit(1)

    return calib_dir, image_dir, label_dir, velo_dir

# Calibration Loading
def load_calibration(calib_path: Path):
    """Load KITTI calibration matrices from a per-frame .txt file.

    Parses the projection matrix P2, rectification matrix R0_rect,
    and the LiDAR-to-camera transform Tr_velo_to_cam.

    Args:
        calib_path: Path to the calibration file.

    Returns:
        Tuple of (P2, R0, Tr):
            P2: 3x4 projection matrix.
            R0: 3x3 rectification matrix.
            Tr: 3x4 LiDAR-to-camera transform.

    Raises:
        SystemExit: If the file cannot be parsed."""
  
    try:
        with open(calib_path, "r") as f:
            lines = {}
            for line in f:
                if line.strip() and ":" in line:
                    key, val = line.split(":", 1)
                    lines[key.strip()] = np.array([float(x) for x in val.strip().split()])

        P2 = lines["P2"].reshape(3, 4)
        R0 = lines["R0_rect"].reshape(3, 3)
        Tr = lines["Tr_velo_to_cam"].reshape(3, 4)
        return P2, R0, Tr

    except Exception as e:
        print(f"[ERROR] Failed to parse calibration file '{calib_path}': {e}")
        sys.exit(1)

# Ground Truth Loading
def load_gt_cars(label_path: Path):
    """Load ground truth car annotations from a KITTI label file.

    Each car entry contains 3D dimensions, location, and yaw angle.

    Args:
        label_path: Path to the label .txt file.

    Returns:
        List of dicts with keys: 'dims' [h, w, l], 'loc' [x, y, z], 'yaw'."""
  
    gt_cars = []

    try:
        with open(label_path, "r") as f:
            for line_num, line in enumerate(f, 1):
                parts = line.strip().split()
                if not parts:
                    continue
                if parts[0] != "Car":
                    continue
                if len(parts) < 15:
                    print(f"[WARNING] Skipping malformed label line {line_num} in '{label_path}'")
                    continue
                gt_cars.append({
                    "dims": [float(parts[8]), float(parts[9]), float(parts[10])],
                    "loc":  np.array([float(parts[11]), float(parts[12]), float(parts[13])]),
                    "yaw":  float(parts[14])
                })
    except Exception as e:
        print(f"[ERROR] Failed to read label file '{label_path}': {e}")

    return gt_cars

# LiDAR Projection
def project_lidar_to_image(raw_pts: np.ndarray, Tr: np.ndarray, R0: np.ndarray, P2: np.ndarray):
    """Project raw Velodyne LiDAR points into the image plane using KITTI
    calibration matrices.

    Pipeline: LiDAR → Camera → Rectified Camera → Image

    Args:
        raw_pts: Nx3 array of LiDAR points in sensor frame.
        Tr:      3x4 LiDAR-to-camera transform.
        R0:      3x3 rectification matrix.
        P2:      3x4 projection matrix.

    Returns:
        Tuple of (u, v, depths, pts_rect_cam):
            u, v:          Integer pixel coordinates for each point.
            depths:        Depth values in camera space (z).
            pts_rect_cam:  Nx3 rectified 3D camera-frame coordinates."""
  
    pts_homo  = np.hstack((raw_pts, np.ones((raw_pts.shape[0], 1))))
    pts_cam   = np.dot(Tr, pts_homo.T)
    pts_rect  = np.dot(R0, pts_cam)

    pts_rect_homo = np.vstack((pts_rect, np.ones((1, raw_pts.shape[0]))))
    pts_2d        = np.dot(P2, pts_rect_homo)

    u      = (pts_2d[0, :] / pts_2d[2, :]).astype(int)
    v      = (pts_2d[1, :] / pts_2d[2, :]).astype(int)
    depths = pts_2d[2, :]

    return u, v, depths, pts_rect.T

# Segmentation Mask Extraction
def get_car_masks(results, h_img: int, w_img: int):
    """Extract binary segmentation masks for detected cars from YOLO results.

    Filters detections by CAR_CLASS_ID and rasterises polygon masks
    onto a blank canvas per detected car.

    Args:
        results: YOLO inference results.
        h_img:   Image height in pixels.
        w_img:   Image width in pixels.

    Returns:
        List of binary mask images (uint8, 0 or 255)."""
  
    masks = []
    for r in results:
        if r.masks is None:
            continue
        for mask, box in zip(r.masks.xy, r.boxes):
            if int(box.cls[0]) != CAR_CLASS_ID:
                continue
            canvas = np.zeros((h_img, w_img), dtype=np.uint8)
            cv2.fillPoly(canvas, [mask.astype(np.int32)], 255)
            masks.append(canvas)
    return masks


# 3D Box Geometry
def compute_3d_box_corners(h: float, w: float, l: float, loc: np.ndarray, yaw: float) -> np.ndarray:
    """Compute the 8 corners of a 3D bounding box in camera coordinates.

    Args:
        h:   Box height.
        w:   Box width.
        l:   Box length.
        loc: 3D location [x, y, z] of the box centre bottom.
        yaw: Rotation angle around the Y axis in radians.

    Returns:
        3x8 array of 3D corner coordinates."""
  
    x_corners = [ l/2,  l/2, -l/2, -l/2,  l/2,  l/2, -l/2, -l/2]
    y_corners = [ 0,    0,    0,    0,    -h,   -h,   -h,   -h  ]
    z_corners = [ w/2, -w/2, -w/2,  w/2,  w/2, -w/2, -w/2,  w/2]

    corners_3d = np.vstack([x_corners, y_corners, z_corners])

    R = np.array([
        [ np.cos(yaw), 0, np.sin(yaw)],
        [ 0,           1, 0          ],
        [-np.sin(yaw), 0, np.cos(yaw)]
    ])

    rotated = np.dot(R, corners_3d)
    rotated[0, :] += loc[0]
    rotated[1, :] += loc[1]
    rotated[2, :] += loc[2]

    return rotated


def project_3d_box(corners_3d: np.ndarray, P2: np.ndarray):
    """Project 3D box corners into 2D image pixel coordinates.

    Args:
        corners_3d: 3x8 array of 3D corner coordinates.
        P2:         3x4 projection matrix.

    Returns:
        Tuple of (u_box, v_box): integer pixel coordinates for 8 corners."""
  
    corners_homo = np.vstack([corners_3d, np.ones((1, 8))])
    proj  = np.dot(P2, corners_homo)
    u_box = (proj[0, :] / proj[2, :]).astype(int)
    v_box = (proj[1, :] / proj[2, :]).astype(int)
    return u_box, v_box

# 3D Wireframe Drawing
def draw_3d_box(display_img: np.ndarray, u_box: np.ndarray, v_box: np.ndarray):
    """Draw a 3D wireframe bounding box onto the display image.

    Connects the 8 projected corners to form the box edges.

    Args:
        display_img: BGR image to draw on (modified in-place).
        u_box:       Pixel x-coordinates of 8 corners.
        v_box:       Pixel y-coordinates of 8 corners."""
  
    for i in range(4):
        cv2.line(display_img,
                 (u_box[i], v_box[i]),
                 (u_box[(i + 1) % 4], v_box[(i + 1) % 4]),
                 COLOR_GT_BOX, 1)
        cv2.line(display_img,
                 (u_box[i + 4], v_box[i + 4]),
                 (u_box[((i + 1) % 4) + 4], v_box[((i + 1) % 4) + 4]),
                 COLOR_GT_BOX, 1)
        cv2.line(display_img,
                 (u_box[i], v_box[i]),
                 (u_box[i + 4], v_box[i + 4]),
                 COLOR_GT_BOX, 1)


# Legend Drawing
def draw_legend(display_img: np.ndarray):
    """Draw a colour-coded legend onto the display image.

    Args:
        display_img: BGR image to draw on (modified in-place)."""
  
    cv2.rectangle(display_img, (15, 15), (300, 95), (0, 0, 0), -1)
    cv2.putText(display_img, "GREEN : Ground Truth 3D Box",  (25, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_GT_BOX,   1)
    cv2.putText(display_img, "BLUE  : Inlier LiDAR Points",  (25, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_INLIER,   1)
    cv2.putText(display_img, "RED   : Bleed-Out Points",     (25, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_BLEEDOUT, 1)

# Per-Car Evaluation
def evaluate_car(
    car:          dict,
    masks:        list,
    u:            np.ndarray,
    v:            np.ndarray,
    depths:       np.ndarray,
    pts_rect_cam: np.ndarray,
    h_img:        int,
    w_img:        int,
    percentile:   int,
):
    """Evaluate a single ground truth car by fusing LiDAR points with
    segmentation masks and estimating bumper depth.

    Steps:
        1. Build a 3D axis-aligned gate from GT dimensions and location.
        2. Select the segmentation mask with the highest 3D-consistent hits.
        3. Separate inliers from bleed-out points.
        4. Estimate bumper depth using the chosen percentile of inlier z-values.

    Args:
        car:          GT car dict with 'dims', 'loc', 'yaw'.
        masks:        List of binary segmentation masks.
        u, v:         Projected LiDAR pixel coordinates.
        depths:       LiDAR point depths in camera space.
        pts_rect_cam: Nx3 rectified 3D camera-frame LiDAR points.
        h_img:        Image height.
        w_img:        Image width.
        percentile:   Depth percentile for bumper estimation.

    Returns:
        Dict with evaluation results, or None if no mask matched."""
  
    c_loc       = car["loc"]
    ch, cw, cl  = car["dims"]

    x_min_gate = c_loc[0] - cw / 2
    x_max_gate = c_loc[0] + cw / 2
    y_min_gate = c_loc[1] - ch
    y_max_gate = c_loc[1]
    z_min_gate = c_loc[2] - cl / 2
    z_max_gate = c_loc[2] + cl / 2

    valid_idx  = (depths > 0) & (u >= 0) & (u < w_img) & (v >= 0) & (v < h_img)
    best_pts   = np.array([])
    best_u_pts = None
    best_v_pts = None
    max_hits   = -1

    for mask in masks:
        mask_filter    = mask[np.clip(v, 0, h_img - 1), np.clip(u, 0, w_img - 1)] == 255
        combined       = valid_idx & mask_filter
        matched_pts    = pts_rect_cam[combined]

        if len(matched_pts) == 0:
            continue

        hits = np.sum(
            (matched_pts[:, 0] >= x_min_gate) & (matched_pts[:, 0] <= x_max_gate) &
            (matched_pts[:, 1] >= y_min_gate) & (matched_pts[:, 1] <= y_max_gate) &
            (matched_pts[:, 2] >= z_min_gate) & (matched_pts[:, 2] <= z_max_gate)
        )

        if hits > max_hits:
            max_hits   = hits
            best_pts   = matched_pts
            best_u_pts = u[combined]
            best_v_pts = v[combined]

    if len(best_pts) == 0 or best_u_pts is None:
        return None

    inside_filter = (
        (best_pts[:, 0] >= x_min_gate) & (best_pts[:, 0] <= x_max_gate) &
        (best_pts[:, 1] >= y_min_gate) & (best_pts[:, 1] <= y_max_gate) &
        (best_pts[:, 2] >= z_min_gate) & (best_pts[:, 2] <= z_max_gate)
    )

    num_correct   = int(np.sum(inside_filter))
    num_bleed     = len(best_pts) - num_correct
    bleed_pct     = (num_bleed   / len(best_pts)) * 100
    inlier_pct    = (num_correct / len(best_pts)) * 100

    eval_pool              = best_pts[inside_filter, 2] if num_correct > 0 else best_pts[:, 2]
    estimated_depth        = float(np.percentile(eval_pool, percentile))
    true_bumper_depth      = float(c_loc[2] - (cl / 2))
    abs_error              = abs(estimated_depth - true_bumper_depth)

    return {
        "best_pts":       best_pts,
        "best_u":         best_u_pts,
        "best_v":         best_v_pts,
        "inside_filter":  inside_filter,
        "num_correct":    num_correct,
        "num_bleed":      num_bleed,
        "bleed_pct":      bleed_pct,
        "inlier_pct":     inlier_pct,
        "estimated_depth": estimated_depth,
        "true_depth":     true_bumper_depth,
        "abs_error":      abs_error,
        "dims":           (ch, cw, cl),
        "loc":            c_loc,
        "yaw":            car["yaw"],
    }

# Main Pipeline
def main():
    """Full sensor fusion pipeline:
        1. Resolve KITTI directory structure.
        2. Load YOLO segmentation model.
        3. For each frame: load image, LiDAR, calibration, and GT labels.
        4. Project LiDAR into image space.
        5. Run YOLO segmentation and extract car masks.
        6. Evaluate each GT car: inlier/bleed-out split, depth estimation.
        7. Visualise 3D boxes and colour-coded LiDAR points.
        8. Optionally save annotated frames."""
  
    args     = parse_args()
    base_dir = Path(args.base)

    if not base_dir.exists():
        print(f"[ERROR] Base directory not found: '{base_dir}'")
        sys.exit(1)

    if not Path(args.model).exists():
        print(f"[ERROR] Model weights not found: '{args.model}'")
        sys.exit(1)

    if args.save:
        os.makedirs(args.output, exist_ok=True)

    calib_dir, image_dir, label_dir, velo_dir = resolve_kitti_paths(base_dir)

    image_files = sorted(list(image_dir.glob("*.png")))
    if not image_files:
        print(f"[ERROR] No .png images found in '{image_dir}'")
        sys.exit(1)

    print(f"[INFO] Discovered {len(image_files)} frame(s).")

    model = YOLO(args.model)
    print(f"[INFO] Loaded model: {args.model}")
    print(f"[INFO] Depth percentile: {args.percentile}")
    print("\n[INFO] Starting sensor fusion loop...\n")

    for img_path in image_files:
        frame_id   = img_path.stem
        calib_path = calib_dir / f"{frame_id}.txt"
        label_path = label_dir / f"{frame_id}.txt"
        velo_path  = velo_dir  / f"{frame_id}.bin"

        print(f"[FRAME {frame_id}]                         ")

        if not all([calib_path.exists(), label_path.exists(), velo_path.exists()]):
            print("[WARNING] Missing required file(s), skipping frame.")
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            print("[WARNING] Could not load image, skipping.")
            continue

        display_img      = img.copy()
        h_img, w_img, _  = img.shape
        raw_pts          = np.fromfile(str(velo_path), dtype=np.float32).reshape(-1, 4)[:, :3]

        P2, R0, Tr = load_calibration(calib_path)
        gt_cars    = load_gt_cars(label_path)

        if not gt_cars:
            print("[INFO] No car labels found in frame, skipping.")
            continue

        u, v, depths, pts_rect_cam = project_lidar_to_image(raw_pts, Tr, R0, P2)

        results = model(img_path, verbose=False)
        masks   = get_car_masks(results, h_img, w_img)

        if not masks:
            print("[INFO] No cars segmented in this frame.")
            continue

        print(f"{'Car':<6}  {'Total':>6}  {'Inlier':>7}  {'Bleed':>6}  "
              f"{'Inlier%':>8}  {'Bleed%':>7}  {'GT Depth':>9}  {'Est Depth':>10}  {'Abs Err':>8}")
        print("-" * 85)

        for idx, car in enumerate(gt_cars):
            result = evaluate_car(
                car, masks, u, v, depths, pts_rect_cam,
                h_img, w_img, args.percentile
            )

            if result is None:
                print(f"  Car #{idx + 1}: Missed by all segmentation masks.")
                continue

            print(f"  #{idx+1:<4}  "
                  f"{len(result['best_pts']):>6}  "
                  f"{result['num_correct']:>7}  "
                  f"{result['num_bleed']:>6}  "
                  f"{result['inlier_pct']:>7.2f}%  "
                  f"{result['bleed_pct']:>6.2f}%  "
                  f"{result['true_depth']:>9.2f}m  "
                  f"{result['estimated_depth']:>9.2f}m  "
                  f"{result['abs_error']:>7.3f}m")

            # Draw colour-coded LiDAR points
            for i in range(len(result["best_pts"])):
                px = int(result["best_u"][i])
                py = int(result["best_v"][i])
                if 0 <= px < w_img and 0 <= py < h_img:
                    if result["inside_filter"][i]:
                        cv2.circle(display_img, (px, py), INLIER_RADIUS,   COLOR_INLIER,   -1)
                    else:
                        cv2.circle(display_img, (px, py), BLEEDOUT_RADIUS, COLOR_BLEEDOUT, -1)

            # Draw 3D wireframe box
            ch, cw, cl = result["dims"]
            corners_3d = compute_3d_box_corners(ch, cw, cl, result["loc"], result["yaw"])
            u_box, v_box = project_3d_box(corners_3d, P2)
            draw_3d_box(display_img, u_box, v_box)

            # Draw estimated depth label
            cv2.putText(display_img,
                        f"{result['estimated_depth']:.2f}m",
                        (u_box[4], v_box[4] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_DEPTH, 2)

        draw_legend(display_img)

        if args.save:
            out_path = os.path.join(args.output, f"{frame_id}_fusion.png")
            cv2.imwrite(out_path, display_img)
            print(f"[INFO] Saved: {out_path}")

        if not args.no_display:
            cv2.namedWindow(f"Sensor Fusion - Frame {frame_id}", cv2.WINDOW_NORMAL)
            cv2.resizeWindow(f"Sensor Fusion - Frame {frame_id}", DISPLAY_WIDTH, DISPLAY_HEIGHT)
            cv2.imshow(f"Sensor Fusion - Frame {frame_id}", display_img)
            print("[INFO] Press any key to continue to the next frame, or 'q' to quit.")
            key = cv2.waitKey(0) & 0xFF
            cv2.destroyAllWindows()
            if key == ord('q'):
                print("[INFO] Quit key pressed — stopping early.")
                break

    print("\n[INFO] ================ PROCESSING COMPLETE ================")

# Entry Point
if __name__ == "__main__":
    main()
