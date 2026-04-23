from pathlib import Path
import pandas as pd
import numpy as np
import argparse
import yaml
import csv
import os

import rerun as rr
import rerun.blueprint as rrb

import cuvslam
from vslamlab_utilities import color_from_id, get_distortion, load_frame

def load_calibration(calibration_yaml: Path, cam_name: str):

    with open(calibration_yaml, 'r') as file:
        data = yaml.safe_load(file)
    cameras = data.get('cameras', [])
    for cam_ in cameras:
        if cam_['cam_name'] == cam_name:
            cam = cam_;
            break;
    
    print(f"\nCamera Name: {cam['cam_name']}")
    print(f"Camera Type: {cam['cam_type']}")
    print(f"Camera Model: {cam['cam_model']}")
    print(f"Focal Length: {cam['focal_length']}")
    print(f"Principal Point: {cam['principal_point']}")
    has_dist, distortion = get_distortion(cam)
    if has_dist:
        print(f"Distortion Type Dimension: {cam['distortion_type']}")
        print(f"Distortion Coefficients: {cam['distortion_coefficients']}")
    print(f"Image Dimension: {cam['image_dimension']}")
    print(f"Fps: {cam['fps']}")

    camera = cuvslam.Camera()
    camera.size = (cam['image_dimension'][0], cam['image_dimension'][1])
    camera.principal = cam['principal_point']
    camera.focal = cam['focal_length']
    if has_dist:
        camera.distortion = distortion
    depth_name = cam['depth_name']

    rgbd_settings = cuvslam.Tracker.OdometryRGBDSettings()
    rgbd_settings.depth_scale_factor = cam['depth_factor']
    rgbd_settings.depth_camera_id = 0
    rgbd_settings.enable_depth_stereo_tracking = False

    return camera, rgbd_settings, depth_name

def main():  
    print("\nRunning vslamlab_pycuvslam_rgbd.py ...")  

    parser = argparse.ArgumentParser()
    
    parser.add_argument("--sequence_path", type=Path, required=True)
    parser.add_argument("--calibration_yaml", type=Path, required=True)
    parser.add_argument("--rgb_csv", type=Path, required=True)
    parser.add_argument("--exp_folder", type=Path, required=True)
    parser.add_argument("--exp_it", type=str, default="0")
    parser.add_argument("--settings_yaml", type=Path, default=None)
    parser.add_argument("--verbose", type=str, help="verbose")

    args, _ = parser.parse_known_args()

    # Load camera name
    with open(args.settings_yaml, 'r') as file:
        data = yaml.safe_load(file)
    
    cam_name = data['cam_mono']

    # Set up camera parameters
    camera, rgbd_settings, depth_name = load_calibration(args.calibration_yaml, cam_name)

    # Configure tracker
    cfg_odom = cuvslam.Tracker.OdometryConfig(
        async_sba=True,
        enable_observations_export=True,
        enable_final_landmarks_export=True,
        horizontal_stereo_camera=False,
        odometry_mode=cuvslam.Tracker.OdometryMode.RGBD,
        rgbd_settings=rgbd_settings)

    # Initialize tracker
    tracker = cuvslam.Tracker(cuvslam.Rig([camera]), cfg_odom)

    # Load rgb images
    df = pd.read_csv(args.rgb_csv)       
    image_list = df[f'path_{cam_name}'].to_list()
    depth_list = df[f'path_{depth_name}'].to_list() 
    timestamps = df[f'ts_{cam_name} (ns)'].to_list()

    # Setup rerun visualizer
    rr.init('vslamlab_dataset', strict=True, spawn=True)

    # Setup coordinate basis for root
    # cuvslam uses right-hand system with X-right, Y-down, Z-forward
    rr.log("/", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)

    # Setup rerun views
    rr.send_blueprint(rrb.Blueprint(
        rrb.TimePanel(state="collapsed"),
        rrb.Horizontal(contents=[
            rrb.Vertical(contents=[
                rrb.Spatial2DView(origin='world/camera/image', name='RGB Camera'),
                rrb.Spatial2DView(origin='world/camera/depth', name='Depth Camera')
            ]),
            rrb.Spatial3DView(
                name="3D",
                defaults=[rr.components.ImagePlaneDistance(0.5)]
            )
        ]),
    ), make_active=True)

    trajectory_odom_t, trajectory_odom = [], []
    for t, imrel in enumerate(image_list):
        rgb_path = args.sequence_path / imrel
        depth_path = args.sequence_path / depth_list[t]

        color_frame = load_frame(rgb_path)
        depth_frame = load_frame(depth_path)

        # Track frame
        odom_pose_estimate, _ = tracker.track(
            timestamps[t], images=[color_frame], depths=[depth_frame]
        )
        if odom_pose_estimate.world_from_rig is None:
            print(f"Warning: Failed to track frame {t}")
            continue

        # Get current pose and observations for the main camera and gravity in rig frame
        odom_pose = odom_pose_estimate.world_from_rig.pose
        trajectory_odom_t.append(odom_pose.translation)
        trajectory_odom.append([timestamps[t]] + list(odom_pose.translation) + list(odom_pose.rotation))

        # Get observations
        observations = [tracker.get_last_observations(0)]
        
        # Extract observation points and colors
        obs_uv = [[o.u, o.v] for o in observations[0]]
        obs_colors = [color_from_id(o.id) for o in observations[0]]

        # Log visualization data
        rr.set_time('frame', sequence=timestamps[t])
        rr.log('trajectory', rr.LineStrips3D(trajectory_odom_t), static=True)

        rr.log(
            "world/camera/rgb",
            rr.Transform3D(
                translation=odom_pose.translation,
                quaternion=odom_pose.rotation
            ),
            rr.Arrows3D(
                vectors=np.eye(3) * 0.2,
                colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]]  # RGB for XYZ axes
            )
        )
        
        # Log RGB and depth images with observations
        rr.log('world/camera/image', rr.Image(color_frame).compress(jpeg_quality=80))
        rr.log(
            'world/camera/image/observations',
            rr.Points2D(obs_uv, radii=5, colors=obs_colors)
        )
        rr.log('world/camera/depth', rr.Image(depth_frame))
        rr.log(
            'world/camera/depth/observations',
            rr.Points2D(obs_uv, radii=5, colors=obs_colors)
        )

    keyframe_csv = args.exp_folder / f"{args.exp_it.zfill(5)}_KeyFrameTrajectory.csv"
    with open(keyframe_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ts (ns)", "tx (m)", "ty (m)", "tz (m)", "qx", "qy", "qz", "qw"])
        for line in trajectory_odom:
            writer.writerow(line)

if __name__ == '__main__':
    main()