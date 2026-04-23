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
from vslamlab_utilities import color_from_id, get_distortion, load_frame, transform_to_cam0_reference, transform_to_pose

def load_calibration(calibration_yaml: Path, cam_l_name: str, cam_r_name: str):
    with open(calibration_yaml, 'r') as file:
        data = yaml.safe_load(file)
    cameras = data.get('cameras', [])
    for cam_ in cameras:
        if cam_['cam_name'] == cam_l_name:
            cam_l = cam_
        if cam_['cam_name'] == cam_r_name:
            cam_r = cam_
    
    cuvslam_cams, T_BS = {}, {}
    for cam, side in zip([cam_l, cam_r],['l', 'r']):
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

        cuvslam_cams[side] = cuvslam.Camera()
        cuvslam_cams[side].size = (cam['image_dimension'][0], cam['image_dimension'][1])
        cuvslam_cams[side].principal = cam['principal_point']
        cuvslam_cams[side].focal = cam['focal_length']

        if has_dist:
            cuvslam_cams[side].distortion = distortion
            
        T_BS[side] = np.array(cam['T_BS']).reshape(4, 4)  
    
    cuvslam_cams['l'].rig_from_camera = cuvslam.Pose(
        rotation=[0, 0, 0, 1],  
        translation=[0, 0, 0]
    )
    cam1_cam0_transform =  transform_to_cam0_reference(T_BS['l'], T_BS['r'])
    cuvslam_cams['r'].rig_from_camera = transform_to_pose(cam1_cam0_transform.flatten().tolist())
    return [cuvslam_cams['l'], cuvslam_cams['r']]


def main():  
    print("\nRunning vslamlab_pycuvslam_stereo.py ...")  

    parser = argparse.ArgumentParser()
    
    parser.add_argument("--sequence_path", type=Path, required=True)
    parser.add_argument("--calibration_yaml", type=Path, required=True)
    parser.add_argument("--rgb_csv", type=Path, required=True)
    parser.add_argument("--exp_folder", type=Path, required=True)
    parser.add_argument("--exp_it", type=str, default="0")
    parser.add_argument("--settings_yaml", type=Path, default=None)
    parser.add_argument("--verbose", type=str, help="verbose")

    args, _ = parser.parse_known_args()

    # Load camera names
    with open(args.settings_yaml, 'r') as file:
        data = yaml.safe_load(file)
    
    cam_l_name = data['cam_stereo'][0]
    cam_r_name = data['cam_stereo'][1]

    # Set up camera parameters
    cameras = load_calibration(calibration_yaml = args.calibration_yaml, 
                                       cam_l_name = cam_l_name, cam_r_name = cam_r_name)

    # Configure tracker
    cfg_odom = cuvslam.Tracker.OdometryConfig(
        async_sba=True,
        enable_observations_export=True,
        enable_final_landmarks_export=True,
        horizontal_stereo_camera=False,
        odometry_mode=cuvslam.Tracker.OdometryMode.Multicamera)

    # Initialize tracker
    tracker = cuvslam.Tracker(cuvslam.Rig(cameras), cfg_odom)

    # Load rgb images
    df = pd.read_csv(args.rgb_csv)       
    images_left = df[f'path_{cam_l_name}'].to_list()
    images_right = df[f'path_{cam_r_name}'].to_list()
    timestamps = df[f'ts_{cam_l_name} (ns)'].to_list()

    # Setup rerun visualizer
    rr.init('vslamlab_dataset', strict=True, spawn=True)

    # Setup coordinate basis for root
    # cuvslam uses right-hand system with X-right, Y-down, Z-forward
    rr.log("/", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)

    # Setup rerun views
    rr.send_blueprint(
        rrb.Blueprint(
            rrb.TimePanel(state="collapsed"),
            rrb.Horizontal(
                column_shares=[0.5, 0.5],
                contents=[
                    rrb.Vertical(contents=[
                        rrb.Horizontal(contents=[
                            rrb.Spatial2DView(origin='world/camera_0'),
                            rrb.Spatial2DView(origin='world/camera_1')
                        ]),                    
                    ]),
                    rrb.Spatial3DView(origin='world')
                ]
            )
        )
    )

    trajectory_odom_t, trajectory_odom = [], []
    for t, (imgL, imgR) in enumerate(zip(images_left, images_right)):
        imgL = os.path.join(args.sequence_path, imgL)
        imgR = os.path.join(args.sequence_path, imgR)

        images = [load_frame(image_path) for image_path in [imgL,imgR,]]

        # Track frame
        odom_pose_estimate, _ = tracker.track(timestamps[t], images=images)

        if odom_pose_estimate.world_from_rig is None:
            print(f"Warning: Failed to track frame {t}")
            continue

        # Get current pose and observations for the main camera and gravity in rig frame
        odom_pose = odom_pose_estimate.world_from_rig.pose
        current_observations_main_cam = tracker.get_last_observations(0)
        trajectory_odom_t.append(odom_pose.translation)
        trajectory_odom.append([timestamps[t]] + list(odom_pose.translation) + list(odom_pose.rotation))
       
        # Visualize
        rr.set_time("frame", sequence=t)
        rr.log("world/trajectory", rr.LineStrips3D(trajectory_odom_t), static=True)
        rr.log(
            "world/camera_0",
            rr.Transform3D(
                translation=odom_pose.translation,
                quaternion=odom_pose.rotation
            ),
            rr.Arrows3D(
                vectors=np.eye(3) * 0.2,
                colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]]  # RGB for XYZ axes
            )
        )

        points = np.array([[obs.u, obs.v] for obs in current_observations_main_cam])
        colors = np.array([color_from_id(obs.id) for obs in current_observations_main_cam])
        rr.log(
            "world/camera_0/observations",
            rr.Points2D(positions=points, colors=colors, radii=5.0),
            rr.Image(images[0]).compress(jpeg_quality=80)
        )

        rr.log(
            "world/camera_1/observations",
            rr.Points2D(positions=points, colors=colors, radii=5.0),
            rr.Image(images[1]).compress(jpeg_quality=80)
        )
        
    keyframe_csv = args.exp_folder / f"{args.exp_it.zfill(5)}_KeyFrameTrajectory.csv"
    with open(keyframe_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ts (ns)", "tx (m)", "ty (m)", "tz (m)", "qx", "qy", "qz", "qw"])
        for line in trajectory_odom:
            writer.writerow(line)

if __name__ == '__main__':
    main()