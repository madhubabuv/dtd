import os
import numpy as np
import argparse
from sdk.build_pointcloud import build_pointcloud
from datasets.robotcar.junk.transform import build_se3_transform
from sdk.camera_model import CameraModel
import tqdm

parser = argparse.ArgumentParser(description="Project LIDAR data into camera image")
parser.add_argument("--models_dir", type=str, help="Directory containing camera models")
parser.add_argument(
    "--extrinsics_dir", type=str, help="Directory containing sensor extrinsics"
)
parser.add_argument("--image_dir", type=str, help="Directory containing images")
parser.add_argument("--testfile_timestamps", type=str, help="test_images timestamps")
args = parser.parse_args()

if __name__ == "__main__":

    model = CameraModel(args.models_dir, args.image_dir)
    extrinsics_path = os.path.join(args.extrinsics_dir, "stereo" + ".txt")
    poses_file = os.path.join(args.image_dir, "gps/rtk.csv")
    laser_dir = os.path.join(args.image_dir, "lms_front/")

    with open(extrinsics_path) as extrinsics_file:
        extrinsics = [float(x) for x in next(extrinsics_file).split(" ")]

    G_camera_vehicle = build_se3_transform(extrinsics)
    G_camera_posesource = None
    with open(os.path.join(args.extrinsics_dir, "ins.txt")) as extrinsics_file:
        extrinsics = next(extrinsics_file)
        G_camera_posesource = G_camera_vehicle * build_se3_transform(
            [float(x) for x in extrinsics.split(" ")]
        )

    print("G_camera_posesource: ", G_camera_posesource)
    timestamps_path = args.testfile_timestamps

    timestamp = 0

    init_flag = True

    depth_all = []

    timestamps_file = np.loadtxt(timestamps_path)

    shape = timestamps_file.shape

    if len(shape) > 1:

        timestamps_file = timestamps_file[:, 0]

    for line in tqdm.tqdm(timestamps_file, total=len(timestamps_file)):

        timestamp = int(line)
        im_shape = (960, 1280)
        pointcloud, reflectance = build_pointcloud(
            laser_dir,
            poses_file,
            args.extrinsics_dir,
            timestamp,
            timestamp + 1e7,
            timestamp,
        )

        # breakpoint()
        pointcloud = np.dot(G_camera_posesource, pointcloud)
        velo_pts_im, velo_depth = model.project(pointcloud, im_shape)
        velo_pts_im = velo_pts_im.T

        velo_pts_im[:, 0] = np.round(velo_pts_im[:, 0]) - 1
        velo_pts_im[:, 1] = np.round(velo_pts_im[:, 1]) - 1

        depth = np.zeros((im_shape[0], im_shape[1]))

        depth[
            velo_pts_im[:, 1].astype(np.int32), velo_pts_im[:, 0].astype(np.int32)
        ] = np.expand_dims(velo_depth, 1)

        depth[depth < 0] = 0
        crop_height = (im_shape[0] * 4) // 5
        depth = depth[:crop_height, :]
        depth_all.append(depth)

        # breakpoint()

    breakpoint()
    print("saving_depths")
    np.save("/hdd1/madhu/data/robotcar/2014-12-16-18-44-24/depth_evaluation/gt_depths.npy",depth_all)
