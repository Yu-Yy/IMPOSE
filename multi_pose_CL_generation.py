import numpy as np
import os
import os.path as osp
import scipy.io as scio
from mapping.Functions import PoseCorrection, PointsFlattenByCircle, GenerateFlattenResult, GenerateAlignmentResult
import pickle
import cv2
from tqdm import tqdm

texture_folder = 'example/contactless_texture'
texture_files = os.listdir(texture_folder)
point_folder = 'example/finger3d'
pose_folder = 'example/finger3d_pose'
save_folder = 'example/multi_pose_CL' # the compared part

# create save folder
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

point_file_list = os.listdir(point_folder)
point_file_list = [f for f in point_file_list if f.endswith('.mat')]

for texture_file in tqdm(texture_files):
    # random select a mesh from point file list
    point_file = np.random.choice(point_file_list)
    pose_range = [0] + list(np.random.randint(10, 31, size=2)) + list(-np.random.randint(10, 31, size=2)) + list(np.random.randint(31, 60, size=2)) + list(-np.random.randint(31, 60, size=2))

    pose_data = np.loadtxt(osp.join(pose_folder, point_file.replace('.mat', '.txt')))
    pickle_data_path = osp.join(point_folder, point_file.replace('.mat', '.pkl'))
    with open(pickle_data_path, 'rb') as f:
        pickle_data = pickle.load(f)
    flattened_3d_position = pickle_data['flattened_3d_position']
    pose_2d = pose_data[:2]
    # to index the 3d position
    pose_2d = np.round(pose_2d).astype(np.int32)
    center_3d_points = flattened_3d_position[pose_2d[1], pose_2d[0]]  # (x, y, z)
    yaw_correction = pose_data[2]

    data = scio.loadmat(osp.join(point_folder, point_file))
    points = data["points"]
    normals = data['normals']
    depth = data['depth']
    # Step 1: Initial Pose Correction
    points_corrected, normals_corrected = PoseCorrection(points, normals, pose_2d, yaw_correction)
    for idx, angle in enumerate(pose_range):
        # Step 2: Further Pose Correction
        uv_points_flat, uv_points_proj, jump_step, normal_rotate_dot = PointsFlattenByCircle(points_corrected, 500, normals_corrected, rolled_angle=angle)
        proj_mask = normal_rotate_dot < 0 
        # Step 3: Generate Flattened Result
        if idx == 0:
            flattened_img, flattened_3d_position, texture_points, flattened_texture_img = GenerateFlattenResult(
                points_corrected,
                depth,
                uv_points_flat,
                size=(800, 800),
                brightness=0.6,
                texture=osp.join(texture_folder, texture_file),
                generate_uv_texture=True,
            )
            save_img_path = osp.join(save_folder, f'{texture_file.split(".")[0]}_texture.png') # 
            cv2.imwrite(save_img_path, flattened_texture_img)

        pose_texture_img = GenerateAlignmentResult(
            uv_points_flat,
            uv_points_proj,
            proj_mask,
            size=(800, 800),
            texture=texture_points
        )
        save_img_path = osp.join(save_folder, f'{texture_file.split(".")[0]}_{angle}.png') #
        cv2.imwrite(save_img_path, pose_texture_img)

