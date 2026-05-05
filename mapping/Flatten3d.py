import os
import os.path as osp
import scipy.io as scio
import argparse
from Functions import PoseCorrection, PointsFlattenByCircle, GenerateFlattenResult
import imageio
import matplotlib.pyplot as plt

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dpi",
                        "-d",
                        default=500,
                        type=int,
                        help="generated figure's dpi")
    parser.add_argument("--edge",
                        "-e",
                        default=30,
                        type=int,
                        help="blank edge's pixels")
    parser.add_argument("--brightness",
                        "-b",
                        default=0.6,
                        type=float,
                        help="Luminance coefficient of image")
    args = parser.parse_args()

    # data_dir = "./data"
    data_dir = "/data/panzhiyu/fingerprint/UNSW_CZ/3d_finger_img/depth/"
    save_dir = "./result"
    # fname = "100_1_2.mat"
    fname = "1_1_1_0.mat"
    save_name = None

    save_dir = osp.join(
        save_dir, 'dpi' + str(args.dpi) + '_' + 'b' + str(args.brightness))
    if not osp.exists(save_dir):
        os.makedirs(save_dir)

    if save_name is None:
        save_name = fname.split('.')[0]

    data = scio.loadmat(osp.join(data_dir, fname))
    depth_result = data["depth_result"]
    points = data["points"]
    depth = data["depth"]
    normals = data['normals']

    # # ---- test 'PoseCorrection' function ----
    # M = Euler2Matrix(0, 60, 0)
    # points = np.dot(points, M.T)
    # normals = np.dot(normals, M.T)
    # pptk.viewer(points)
    # DrawPCA_2D(points,0,1)
    # points,normals = PoseCorrection(points,normals)
    # pptk.viewer(points)
    # DrawPCA_2D(points,0,1)
    # plt.show()
    # # ----------------------------------------

    points, normals = PoseCorrection(points, normals)
    uv_points, jump_step = PointsFlattenByCircle(points, args.dpi)
    grid_fp, grid_gt = GenerateFlattenResult(points,
                                             depth,
                                             uv_points,
                                             args.edge,
                                             brightness=args.brightness,
                                             fill_value=255)
    # uv_points is the 2D coordinates of points in the flattened image
    # plot the uv_points
    plt.figure(figsize=(10, 10), dpi=args.dpi)
    plt.plot(uv_points[:, 0], uv_points[:, 1], 'r.', markersize=0.5)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.axis('off')
    plt.savefig(osp.join(save_dir, save_name + '_uv.png'),
                dpi=args.dpi,
                bbox_inches='tight',
                pad_inches=0)
    import pdb;pdb.set_trace()

    imageio.imwrite(osp.join(save_dir, save_name + '.png'), grid_fp)
    import pdb; pdb.set_trace()
    scio.savemat(osp.join(save_dir, save_name + '.mat'), {
        'points': points,
        'normals': normals,
        'depth': depth,
        'grid_gt': grid_gt
    })
