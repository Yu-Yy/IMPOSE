import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from sklearn.decomposition import PCA
from scipy import optimize
from scipy.spatial.transform import Rotation
from scipy.interpolate import griddata
from skimage.measure import CircleModel
import cv2

def GetNearestEllipseAngle(p, params):
    """Get the angle from the negative direction of the vertical axis to the point on the ellipse closest to the specified point (anti-clockwise).

    Args:
        p ((N, 2) array): Points.
        params (list): [xc, yc, a, b, theta] for [horizontal axis center, vertical axis center,  major axis, minor axis, inclination angle(counterclockwise)].

    Returns:
        t ((N, ) array): Anti-clockwise angle.
        residuals ((N, ) array): Euclidean distance from the nearest point on the ellipse to the specified point.
    """
    xc, yc, a, b, theta = params

    ctheta = math.cos(theta)
    stheta = math.sin(theta)

    x = p[:, 0]
    y = p[:, 1]

    def fun(t, xi, yi):
        ct = math.cos(t)
        st = math.sin(t)
        xt = xc + a * ctheta * st + b * stheta * ct
        yt = yc + a * stheta * st - b * ctheta * ct
        return (xi - xt)**2 + (yi - yt)**2

    N = p.shape[0]
    t = np.empty((N, ), dtype=np.double)
    residuals = np.empty((N, ), dtype=np.double)

    # initial guess for parameter t of closest point on ellipse
    t0 = np.arctan2(x - xc, yc - y) - theta

    # determine shortest distance to ellipse for each point
    for i in range(N):
        xi = x[i]
        yi = y[i]
        ti, _ = optimize.leastsq(fun, t0[i], args=(xi, yi))
        t[i] = np.mod(ti, 2 * math.pi)
        residuals[i] = np.sqrt(fun(t[i], xi, yi))

    return t, residuals


def Euler2Matrix(roll, pitch, yaw):
    """Convert Euler angle to rotation matrix R.
    If the points' shape is (N, 3), then the rotated vector is np.dot(points, R.T).

    Args:
        roll (float): X.
        pitch (float): Y.
        yaw (float): Z.

    Returns:
        R ((3, 3) array): Rotation matrix.
    """
    R = Rotation.from_euler("ZYX", [yaw, pitch, roll],
                            degrees=True).as_matrix()
    return R


def DrawPCA_2D(points, d1, d2, center=None):
    """project [the point cloud] and [its principal axis vector obtained by PCA] in 2D plane.

    Args:
        points ((N, 3) array): Points for [x ,y, z]
        d1 (int): Dimension (0, 1, 2 for x, y, z)
        d2 (int): The other dimension (0, 1, 2 for x, y, z)
        center ((2, ) array, optional): Points' center. Defaults to None (decentralize the point set and set the center to [0,0,0]).
    """
    dim = ['x', 'y', 'z']

    pca = PCA(n_components=1)
    pca.fit(points)
    main_vec = pca.components_

    if center is None:
        points -= np.mean(points, axis=0)
        center = np.zeros(3)
    plt.figure()
    plt.scatter(points[:, d1], points[:, d2], s=1, c='b')
    plt.axis("equal")
    plt.scatter(center[d1], center[d2], s=8, c='r')
    plt.plot(np.array([center[d1], center[d1] + 10 * main_vec[0, d1]]),
             np.array([center[d2], center[d2] + 10 * main_vec[0, d2]]),
             '-r',
             linewidth=2)
    plt.xlabel(dim[d1])
    plt.ylabel(dim[d2])


def PoseCorrection(points, normals=None, pose_2d=None, yaw_correction=0):
    """Correct finger posture. Make the finger surface face the negative direction of Y axis.
    Use PCA method to correct x-axis and z-axis.
    Use the normal vector to correct y-axis.

    Args:
        points ((N,3) array): [x,y,z] points.
        normals ((N,3) array, optional): Normal vectors of points. Defaults to None (will not correct the y-axis posture).

    Returns:
        points ((N,3) array): [x,y,z] points.
        normals ((N,3) array): Normal vectors of points. 
    """
    points -= np.mean(points, axis=0)
    pca = PCA(n_components=1)
    pca.fit(points)

    main_vec = pca.components_

    standard_vec = np.array([0, -1, 0])
    if np.dot(main_vec, standard_vec) < 0:
        main_vec = -main_vec

    yaw = math.degrees(math.atan2(main_vec[0, 0], -main_vec[0, 1]))
    yaw -= yaw_correction
    roll = -math.degrees(math.asin(main_vec[0, 2]))
    R = Euler2Matrix(roll=roll, pitch=0, yaw=yaw)

    points = np.dot(points, R)  # (R.T).T
    points -= np.mean(points, axis=0)

    if normals is not None:
        normals = np.dot(normals, R)  # (R.T).T
        mean_normal = np.mean(normals, axis=0)
        pitch = math.degrees(math.atan2(mean_normal[0], -mean_normal[2]))
        R = Euler2Matrix(roll=0, pitch=pitch, yaw=0)
        normals = np.dot(normals, R.T)
        points = np.dot(points, R.T)
        points -= np.mean(points, axis=0)

    return points, normals


def ClockwiseAngle(base_vec, rotate_vec):
    """Clockwise angle from base_vec to rotate_vec

    Args:
        base_vec ((2, ) array): Reference vector.
        rotate_vec ((2, ) array): Rotated vector.

    Returns:
        theta (float): Clockwise angle (rad).
    """
    x1, y1 = base_vec
    x2, y2 = rotate_vec
    dot = x1 * x2 + y1 * y2
    det = x1 * y2 - y1 * x2
    theta = np.arctan2(det, dot)
    theta = theta if theta > 0 else 2 * np.pi + theta
    return theta


def DrawEllipse(points_x, points_y, xc, yc, a, b, theta):
    """Draw points and corresponding fitted ellipse

    Args:
        points_x ((N, ) array): Abscissa of points.
        points_y ((N, ) array)): Ordinate of points.
        xc (float): Horizontal axis center.
        yc (float): Vertical axis center.
        a (float): Major axis.
        b (float): Minor axis.
        theta (float): Inclination angle(counterclockwise).
    """
    plt.figure()
    plt.axes().set_aspect('equal', adjustable='box')
    plt.axes().scatter(points_x, points_y, c='b')
    ell_patch = Ellipse((xc, yc),
                        2 * a,
                        2 * b,
                        np.rad2deg(theta),
                        edgecolor='red',
                        facecolor='none')
    plt.axes().add_patch(ell_patch)


def arc_int(phi, a, b):
    """ Differential function of elliptic arc length (Start with the minor axis below).
    The parametric equation of the ellipse is:
        x = a sin(phi),
        y = -b cos(phi).
    Therefore, the differential formula of arc length is:
        [dL] = sqrt( (a cos(phi))**2 + (b sin(phi))**2 ) [dphi].

    Args:
        phi (float): Anti-clockwise angle from the lower minor axis to the corresponding point (rad).
        a (float): Major axis.
        b (float): Minor axis.

    Returns:
        float: [dL]/[dphi].
    """
    return math.sqrt(
        math.pow(a * math.cos(phi), 2) + math.pow(b * math.sin(phi), 2))


def GetEllPoints(t, params):
    """Get the coordinates of a point according to [the ellipse parameters] and [the corresponding angle of the point on the ellipse]

    Args:
        t (float): Anti-clockwise angle from the lower minor axis of the ellipse to this point (rad).
        params (list): [xc, zc, a, b, theta] for [horizontal axis center, vertical axis center,  major axis, minor axis, inclination angle(counterclockwise)].

    Returns:
        xt (float): Abscissa of point.
        zt (float): Ordinate of point.
    """
    xc, zc, a, b, theta = params
    ctheta = math.cos(theta)
    stheta = math.sin(theta)
    ct = math.cos(t)
    st = math.sin(t)
    xt = xc + a * ctheta * st + b * stheta * ct
    zt = zc + a * stheta * st - b * ctheta * ct
    return xt, zt

def rotate_y(points: np.ndarray, angle_deg: float) -> np.ndarray:
    """
    points: (N, 3)d
    angle_rad: radians, positive for counter-clockwise rotation
    return: (N, 3)
    """
    angle_rad = np.deg2rad(angle_deg)
    points = np.asarray(points, dtype=np.float64)
    c, s = np.cos(angle_rad), np.sin(angle_rad)

    R = np.array([
        [ c, 0.0,  s],
        [0.0, 1.0, 0.0],
        [-s, 0.0,  c],
    ], dtype=np.float64)

    return points @ R.T  # (N,3) @ (3,3)^T


def PointsFlattenByCircle(points, dpi, normals, rolled_angle=0):
    """Calculate the corresponding coordinates in a 2D plane after expanding the 3D point cloud
    * The expanded abscissa is obtained by fitting the circle and integrating according to the circle curve. 
    The starting point of integration is the corresponding point closest to the longitudinal axis
    * The expanded ordinate is obtained by calculating the distance and accumulating the integral 
    according to the corresponding points in the negative direction of the longitudinal axis of each slice fitting ellipse

    Args:
        points ((N, 3) array): Points for [x, y, z]. The scale unit of point cloud is [mm] by default.
        dpi (flaot): Dots per inch. 1 inch = 25.4 mm.

    Returns:
        uv_points ((N, 2) array): 2D coordinates (vertical axis, horizontal axis) after expanding.
        jump_step (list): Indexes of sections where fitting failed.
    """
    mm2pixel = dpi / 25.4
    pixel2mm = 1 / mm2pixel

    N = points.shape[0]
    # expanded coordinates
    uv_points_flat = np.ones([N, 2]) * np.nan
    uv_points_proj = np.ones([N, 2]) * np.nan

    # longitudinal slice according to dpi
    max_y = np.max(points[:, 1])
    min_y = np.min(points[:, 1])
    len_y = math.ceil((max_y - min_y) * mm2pixel)

    # indexes corresponding to each point
    id = np.arange(points.shape[0])

    # cumulative integral initialization of vertical axis
    v = 0
    init_step = 0
    while True:
        step_flag = (points[:, 1] >= min_y + init_step * pixel2mm) & (
            points[:, 1] < min_y + (init_step + 1) * pixel2mm)
        points_step = points[step_flag, :]
        try:
            circle = CircleModel()
            circle.estimate(points_step[:, [0, 2]])
            xc, zc, r = circle.params
            if points_step.shape[0] == 0 or abs(xc) > r or zc < np.mean(
                    points_step[:, 2]):
                init_step += 1
                continue
            z_pre = zc - r
            z_post = z_pre
            break
        except:
            init_step += 1

    # failure of circle fitting
    jump_step = []
    points_rotated = rotate_y(points, angle_deg=rolled_angle) #
    # accordingly, rotate the normals too
    normals_rotated = rotate_y(normals, angle_deg=rolled_angle)
    project_vector = np.array([0.0, 0.0, 1.0], dtype=np.float64) # 
    # dot product of project_vector and normals_rotated to determine the facing direction
    normal_rotate_dot = normals_rotated @ project_vector.T # (N,3) @ (3,) -> (N,)

    u_displacement = 0.0

    # for step in tqdm(range(init_step, len_y, 1)):
    for step in range(init_step, len_y, 1):
        # points and indexes corresponding to each longitudinal slice
        step_flag = (points[:, 1] >= min_y + step * pixel2mm) & (
            points[:, 1] < min_y + (step + 1) * pixel2mm)
        current_y = min_y + step * pixel2mm
        points_step = points[step_flag, :]
        points_rotated_step = points_rotated[step_flag, :] 
        id_step = id[step_flag]

        # fit x-y points with circle
        try:
            circle = CircleModel()
            circle.estimate(points_step[:, [0, 2]])
            xc, zc, r = circle.params
            # When fitting is incorrect, the last longitudinal gradient is used as the approximation
            if points_step.shape[0] == 0 or abs(xc) > r or zc < np.mean(
                    points_step[:, 2]):
                v += math.sqrt(
                    math.pow(z_post - z_pre, 2) + math.pow(pixel2mm, 2))
                jump_step.append(step)
                continue
        except:
            # When fitting is not possible, the last longitudinal gradient is used as the approximation
            v += math.sqrt(math.pow(z_post - z_pre, 2) + math.pow(pixel2mm, 2))
            jump_step.append(step)
            continue

        # flat z
        z_post = zc - r
        v += math.sqrt(math.pow(z_post - z_pre, 2) + math.pow(pixel2mm, 2))
        z_pre = z_post

        circle_center_rotated = rotate_y(
            np.array([[xc, current_y, zc]], dtype=np.float64),
            angle_deg=rolled_angle
        )
        xc_rotated = circle_center_rotated[0, 0]
        zc_rotated = circle_center_rotated[0, 2]
        base_vec = np.array([0.0, -1.0])   
        phi_begin = math.asin(xc / r) # 
        if len_y * 0.25 <= step <= len_y * 0.75:
            if abs(xc_rotated) > r:
                jump_step.append(step)
                continue
            phi_begin_rotated = math.asin(xc_rotated / r) 
        # find the u_range of this circle
        u_min = np.inf
        u_min_rotated = np.inf

        for ps_i in range(points_step.shape[0]):
            point = points_step[ps_i, [0, 2]]  # (x,z) 
            point_rotated = points_rotated_step[ps_i, [0, 2]] #
            rotate_vec = point - np.array([xc, zc]) # 
            phi_end = ClockwiseAngle(base_vec, rotate_vec) # using two version of angle calculation
            phi = phi_end + phi_begin
            if phi > math.pi:
                phi = phi - 2 * math.pi

            u_flat_mm = phi * r       
            v_mm = v                  

            uv_points_flat[id_step[ps_i], 0] = v_mm * mm2pixel
            uv_points_flat[id_step[ps_i], 1] = u_flat_mm * mm2pixel


            uv_points_proj[id_step[ps_i], 0] = v_mm * mm2pixel
            # only consider the roll angle
            u_proj_mm = point_rotated[0] # direct x coordinate as projection
            uv_points_proj[id_step[ps_i], 1] = u_proj_mm * mm2pixel 
            # update the u_range according to steps
            if len_y * 0.25 <= step <= len_y * 0.75: # NOTE: need to further clarify
                rotate_vec_rotated = point_rotated - np.array([xc_rotated, zc_rotated])
                phi_end_rotated = ClockwiseAngle(base_vec, rotate_vec_rotated)
                phi_rotated = phi_end_rotated + phi_begin_rotated
                if phi_rotated > math.pi:
                    phi_rotated = phi_rotated - 2 * math.pi
                u_flat_mm_rotated = phi_rotated * r # 
                # update the u_range
                if u_flat_mm < u_min:
                    u_min = u_flat_mm
                if u_flat_mm_rotated < u_min_rotated:
                    u_min_rotated = u_flat_mm_rotated
        # 根据 u 范围, 计算本次滚动带来的整体translation
        current_delta_u = u_min_rotated - u_min
        if len_y * 0.25 <= step <= len_y * 0.75:
            if abs(current_delta_u) > abs(u_displacement):
                u_displacement = current_delta_u

    uv_points_proj[:, 1] -= u_displacement * mm2pixel # make a delta to meet the center alignment

    return uv_points_flat, uv_points_proj, jump_step, normal_rotate_dot


def HistEqualization(arr):
    """Normalize arr to [0, 255] and perform histogram equalization

    Args:
        arr ((N, ) array): Arr
    Returns:
        arr_equal ((N, ) array): Arr after histogram equalization
    """
    min_arr = arr.min()
    max_arr = arr.max()
    arr = 255 * (arr - min_arr) / (max_arr - min_arr)
    arr = np.squeeze(np.uint8(arr))
    arr_hist, arr_bins = np.histogram(arr, bins=256, range=(0, 256))

    cdf = np.cumsum(arr_hist)
    cdf = (cdf - cdf[0]) * 255 / (cdf[-1] - 1)
    cdf = np.uint8(cdf)
    arr_equal = cdf[arr]

    return arr_equal


def GenerateFlattenResult(points,
                          depth,
                          uv_points,
                          size=(800, 800),
                          brightness=0.6,
                          fill_value=255,
                          texture=None,
                          generate_uv_texture=False):
    """Get the interpolated results from the expanded coordinates.

    Args:
        points ((N, 3) array): Points for [x, y, z].
        depth ((N, ) array): The surface depth of each point, with this value as the gray value.
        uv_points ((N, 2) array): Expanded coordinates [v, u].
        edge (int): Pixels of the blank boundary of the image.
        brightness (float, optional): Luminance coefficient of image. If it is 1, the image does not change. Defaults to 0.6.
        fill_value (int, optional): The fill value of the blank area of the image. Defaults to 255.

    Returns:
        grid_fp ((r, c) array): Generated image.
        grid_gt ((r, c, 3) array): Coordinates [x, y, z] of each pixel in the image on the point cloud.
    """

    # remove points where deployment failed
    depth = depth[~np.isnan(uv_points[:, 0])]
    points = points[~np.isnan(uv_points[:, 0]), :]
    uv_points = uv_points[~np.isnan(uv_points[:, 0]), :]

    # translate to image coordinates and sets boundaries.
    min_edge = - size[0] // 2
    # uv_points[:, 0] = uv_points[:, 0] - np.min(uv_points[:, 0]) + edge
    uv_points[:, 1] = uv_points[:, 1] - min_edge #np.min(uv_points[:, 1]) + edge

    # image size
    max_v = size[1]  # np.round(np.max(uv_points[:, 0])) + edge 
    max_u = size[0]  # np.round(np.max(uv_points[:, 1])) + edge
    uv_points[:, 0] = uv_points[:, 0] + (max_v - np.max(uv_points[:, 0])) / 2
 
    depth = 255 - HistEqualization(depth)
    grid_v, grid_u = np.mgrid[1:max_v:1, 1:max_u:1]

    grid_fp = None
    grid_gt = None

    depth_texture = None
    if texture is not None:
        if isinstance(texture, str):
            texture = cv2.imread(texture, cv2.IMREAD_GRAYSCALE)
            # texture = cv2.flip(texture, -1)  # flip 
            H, W = grid_v.shape
            org_center = (texture.shape[1] // 2, texture.shape[0] // 2)
            target_center = (int(W) // 2, int(H) // 2)
            delta_x = target_center[0] - org_center[0]
            delta_y = target_center[1] - org_center[1]
            # crop or pad
            if texture.shape[0] > H:
                start_y = delta_y if delta_y > 0 else 0
                end_y = start_y + int(H)
                texture = texture[start_y:end_y, :]
            else:
                pad_top = delta_y if delta_y > 0 else 0
                pad_bottom = int(H) - texture.shape[0] - pad_top
                texture = cv2.copyMakeBorder(texture, pad_top, pad_bottom, 0, 0, cv2.BORDER_CONSTANT, value=0)
            if texture.shape[1] > W:
                start_x = delta_x if delta_x > 0 else 0
                end_x = start_x + int(W)
                texture = texture[:, start_x:end_x]
            else:
                pad_left = delta_x if delta_x > 0 else 0
                pad_right = int(W) - texture.shape[1] - pad_left
                texture = cv2.copyMakeBorder(texture, 0, 0, pad_left, pad_right, cv2.BORDER_CONSTANT, value=0)
            depth_texture = sample_image_bilinear(texture, uv_points[:, ::-1]) #   uv_points
        if generate_uv_texture:
            grid_fp_texture = np.squeeze(   
            griddata(uv_points,
                     depth_texture, (grid_v, grid_u),
                     method='linear', # 
                     fill_value=0))
            grid_fp_texture = np.uint8(grid_fp_texture)
    return grid_fp, grid_gt, depth_texture, grid_fp_texture if generate_uv_texture else None

def sample_image_bilinear(image, uv):
    H, W = image.shape

    u = uv[:, 0]
    v = uv[:, 1]

    valid = (u >= 0) & (u <= W - 1) & (v >= 0) & (v <= H - 1)

    u0 = np.floor(u).astype(np.int64)
    v0 = np.floor(v).astype(np.int64)
    u1 = u0 + 1
    v1 = v0 + 1

    u0 = np.clip(u0, 0, W - 1)
    u1 = np.clip(u1, 0, W - 1)
    v0 = np.clip(v0, 0, H - 1)
    v1 = np.clip(v1, 0, H - 1)

    Ia = image[v0, u0]
    Ib = image[v1, u0]
    Ic = image[v0, u1]
    Id = image[v1, u1]

    wa = (u1 - u) * (v1 - v)
    wb = (u1 - u) * (v - v0)
    wc = (u - u0) * (v1 - v)
    wd = (u - u0) * (v - v0)

    out = wa * Ia + wb * Ib + wc * Ic + wd * Id
    out[~valid] = 0.0   

    return out


def GenerateAlignmentResult(uv_points,
                            uv_proj_points,
                            proj_mask,
                            size,
                            texture):
    # remove points where deployment failed
    proj_mask = proj_mask[~np.isnan(uv_points[:, 0])]
    uv_points = uv_points[~np.isnan(uv_points[:, 0]), :] 
    uv_proj_points = uv_proj_points[~np.isnan(uv_proj_points[:, 0]), :]

    # standardize the flatten points coordinates aligned with uv_points

    # translate to image coordinates and sets boundaries.
    min_edge = - size[0] // 2
    # uv_proj_points[:, 0] = uv_proj_points[:, 0] #np.min(uv_points[:, 0]) + edge 
    uv_proj_points[:, 1] = uv_proj_points[:, 1] - min_edge #np.min(uv_points[:, 1]) + edge

    # image size
    max_v = size[1] # unified the uv spaces
    max_u = size[0]

    uv_proj_points[:, 0] = uv_proj_points[:, 0] + (max_v - np.max(uv_points[:, 0])) / 2

    grid_v, grid_u = np.mgrid[1:max_v:1, 1:max_u:1]
    uv_proj_points = uv_proj_points[proj_mask, :]
    texture = texture[proj_mask]

    valid_mask = (uv_proj_points[:, 0] > 0) & (uv_proj_points[:, 0] < size[1]) & (uv_proj_points[:, 1] > 0) & (uv_proj_points[:, 1] < size[0])
    uv_proj_points = uv_proj_points[valid_mask, :]
    texture = texture[valid_mask]

    grid_fp_texture = np.squeeze(   
        griddata(uv_proj_points,
                texture, (grid_v, grid_u),
                method='linear', # 
                fill_value=0))
    grid_fp_texture = np.uint8(grid_fp_texture)
   

    return grid_fp_texture