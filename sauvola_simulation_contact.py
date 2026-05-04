import cv2
import numpy as np

def simulate_sauvola_enhancement(image_path, finger_mask_path, block_size=25, k=0.15, R=128):
    """
    Simulate the Sauvola enhancement process for a given fingerprint image and its corresponding finger mask.
    Parameters:
    - image_path: Path to the input fingerprint image (grayscale).
    - finger_mask_path: Path to the corresponding finger mask image (grayscale).
    - block_size: Size of the neighborhood for calculating mean and standard deviation (default: 25).
    - k: Sauvola's k parameter (default: 0.15).
    - R: Dynamic range of standard deviation (default: 128).
    Returns:
    - Enhanced binary image as a NumPy array.
    """
    img = cv2.imread(image_path, 0)
    finger_mask = cv2.imread(finger_mask_path, 0) > 10
    dist_transform = cv2.distanceTransform(finger_mask.astype(np.uint8), cv2.DIST_L2, 5)
    fade_width = dist_transform.max() * 0.1  
    alpha = np.clip(dist_transform / fade_width, 0, 1)

    img[img < 10] = 0  
    if img is None:
        print("Error: Unable to read image at", image_path)
        return None

    img_blurred = cv2.medianBlur(img, 3)
    
    mean = cv2.blur(img_blurred, (block_size, block_size))
    

    img_sq = img_blurred.astype(np.float32)**2
    mean_sq = cv2.blur(img_sq, (block_size, block_size))
    std = np.sqrt(np.maximum(mean_sq - mean.astype(np.float32)**2, 0))

    # Threshold = Mean * (1 + k * (Std / R - 1))
    threshold = mean * (1 + k * (std / R - 1))


    th2_result = np.where(img_blurred > threshold, 255, 0).astype(np.uint8)
    th2_result[~finger_mask.astype(bool)] = 255


    kernel = np.ones((2,2), np.uint8)
    th2_result = cv2.morphologyEx(th2_result, cv2.MORPH_OPEN, kernel)

    th2_result = (th2_result.astype(np.float32) * alpha + 255 * (1 - alpha)).astype(np.uint8)
    return th2_result

import os
folder = "example/source_rolled"
mask_folder = "example/mask"
save_folder = "example/binary_enhancement"
for root, dirs, files in os.walk(folder):
    for file in files:
        input_file = os.path.join(root, file)
        mask_file = os.path.join(mask_folder, file)
        result = simulate_sauvola_enhancement(input_file, mask_file, block_size=11, k=0.007) # k = 0.12
        if result is not None:
            # save the result
            relative_path = os.path.relpath(root, folder)
            target_dir = os.path.join(save_folder, relative_path)
            os.makedirs(target_dir, exist_ok=True)
            save_path = os.path.join(target_dir, file)
            cv2.imwrite(save_path, result)
