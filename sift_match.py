"""
Functionality for attempting to match a jigsaw piece onto a base image.

Please `import` and use the `find_match` method.

When run on its own, the program will attempt to check its accuracy on all images in the dataset.

Results for `check_accuracy()` run on Hydra with 8 CPUs and 16G memory:
Finished in 9996.598049879074 seconds with accuracy 0.510655737704918
Got 623 / 1220 with 2 false positives.
Average processing time per piece: 8.193932827769732 seconds.
"""
import base64

import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import os
import time

RATIO_FOR_MATCH = 0.75
MATCH_CUTOFF = 10
DATASET_PATH = "JigsawDataset/"
CHECKING_CORRECT_MARGIN = 0.03  # 3% of image size


# Note that this algorithm is unlikely to randomly place the jigsaw piece on the base,
# so the margin shouldn't need to be extremely low.


# noinspection ProblematicWhitespace
def find_match(base, piece, save_image, request_time):
    """
	Attempt to find a match of `piece` within `base`.

	`base` and `piece` should be images compatible with OpenCV,
	i.e. should be opened with `cv2.imread(path, cv2.IMREAD_UNCHANGED)`

	`save_image` is a boolean to determine if a visualisation of the piece on the base should be saved.

	Returns a NumPy array containing the where the corners of the piece image are on the base piece as a percentage (range 0 to 1), or `None` if a match was not found.

	Note that the points it returns are the corners of the image, not the piece itself.
	"""

    base_h, base_w, base_c = base.shape
    piece_h, piece_w, piece_c = piece.shape

    sift = cv2.SIFT_create()

    # Keypoints and descriptors for both images
    kp_base, desc_base = sift.detectAndCompute(base, None)
    kp_piece, desc_piece = sift.detectAndCompute(piece, None)

    # Use bruce force matcher to ensure best match
    bf = cv2.BFMatcher.create(cv2.NORM_L2)
    matches = bf.knnMatch(desc_piece, desc_base, k=2)

    # Loop through all matches and only keep the ones that we are sufficiently confidient are correct.
    # This is achieve through a "ratio test" by comparing the distance between the two matched features.
    # see Section 7.1. Keypoint matching: https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf
    good = []
    for (m, n) in matches:
        if m.distance < RATIO_FOR_MATCH * n.distance:
            good.append(m)

    # Stop if number of matches is below cutoff
    # i.e. features are aligned, but there are too few of them to be sure.
    if len(good) <= MATCH_CUTOFF:
        return

    # https://docs.opencv.org/3.4/d1/de0/tutorial_py_feature_homography.html
    # Convert keypoint indexes into pixel points using list comprehension.
    # Reshape is needed to add an extra (empty) dimension to the data (from 2D points to 3D) so it may be used in functions that require 3D data.
    piece_points = np.float32([kp_piece[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    base_points = np.float32([kp_base[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    # Find the perspective transform to turn the piece's points into their position on the base.
    # Save the matrix only.
    # Parameters taken from https://docs.opencv.org/3.4/d1/de0/tutorial_py_feature_homography.html
    matrix, _ = cv2.findHomography(piece_points, base_points, cv2.RANSAC, 5.0)

    # Get an array of the corners of the piece image.
    # Reshape is needed for same reason as earlier.
    corners = np.float32([
        [0, 0],
        [0, piece_h - 1],
        [piece_w - 1, piece_h - 1],
        [piece_w - 1, 0]
    ]).reshape(-1, 1, 2)

    # Move the corners of the image to their place on the base image.
    transformed_points = cv2.perspectiveTransform(corners, matrix)

    if save_image:
        # Create an alpha channel in the base if it doesn't exist
        if piece_c != 4:
            piece = np.dstack((piece, np.ones((piece_h, piece_w), "uint8") * 255))
        if base_c != 4:
            base = np.dstack((base, np.ones((base_h, base_w), "uint8") * 255))
        
        # Increase piece image size to match base. Pad with black pixels.
        piece_transformed = cv2.copyMakeBorder(piece,
            0, int(base_h - piece_h), 0, int(base_w - piece_w),
            cv2.BORDER_CONSTANT, value=[0,0,0,0]
        )

        # Apply matrix transformation to move piece to correct position on base.
        # Handles position, rotation, scale, and perspective skews.
        piece_transformed = cv2.warpPerspective(piece_transformed, matrix, (base_w, base_h))

        # Combine both the piece and the base.
        # Convert to float for division to work properly.
        piece_transformed = piece_transformed.astype(float)
        base = base.astype(float)

        # Use alpha from the jigsaw piece.
        alpha = piece_transformed[:, :, 3].astype(float) / 255.0

        # Reshape alpha array to match shape of base.
        alpha = np.array(
            [alpha,
             alpha,
             alpha,
             alpha]
        ).transpose((1, 2, 0))

        # Cut out a transparent hole for the jigsaw piece from the base image using its alpha.
        base = cv2.multiply(1 - alpha, base)

        # Combine the jigsaw piece with the base image.
        final = cv2.add(base, piece_transformed)
        solved_png_path = "output/" + str(request_time) + "_solved.jpg"
        cv2.imwrite(solved_png_path, final)

        # Convert to webp
        solved_webp = Image.open(solved_png_path)
        solved_webp = solved_webp.convert("RGB")
        solved_webp_path = "output/" + str(request_time) + "_solved.webp"
        solved_webp.save(solved_webp_path, "webp")

        # Delete the PNG file
        try:
            os.remove(solved_png_path)
        except:
            pass

        # Encode to base64 and return
        return encode_base64(solved_webp_path)

    # Return the found points as a percent of the base's size
    # (same format as testing data)
    return transformed_points / np.array([base_w, base_h])

def encode_base64(file):
    """
    Encode to base64
    :param file: file to encode
    :return: encoded file
    """
    fh = open(file, "rb")
    out_bytes = fh.read()
    fh.close()

    out_bytes_base64 = base64.b64encode(out_bytes)
    out_bytes_base64 = out_bytes_base64.decode("ascii")

    return out_bytes_base64

def check_accuracy():
    """
	Perform a check of all images in the dataset folder to assess accuracy.
	"""

    data = pd.read_csv(DATASET_PATH + "data.csv")
    total = 0
    correct = 0
    false_positives = 0
    start_time = time.time()

    # For each row in the dataset...
    for index, row in data.iterrows():

        # Check if the image paths exist, if they don't, skip this row.
        base_path = DATASET_PATH + row["base_path"]
        piece_path = DATASET_PATH + row["piece_id"] + ".png"
        if not os.path.exists(base_path):
            continue
        if not os.path.exists(piece_path):
            continue

        # Load the images
        piece = cv2.imread(piece_path, cv2.IMREAD_UNCHANGED)
        base = cv2.imread(base_path, cv2.IMREAD_UNCHANGED)

        # Attempt to match the points
        match_points = find_match(base, piece, False)
        total += 1

        # If the match failed, move to next row.
        if match_points is None:
            continue

        # Accuracy is assessed by the distance between the center points of the image and the piece.
        # This is because the coords in data.csv are the coords of the piece's corners, not the image's corners.
        # As the piece should be centered within the image, the center coords should be similar.
        match_center_point = np.average(match_points, 0)

        # Get the target points from the CSV and then calculate the center.
        actual_points = np.array([
            [row["lower_left_x"], 1 - row["lower_left_y"]],
            [row["top_left_x"], 1 - row["top_left_y"]],
            [row["top_right_x"], 1 - row["top_right_y"]],
            [row["bottom_right_x"], 1 - row["bottom_right_y"]]
        ])
        actual_center_point = np.average(actual_points, 0)

        # Compute the distance between the two points.
        distance = np.linalg.norm(actual_center_point - match_center_point)

        # If it is sufficiently small, mark the match as correct, otherwise, it is a false positive.
        # (False positives are much worse that simply not making a match and should be kept as low as possible.)
        if distance <= CHECKING_CORRECT_MARGIN:
            correct += 1
        else:
            # Matched, but in incorrect location
            false_positives += 1

    # Output results to the user.
    finished_time = time.time() - start_time
    print("Finished in " + str(finished_time) + " seconds with accuracy " + str(correct / total))
    print("Got " + str(correct) + " / " + str(total) + " with " + str(false_positives) + " false positives.")
    print("Average processing time per piece: " + str(finished_time / total) + " seconds.")


if __name__ == "__main__":
    check_accuracy()
