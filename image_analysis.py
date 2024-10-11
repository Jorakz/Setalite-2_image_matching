import cv2
import numpy as np


def load_image(image_path):
    """Load an image in .jp2 format."""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")
    return image


def detect_keypoints_and_descriptors(image):
    """Detect key points and descriptors using ORB."""
    orb = cv2.ORB_create()
    keypoints, descriptors = orb.detectAndCompute(image, None)
    return keypoints, descriptors


def match_keypoints(descriptors1, descriptors2):
    """Find matching key points between two images."""

    descriptors1 = np.float32(descriptors1)
    descriptors2 = np.float32(descriptors2)

    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)

    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    return good_matches


import cv2
import numpy as np


def load_image(image_path):
    """Load an image in .jp2 format."""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")
    return image


def detect_keypoints_and_descriptors(image):
    """Detect key points and descriptors using ORB."""
    orb = cv2.ORB_create()
    keypoints, descriptors = orb.detectAndCompute(image, None)
    return keypoints, descriptors


def match_keypoints(descriptors1, descriptors2):
    """Find matching key points between two images using BFMatcher."""
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)
    matches = sorted(matches, key=lambda x: x.distance)
    return matches


def draw_matches(image1, keypoints1, image2, keypoints2, matches):
    """Draw lines between matching key points with bright colors."""
    # Limit the number of matches to display
    matches = matches[:len(matches)]

    # Create an empty canvas for drawing
    height1, width1 = image1.shape
    height2, width2 = image2.shape
    result_image = np.zeros((max(height1, height2), width1 + width2, 3), dtype=np.uint8)

    # Add images to the common canvas
    result_image[:height1, :width1] = cv2.cvtColor(image1, cv2.COLOR_GRAY2BGR)
    result_image[:height2, width1:width1 + width2] = cv2.cvtColor(image2, cv2.COLOR_GRAY2BGR)

    # Draw matches with bright colored lines
    for match in matches:
        # Coordinates of key points
        pt1 = tuple(np.round(keypoints1[match.queryIdx].pt).astype(int))
        pt2 = tuple(np.round(keypoints2[match.trainIdx].pt).astype(int) + np.array([width1, 0]))

        # Generate a bright color for each line (bright shades)
        color = tuple(np.random.randint(10, 255, 3).tolist())  # Minimum 100 for brightness

        # Draw circles and lines between matches
        cv2.circle(result_image, pt1, 7, color, 4)
        cv2.circle(result_image, pt2, 7, color, 4)
        cv2.line(result_image, pt1, pt2, color, 5)

    return result_image


def process_images(image_path1, image_path2):
    """Main function for processing images and finding key points."""
    # Load images
    image1 = load_image(image_path1)
    image2 = load_image(image_path2)

    # Detect key points and descriptors
    keypoints1, descriptors1 = detect_keypoints_and_descriptors(image1)
    keypoints2, descriptors2 = detect_keypoints_and_descriptors(image2)

    # Match key points
    matches = match_keypoints(descriptors1, descriptors2)

    # Visualize matches
    result_image = draw_matches(image1, keypoints1, image2, keypoints2, matches)

    return result_image



