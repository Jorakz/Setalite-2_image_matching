import cv2
import numpy as np
import matplotlib.pyplot as plt


def load_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")
    return image


def detect_keypoints_and_descriptors(image):
    orb = cv2.ORB_create()
    keypoints, descriptors = orb.detectAndCompute(image, None)
    if keypoints is None or descriptors is None:
        raise ValueError("No keypoints or descriptors found")
    return keypoints, descriptors

def match_keypoints(descriptors1, descriptors2):
    if descriptors1 is None or descriptors2 is None:
        return []

    if len(descriptors1) == 0 or len(descriptors2) == 0:
        return []

    descriptors1 = np.float32(descriptors1)
    descriptors2 = np.float32(descriptors2)

    # Используем BFMatcher с нормой L2, так как SIFT использует эту норму
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)


    matches = bf.knnMatch(descriptors1, descriptors2, k=2)


    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    return good_matches
def match_keypoints1(descriptors1, descriptors2):
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)
    if not matches:
        raise ValueError("No matches found")
    matches = sorted(matches, key=lambda x: x.distance)
    return matches


def draw_matches(image1, keypoints1, image2, keypoints2, matches):
    height1, width1 = image1.shape
    height2, width2 = image2.shape
    result_image = np.zeros((max(height1, height2), width1 + width2, 3), dtype=np.uint8)

    result_image[:height1, :width1] = cv2.cvtColor(image1, cv2.COLOR_GRAY2BGR)
    result_image[:height2, width1:width1 + width2] = cv2.cvtColor(image2, cv2.COLOR_GRAY2BGR)

    for match in matches:
        pt1 = tuple(np.round(keypoints1[match.queryIdx].pt).astype(int))
        pt2 = tuple(np.round(keypoints2[match.trainIdx].pt).astype(int) + np.array([width1, 0]))

        color = tuple(np.random.randint(10, 255, 3).tolist())

        cv2.circle(result_image, pt1, 7, color, 4)
        cv2.circle(result_image, pt2, 7, color, 4)
        cv2.line(result_image, pt1, pt2, color, 5)

    return result_image


def resize_image(image, target_shape):
    return cv2.resize(image, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_LINEAR)


def process_images(image_path1, image_path2):
    try:
        image1 = load_image(image_path1)
        image2 = load_image(image_path2)

        if image1.shape[0] * image1.shape[1] < image2.shape[0] * image2.shape[1]:
            image1 = resize_image(image1, image2.shape)
        else:
            image2 = resize_image(image2, image1.shape)

        keypoints1, descriptors1 = detect_keypoints_and_descriptors(image1)
        keypoints2, descriptors2 = detect_keypoints_and_descriptors(image2)

        matches = match_keypoints(descriptors1, descriptors2)
        result_image = draw_matches(image1, keypoints1, image2, keypoints2, matches)

        return result_image

    except Exception as e:
        print(f"Error processing images: {e}")
        return None



