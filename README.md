# Setalite-2_image_matching
## Overview

The project appears to involve comparing satellite images taken at different times (or from different sensors) to identify keypoints and matching features between them. This could be useful for a variety of tasks such as:

Change detection: Identifying changes in landscape, urban growth, vegetation, etc.
Image alignment: Aligning satellite images taken in different conditions (e.g., different lighting or seasons).
The images are loaded in grayscale, keypoints are detected using ORB, descriptors are matched using a brute-force approach, and matches are visualized.

## Database

Deforestation in Ukraine from Sentinel2 data
This dataset was developed by the Quantum in collaboration with V. N. Karazin Kharkiv National University. Dataset is used in Deep Learning for Regular Change Detection in Ukrainian Forest Ecosystem With Sentinel-2 paper. The main idea of this dataset is to identify the regions of deforestation in Kharkiv, Ukraine from the Sentinel2 satellite imagery. Can be used to solve Semantic Segmentation problem on high-resolution data. (https://www.kaggle.com/datasets/isaienkov/deforestation-in-ukraine)

## Usage

* data_analyze.py: algorithm for redaction data and matching satellite images.
* data_main.py: Runs algorithm for image analysis.
* requirements.txt: Lists the required libraries for the project.
* data_demo.ipynb: Jupyter notebook with demo.
  
## Funtion 

* load_image(image_path):

This function reads an image from a specified file path using OpenCV's cv2.imread() with grayscale mode (cv2.IMREAD_GRAYSCALE).
It checks whether the image was successfully loaded and raises an error if it fails.
Returns the loaded grayscale image.

* detect_keypoints_and_descriptors(image):

Detects keypoints (distinctive features in the image) and computes descriptors (a set of values describing each keypoint) using the ORB (Oriented FAST and Rotated BRIEF) algorithm.
ORB is a fast and efficient algorithm commonly used for feature detection in tasks like object recognition, image stitching, and satellite image matching.
Raises an error if no keypoints or descriptors are found.
Returns the keypoints and descriptors for the image.

* match_keypoints(descriptors1, descriptors2):

Matches the descriptors of the two images using a brute-force matcher (BFMatcher) with the Hamming distance as a similarity metric. The crossCheck=True ensures that only mutually consistent matches are kept.
Sorts the matches by their distance (closer matches are more similar).
Raises an error if no matches are found.
Returns a list of the best matches between the two images.
In code, KNN (K-Nearest Neighbors) refers to the algorithm used to find the best matches between feature descriptors extracted from two satellite images. 

* draw_matches(image1, keypoints1, image2, keypoints2, matches):

This function visually illustrates the matches between the two images. It creates a new image that combines both input images side by side and draws lines between corresponding matched keypoints.
The color and thickness of the circles and lines are randomized to make the matches visually distinctive.
Returns the combined image with the drawn matches.'

* resize_image(image, target_shape):

Resizes the input image to match the shape of a target image using linear interpolation. This ensures that both images are of the same size before feature matching.
This is particularly useful for satellite images, which might have different resolutions depending on the sensor or acquisition conditions.

* process_images(image_path1, image_path2):

This is the main function that coordinates the entire process. It:
Loads the two images from their file paths.
Resizes the images to have the same dimensions if needed.
Detects keypoints and computes descriptors for both images.
Matches the descriptors from both images.
Draws the matches on a combined image.

## Conclusion
This project implements a keypoint detection and matching system for satellite imagery using OpenCV's ORB algorithm. The core functionality involves loading grayscale images, detecting keypoints, and matching descriptors using brute-force methods. It also visualizes the matches between two images, highlighting their similarities. This approach can be applied to tasks such as satellite image alignment, change detection, and temporal analysis of geographic areas.

The workflow includes:

* Loading satellite images (.jp2 format).
* Resizing images to a consistent size for better keypoint matching.
* Detecting keypoints and computing ORB descriptors.
* Matching keypoints between two images.
* Visualizing the matched keypoints with randomly colored lines for clarity.
This pipeline is valuable for comparing satellite data, such as analyzing environmental changes or mapping geographic shifts over time.
