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

This function loads a grayscale image from a specified file path (in .jp2 format). If the image fails to load, it raises a ValueError.

* detect_keypoints_and_descriptors(image):

It detects keypoints and descriptors from an image using the ORB (Oriented FAST and Rotated BRIEF) algorithm.

* match_keypoints(descriptors1, descriptors2):

This function matches keypoints between two images using a brute-force matcher (BFMatcher) with Hamming distance, and sorts the matches based on their distance (i.e., similarity).

* draw_matches(image1, keypoints1, image2, keypoints2, matches):

Draws lines between the matched keypoints from two images. It creates a composite image where both images are placed side by side, and colorful lines are drawn between the matching points.

* process_images(image_path1, image_path2):

This is the main function to process two images. It loads them, detects keypoints and descriptors, matches the keypoints, and then visualizes the matches by drawing lines between the corresponding keypoints.

## Conclusion
This project implements a keypoint detection and matching system for satellite imagery using OpenCV's ORB algorithm. The core functionality involves loading grayscale images, detecting keypoints, and matching descriptors using brute-force methods. It also visualizes the matches between two images, highlighting their similarities. This approach can be applied to tasks such as satellite image alignment, change detection, and temporal analysis of geographic areas.

The workflow includes:

* Loading satellite images (.jp2 format).
* Resizing images to a consistent size for better keypoint matching.
* Detecting keypoints and computing ORB descriptors.
* Matching keypoints between two images.
* Visualizing the matched keypoints with randomly colored lines for clarity.
This pipeline is valuable for comparing satellite data, such as analyzing environmental changes or mapping geographic shifts over time.
