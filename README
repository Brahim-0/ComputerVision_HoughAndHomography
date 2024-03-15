# Introduction

This repository contains Python scripts for implementing computer vision algorithms. The scripts provided here offer solutions to various problems without relying on advanced libraries like skimage or OpenCV, aiming to demonstrate proficiency in algorithmic implementation.

## Hough Transform Implementation (Part 1)

### Overview

The script `hough.py` implements the Hough transform algorithm to detect straight lanes in an image. The process involves several steps:

1. **Edge Detection**: The image "road.jpg" is loaded and converted to grayscale. Canny edge detection is then applied to identify edges.

2. **Region of Interest (ROI)**: A binary mask is created to select a subset of edge points within a region of interest (ROI). This is achieved using the provided function `create_mask` in `utils.py`.

3. **Hough Transform**: The Hough transform is implemented using the polar representation (ρ,θ) as the parameter space. The algorithm identifies two major lanes by finding cells with the highest values in the Hough space.

4. **Non-Maximum Suppression (NMS)**: Non-maximum suppression is applied to suppress the values of neighboring cells in the Hough space. This ensures accurate detection of lanes.

### Usage

To run the Hough transform implementation:

```bash
python hough.py

## RANSAC and Homography (Part 2)

### Overview

The script `homography.py` implements RANSAC for estimating the homography between two images. It also applies the estimated homography in a simple augmented reality (AR) application.

- **Feature Matching**: SIFT features are extracted from the input images using existing libraries. Candidate matching points between the images are then identified.

- **RANSAC Algorithm**: RANSAC is implemented to estimate the homography matrix while handling outliers in the matching points. This is achieved using only Numpy for computation, without relying on external libraries.

- **Homography Application**: The estimated homography is used to warp one image and composite it onto another image. This creates a visually appealing AR effect where one image covers another seamlessly.

### Usage

To run the RANSAC and homography implementation:

```bash
python homography.py

The script will produce visualizations of the raw matching result, the result after RANSAC, and the final composite image.

### Conclusion

These scripts demonstrate proficiency in implementing computer vision algorithms from scratch, including the Hough transform and RANSAC for homography estimation. They provide valuable insights into fundamental concepts and techniques used in computer vision applications.
