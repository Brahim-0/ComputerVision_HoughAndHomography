import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import feature
from utils import create_line, create_mask

# Step 1: Load the image and convert it to grayscale
image = cv2.imread('road.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Step 2: Run Canny edge detector to find edges
edges = feature.canny(gray, sigma=1.05)

# Step 2b: Create a mask for the region of interest
mask = create_mask(*edges.shape)
plt.imshow(mask, cmap='gray')
plt.title('Mask')


# Extract edge points within ROI
edges_roi = edges * mask

# Step 3: Perform Hough transform
H, W = edges_roi.shape
max_rho = int(np.ceil(np.sqrt(H**2 + W**2)))  # Max possible distance from origin to a point
theta_range = np.deg2rad(np.arange(-70, 55))  # Restrict theta range
acc = np.zeros((2 * max_rho, len(theta_range)))  # Initialize accumulator

# Voting
edge_points = np.argwhere(edges_roi)
for y, x in edge_points:
    for theta_idx, theta in enumerate(theta_range):
        rho = int(round(x * np.cos(theta) + y * np.sin(theta))) + max_rho
        acc[rho, theta_idx] += 1

# Find the peak in the Hough space for the right lane
max_idx = np.argmax(acc)
rho_idx, theta_idx = np.unravel_index(max_idx, acc.shape)
rho = rho_idx - max_rho
theta = theta_range[theta_idx]

# Create the right lane line
xs_right, ys_right = create_line(rho, theta, edges_roi)

# Apply non-maximum suppression to suppress values around the peak
acc[rho_idx-50:rho_idx+50, theta_idx-5:theta_idx+5] = 0


# Find the peak for the left lane
max_idx = np.argmax(acc)
rho_idx, theta_idx = np.unravel_index(max_idx, acc.shape)
rho = rho_idx - max_rho
theta = theta_range[theta_idx]

# Create the left lane line
xs_left, ys_left = create_line(rho, theta, edges_roi)

# Plot the results
plt.figure(figsize=(10, 8))

plt.subplot(2, 2, 1)
plt.imshow(gray, cmap='gray')
plt.title('Grayscale Image')

plt.subplot(2, 2, 2)
plt.imshow(edges, cmap='gray')
plt.title('Canny Edge Detection')

plt.subplot(2, 2, 3)
plt.imshow(edges_roi, cmap='gray')
plt.title('Edges in ROI')

plt.subplot(2, 2, 4)
plt.imshow(image)
plt.plot(xs_right, ys_right, color='blue', linewidth=3)
plt.plot(xs_left, ys_left, color='orange', linewidth=3)
plt.title('Detected Lanes')

plt.tight_layout()
plt.show()

