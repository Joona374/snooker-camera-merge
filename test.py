import cv2
import numpy as np
import time

def click_points(event, x, y, flags, params):
    """Mouse callback to capture points on an image."""
    image, points = params
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        cv2.circle(image, (x, y), 5, (0, 255, 0), -1)
        cv2.imshow("Image", image)

def collect_points(img, prompt):
    """Display an image and collect 4 points from user clicks."""
    points = []
    temp_img = img.copy()
    cv2.imshow("Image", temp_img)
    cv2.setMouseCallback("Image", click_points, (temp_img, points))
    print(f"Please click the 4 points for: {prompt}")
    while len(points) < 4:
        cv2.waitKey(1)
    cv2.destroyWindow("Image")
    return np.array(points, dtype=np.float32)

time.sleep(3)

# Load your two images
img1 = cv2.imread("left_cam.jpg")
img2 = cv2.imread("right_cam.jpg")
if not img2 or img1:
    print(f"Images not there yet")


# Step 1: Collect corresponding points
pts1 = collect_points(img1, "Left Camera (Top-Left, Top-Middle, Bottom-Left, Bottom-Middle)")
pts2 = collect_points(img2, "Right Camera (Top-Left, Top-Middle, Bottom-Left, Bottom-Middle)")

# Step 2: Compute perspective transform to align img2 to img1
H = cv2.getPerspectiveTransform(pts2, pts1)
warped = cv2.warpPerspective(img2, H, (img1.shape[1], img1.shape[0]))

# Step 3: Compute seam line between the mid-pocket points in img1
# Points 1 and 3 correspond to top-middle and bottom-middle
(x1, y1), (x3, y3) = pts1[1], pts1[3]
# For each row, calculate x-position of the seam
rows = img1.shape[0]
seam_x = np.array([x1 + (y - y1) * (x3 - x1) / (y3 - y1) for y in range(rows)], dtype=np.int32)

# Step 4: Create composite image
composite = np.zeros_like(img1)
for y in range(rows):
    sx = seam_x[y]
    composite[y, :sx] = img1[y, :sx]
    composite[y, sx:] = warped[y, sx:]

# Step 5: Display result
cv2.imshow("Stitched Composite", composite)
print("Stitching complete. Press any key in the image window to exit.")
cv2.waitKey(0)
cv2.destroyAllWindows()
